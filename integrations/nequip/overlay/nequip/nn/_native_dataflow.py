"""Native irrep dataflow without per-slice zero-fill adjoints.

The original e3nn Linear parameters, paths and normalization remain live.
A single split adjoint concatenates feature blocks, rather than zero-filling
the full feature tensor for each independent slice. No capture or weight cache.
"""
import os

import torch


def enabled():
    return os.environ.get("NEQUIP_CHORUS_NATIVE_DATAFLOW", "1") != "0"


def _plan(module):
    cached = getattr(module, "_chorus_dataflow_plan", None)
    if cached is not None:
        return cached
    if (not getattr(module, "internal_weights", False) or
            not getattr(module, "shared_weights", False) or
            getattr(module, "bias_numel", 0) != 0 or module.weight.ndim != 1):
        return None
    inputs = tuple((int(mul), int(ir.dim)) for mul, ir in module.irreps_in)
    outputs = tuple((int(mul), int(ir.dim)) for mul, ir in module.irreps_out)
    paths = []
    offset = 0
    for ins in module.instructions:
        if ins.i_in < 0 or ins.i_out < 0:
            return None
        cin, dim = inputs[ins.i_in]
        cout, out_dim = outputs[ins.i_out]
        if dim != out_dim or tuple(ins.path_shape) != (cin, cout):
            return None
        paths.append((int(ins.i_in), int(ins.i_out), cin, cout, dim, float(ins.path_weight)))
        offset += cin * cout
    if offset != module.weight.numel() or not paths:
        return None
    cached = (inputs, outputs, tuple(paths))
    module._chorus_dataflow_plan = cached
    return cached


def linear(module, value):
    """Same channel maps as an internal/shared/bias-free e3nn Linear."""
    if (not value.is_cuda or torch.jit.is_scripting() or
            module._forward_hooks or module._forward_pre_hooks or
            module._backward_hooks or module._backward_pre_hooks):
        return module(value)
    plan = _plan(module)
    if plan is None:
        return module(value)
    inputs, outputs, paths = plan
    leading = value.shape[:-1]
    value = value.reshape(-1, value.shape[-1])
    n = value.shape[0]
    sections = torch.split(value, [c*m for c,m in inputs], dim=-1)
    weights = torch.split(module.weight, [cin*cout for _,_,cin,cout,_,_ in paths])
    angular = [part.reshape(n,c,m).transpose(1,2).reshape(n*m,c)
               for part,(c,m) in zip(sections,inputs)]
    mapped = [[] for _ in outputs]
    for (source, target, cin, cout, dim, scale), weight in zip(paths, weights):
        matrix = weight.reshape(cin,cout)
        # Normalize the parameter-sized matrix, not an edge-sized activation.
        result = angular[source] @ (matrix * scale)
        result = result.reshape(n,dim,cout).transpose(1,2).reshape(n,cout*dim)
        mapped[target].append(result)
    blocks = []
    for values,(cout,dim) in zip(mapped,outputs):
        if not values:
            blocks.append(value.new_zeros(n,cout*dim))
        elif len(values)==1:
            blocks.append(values[0])
        else:
            blocks.append(torch.stack(values,dim=0).sum(dim=0))
    return torch.cat(blocks,dim=-1).reshape(*leading,sum(c*m for c,m in outputs))
