"""Locate nonfinite values on exactly the failed checkpoint/graph, unchanged."""
import argparse
import json
from pathlib import Path

import torch
from chorus.hp import configure_model
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from chorus.synthetic import compute_energy_forces, make_fixed_graph


def stats(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        finite = torch.isfinite(value)
        return {"shape": list(value.shape), "nonfinite": int((~finite).sum()),
                "max_abs_finite": float(value[finite].abs().max()) if bool(finite.any()) else None}
    if isinstance(value, (tuple, list)):
        return [stats(v) for v in value]
    if isinstance(value, dict):
        return {key: stats(val) for key, val in value.items()}
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LAMMPS_MLIAP_MFF.from_checkpoint(
        args.checkpoint, element_types=["H", "C", "N", "O"], device="cuda").wrapper.model
    model.eval().requires_grad_(False)
    graph = make_fixed_graph(num_nodes=128, avg_degree=24, dtype=torch.float32,
                             device=torch.device("cuda"), seed=20261017)
    graph = (graph[0]*0.4,) + graph[1:]
    results = {}
    for mode in ("reference", "hp_without_density_fusion", "hp"):
        configure_model(model, policy="reference" if mode == "reference" else "generated",
                         density_fusion=mode != "hp_without_density_fusion",
                         symmetric_contraction=False)
        trace, handles = [], []
        def before(name):
            def hook(module, args, kwargs):
                trace.append({"module": name, "input": stats(kwargs)})
            return hook
        def after(name):
            def hook(module, args, value):
                trace.append({"module": name, "output": stats(value)})
            return hook
        for name, module in model.named_modules():
            if name.startswith(("interactions.", "products.")) and name.count(".") == 1:
                handles.append(module.register_forward_pre_hook(before(name), with_kwargs=True))
                handles.append(module.register_forward_hook(after(name)))
        values = compute_energy_forces(model, graph, create_graph=False)
        results[mode] = {"outputs": stats(values), "trace": trace}
        for handle in handles:
            handle.remove()
    args.out.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
