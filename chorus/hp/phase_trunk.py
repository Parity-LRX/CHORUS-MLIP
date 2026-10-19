"""Factor the edge-context affine map into node GEMMs and fused edge gathers.

W[destination_context,source_context,radial] is the original trainable matrix.
Node projections are performed once per node, not once per incident edge.
The fused gather/add/normalized-SiLU has an exact differentiable VJP fallback.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _context_affine(D, S, R, W, B, SRC, DST, G, Y, DD, DS, DR,
                    E: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                    SCALE: tl.constexpr, BACKWARD: tl.constexpr,
                    NEED_D: tl.constexpr, NEED_S: tl.constexpr, NEED_R: tl.constexpr,
                    BLOCK_E: tl.constexpr):
    edge = tl.program_id(0)*BLOCK_E+tl.arange(0, BLOCK_E)
    channel = tl.arange(0, triton.next_power_of_2(H))
    valid = (edge[:, None] < E) & (channel[None, :] < H)
    src = tl.load(SRC+edge, edge < E, 0)
    dst = tl.load(DST+edge, edge < E, 0)
    radial = tl.full((BLOCK_E, triton.next_power_of_2(H)), 0., tl.float32)
    for k in range(K):
        rv = tl.load(R+edge*K+k, edge < E, 0.)
        weight = tl.load(W+channel*K+k, channel < H, 0.)
        radial = tl.fma(rv[:, None], weight[None, :], radial)
    radial += tl.load(B+channel, channel < H, 0.)[None, :]
    value = tl.load(D+dst[:, None]*H+channel[None, :], valid, 0.)
    value += tl.load(S+src[:, None]*H+channel[None, :], valid, 0.)
    value += radial
    denominator = 1.+libdevice.exp(-value)
    sigmoid = tl.div_rn(1., denominator)
    if BACKWARD:
        grad = tl.load(G+edge[:, None]*H+channel[None, :], valid, 0.)
        grad = grad*SCALE*sigmoid*(1.+value*(1.-sigmoid))
        if NEED_D:
            tl.atomic_add(DD+dst[:, None]*H+channel[None, :], grad, valid, sem="relaxed")
        if NEED_S:
            tl.atomic_add(DS+src[:, None]*H+channel[None, :], grad, valid, sem="relaxed")
        if NEED_R:
            for k in range(K):
                weight = tl.load(W+channel*K+k, channel < H, 0.)
                dr = tl.sum(grad*weight[None, :], axis=1)
                tl.store(DR+edge*K+k, dr, edge < E)
    else:
        tl.store(Y+edge[:, None]*H+channel[None, :], tl.div_rn(value, denominator)*SCALE, valid)


class _FrozenContextSiLU(torch.autograd.Function):
    """Fold the small radial affine map into the edge-context activation.

    Only frozen radial-affine parameters are eligible. Live node/radial
    coordinates and all their mixed/higher derivatives remain differentiable.
    """
    @staticmethod
    def forward(ctx, d, s, r, weight, bias, src, dst, scale):
        if weight.requires_grad or bias.requires_grad:
            raise ValueError("Context-affine fusion requires frozen weight and bias")
        ctx.save_for_backward(d, s, r, weight, bias, src, dst)
        ctx.scale = scale
        out = r.new_empty(r.shape[0], weight.shape[0])
        if r.shape[0]:
            _context_affine[(triton.cdiv(r.shape[0], 4),)](
                d, s, r, weight, bias, src, dst, r, out, d, s, r,
                r.shape[0], weight.shape[0], r.shape[1], scale, False,
                False, False, False, 4, enable_fp_fusion=False)
        return out

    @staticmethod
    def backward(ctx, grad):
        d, s, r, weight, bias, src, dst = ctx.saved_tensors
        from mace_ictc.models.adjoint_requirements import required_gradients
        needed = required_gradients((d, s, r), ctx.needs_input_grad[:3])
        if torch.is_grad_enabled():
            value = d[dst]+s[src]+F.linear(r, weight, bias)
            sigmoid = value.sigmoid()
            dy = grad*ctx.scale*sigmoid*(1.+value*(1.-sigmoid))
            dd = torch.zeros_like(d).index_add(0, dst, dy) if needed[0] else None
            ds = torch.zeros_like(s).index_add(0, src, dy) if needed[1] else None
            dr = dy@weight if needed[2] else None
        else:
            dd = torch.zeros_like(d) if needed[0] else None
            ds = torch.zeros_like(s) if needed[1] else None
            dr = torch.empty_like(r) if needed[2] else None
            if r.shape[0] and any(needed):
                _context_affine[(triton.cdiv(r.shape[0], 4),)](
                    d, s, r, weight, bias, src, dst, grad.contiguous(), r,
                    dd if needed[0] else d, ds if needed[1] else s, dr if needed[2] else r,
                    r.shape[0], weight.shape[0], r.shape[1], ctx.scale, True,
                    *needed, 4, enable_fp_fusion=False)
        return dd, ds, dr, None, None, None, None, None


@triton.jit
def _gather(D, S, R, SRC, DST, G, Y, DD, DS, DR, TOTAL: tl.constexpr,
            H: tl.constexpr, SCALE: tl.constexpr, BACKWARD: tl.constexpr,
            NEED_D: tl.constexpr, NEED_S: tl.constexpr, NEED_R: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < TOTAL
    edge, channel = i//H, i%H
    src, dst = tl.load(SRC+edge, valid, 0), tl.load(DST+edge, valid, 0)
    value = tl.load(D+dst*H+channel, valid, 0.) + tl.load(S+src*H+channel, valid, 0.)
    value = value + tl.load(R+i, valid, 0.)
    denominator = 1.+libdevice.exp(-value)
    sigmoid = tl.div_rn(1., denominator)
    if BACKWARD:
        grad = tl.load(G+i, valid, 0.)*SCALE*sigmoid*(1.+value*(1.-sigmoid))
        if NEED_D:
            tl.atomic_add(DD+dst*H+channel, grad, valid, sem="relaxed")
        if NEED_S:
            tl.atomic_add(DS+src*H+channel, grad, valid, sem="relaxed")
        if NEED_R:
            tl.store(DR+i, grad, valid)
    else:
        tl.store(Y+i, tl.div_rn(value, denominator)*SCALE, valid)


class _GatherSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dst_projection, src_projection, radial, src, dst, scale):
        dst_projection, src_projection, radial = (
            value.contiguous() for value in (dst_projection, src_projection, radial))
        ctx.save_for_backward(dst_projection, src_projection, radial, src, dst)
        ctx.scale = scale
        out = torch.empty_like(radial)
        if radial.numel():
            _gather[(triton.cdiv(radial.numel(), 256),)](
                dst_projection, src_projection, radial, src, dst, radial, out,
                dst_projection, src_projection, radial, radial.numel(), radial.shape[1],
                scale, False, False, False, False, 256, enable_fp_fusion=False)
        return out

    @staticmethod
    def backward(ctx, grad):
        d, s, r, src, dst = ctx.saved_tensors
        from mace_ictc.models.adjoint_requirements import required_gradients
        needed = required_gradients((d, s, r), ctx.needs_input_grad[:3])
        if not any(needed):
            return None, None, None, None, None, None
        if torch.is_grad_enabled():
            value = d.index_select(0, dst) + s.index_select(0, src) + r
            sigmoid = value.sigmoid()
            dr = grad*ctx.scale*sigmoid*(1.+value*(1.-sigmoid))
            dd = torch.zeros_like(d).index_add(0, dst, dr) if needed[0] else None
            ds = torch.zeros_like(s).index_add(0, src, dr) if needed[1] else None
        else:
            dd = torch.zeros_like(d) if needed[0] else None
            ds = torch.zeros_like(s) if needed[1] else None
            dr = torch.empty_like(r) if needed[2] else None
            if r.numel():
                _gather[(triton.cdiv(r.numel(), 256),)](
                    d, s, r, src, dst, grad.contiguous(), r,
                    dd if needed[0] else d, ds if needed[1] else s, dr if needed[2] else r,
                    r.numel(), r.shape[1], ctx.scale, True, *needed, 256, enable_fp_fusion=False)
        return dd, ds, dr if needed[2] else None, None, None, None


class _GatherSiLUMixed(_GatherSiLU):
    """Explicit training candidate; no default whole-step gain established."""
    @staticmethod
    def backward(ctx, grad):
        if not torch.is_grad_enabled():
            return _GatherSiLU.backward(ctx, grad)
        d, s, r, src, dst = ctx.saved_tensors
        from mace_ictc.models.adjoint_requirements import required_gradients
        from mace_ictc.hp import gather_silu_adjoint
        needed = required_gradients((d, s, r), ctx.needs_input_grad[:3])
        if not any(needed):
            return (None,) * 6
        dd, ds, dr = gather_silu_adjoint(d, s, r, grad, src, dst, ctx.scale, needed)
        return dd, ds, dr, None, None, None


class PhaseTrunk:
    def __init__(self, layer, *, fused_radial=False, fused_tail=False, auto_network=False,
                 mixed_adjoint=False):
        self.layer = layer
        self.node_width = layer.phase_node_norm.normalized_shape[0]
        self.scale = layer.phase_trunk[1]._scale
        self.fused_radial = fused_radial
        self.fused_tail = fused_tail
        self.fused_network = False
        self.network_preprojected = False
        self.network = None
        self.auto_network = auto_network
        self.mixed_adjoint = mixed_adjoint

    def __call__(self, nodes, radial, src, dst):
        if self.fused_network or (self.auto_network and radial.shape[0] >= 327680):
            from .phase_network import FrozenPhaseNetwork
            if self.network is None or self.network.preprojected != self.network_preprojected:
                self.network = FrozenPhaseNetwork(self.layer, preprojected=self.network_preprojected)
            if self.fused_network or self.network.eligible(radial):
                return self.network(nodes, radial, src, dst)
        first = self.layer.phase_trunk[0]
        width = self.node_width
        weight = first.weight
        d = F.linear(nodes, weight[:, :width])
        s = F.linear(nodes, weight[:, width:2*width])
        if (self.fused_radial and not weight.requires_grad and first.bias is not None
                and not first.bias.requires_grad and radial.shape[1] <= 16
                and first.out_features in (16, 32, 64)):
            # Slicing a larger context matrix keeps its original row stride.
            radial_weight = weight[:, 2*width:].contiguous()
            hidden = _FrozenContextSiLU.apply(d, s, radial, radial_weight,
                                               first.bias, src, dst, self.scale)
        else:
            r = F.linear(radial, weight[:, 2*width:], first.bias)
            activation = _GatherSiLUMixed if self.mixed_adjoint else _GatherSiLU
            hidden = activation.apply(d, s, r, src, dst, self.scale)
        tail = self.layer.phase_trunk[2:]
        if (self.fused_tail and not self.layer.training and len(tail) == 2
                and isinstance(tail[0], torch.nn.Linear) and tail[0].bias is not None
                and tail[0].in_features == tail[0].out_features == 32
                and hasattr(tail[1], '_scale')
                and not any(p.requires_grad for p in tail.parameters())):
            from .affine_activation import FrozenAffineSiLU
            hidden = FrozenAffineSiLU.apply(hidden, tail[0].weight, tail[0].bias, tail[1]._scale)
        else:
            for module in tail:
                hidden = module(hidden)
        return hidden
