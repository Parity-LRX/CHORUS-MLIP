"""Joint frozen IEEE-FP32 phase MLP/heads, with live context and exact VJP.

Two kernels replace the edge radial affine, two activations, hidden affine,
and separate scalar heads. Node-context projections remain node-owned GEMMs.
No live activation is cached between calls and original parameters are kept.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _phase_mlp(D, S, R, W0, B0, W1, B1, WH, BH, SRC, DST,
               GT, GA, T, A, DD, DS, DR, Z0, Z1,
               E: tl.constexpr, K: tl.constexpr, H: tl.constexpr, F: tl.constexpr,
               SCALE0: tl.constexpr, SCALE1: tl.constexpr,
               BACKWARD: tl.constexpr, NEED_D: tl.constexpr, NEED_S: tl.constexpr,
               NEED_R: tl.constexpr, PREPROJECTED: tl.constexpr, BLOCK_E: tl.constexpr):
    edge = tl.program_id(0)*BLOCK_E+tl.arange(0, BLOCK_E)
    k, h, f = tl.arange(0, 16), tl.arange(0, H), tl.arange(0, 16)
    src, dst = tl.load(SRC+edge, edge < E, 0), tl.load(DST+edge, edge < E, 0)
    if BACKWARD:
        z0 = tl.load(Z0+edge[:, None]*H+h[None, :], edge[:, None] < E, 0.)
        z1 = tl.load(Z1+edge[:, None]*H+h[None, :], edge[:, None] < E, 0.)
        phase = f % F
        gt = tl.load(GT+edge[:, None]*F+phase[None, :], edge[:, None] < E, 0.)
        ga = tl.load(GA+edge[:, None]*F+phase[None, :], edge[:, None] < E, 0.)
        gh = tl.where(f[None, :] < F, gt, tl.where(f[None, :] < 2*F, ga, 0.))
        wh = tl.load(WH+f[:, None]*H+h[None, :])
        dh1 = tl.dot(gh, wh, input_precision='ieee')
        sig1 = 1./(1.+tl.exp(-z1))
        dz1 = dh1*SCALE1*sig1*(1.+z1*(1.-sig1))
        w1 = tl.load(W1+h[:, None]*H+h[None, :])
        dh0 = tl.dot(dz1, w1, input_precision='ieee')
        sig0 = 1./(1.+tl.exp(-z0))
        dz0 = dh0*SCALE0*sig0*(1.+z0*(1.-sig0))
        valid = edge[:, None] < E
        if NEED_D:tl.atomic_add(DD+dst[:, None]*H+h[None, :], dz0, valid, sem='relaxed')
        if NEED_S:tl.atomic_add(DS+src[:, None]*H+h[None, :], dz0, valid, sem='relaxed')
        if NEED_R:
            if PREPROJECTED:
                tl.store(DR+edge[:, None]*H+h[None, :], dz0, valid)
            else:
                w0 = tl.load(W0+h[:, None]*K+k[None, :], k[None, :] < K, 0.)
                dr = tl.dot(dz0, w0, input_precision='ieee')
                tl.store(DR+edge[:, None]*K+k[None, :], dr, valid & (k[None, :] < K))
    else:
        if PREPROJECTED:
            radial = tl.load(R+edge[:, None]*H+h[None, :], edge[:, None] < E, 0.)
        else:
            r = tl.load(R+edge[:, None]*K+k[None, :], (edge[:, None] < E) & (k[None, :] < K), 0.)
            w0 = tl.load(W0+h[None, :]*K+k[:, None], k[:, None] < K, 0.)
            radial = tl.dot(r, w0, input_precision='ieee')+tl.load(B0+h)[None, :]
        z0 = tl.load(D+dst[:, None]*H+h[None, :], edge[:, None] < E, 0.)
        z0 += tl.load(S+src[:, None]*H+h[None, :], edge[:, None] < E, 0.)
        z0 += radial
        sig0 = 1./(1.+tl.exp(-z0))
        h0 = z0*sig0*SCALE0
        w1 = tl.load(W1+h[None, :]*H+h[:, None])
        z1 = tl.dot(h0, w1, input_precision='ieee')+tl.load(B1+h)[None, :]
        sig1 = 1./(1.+tl.exp(-z1))
        h1 = z1*sig1*SCALE1
        wh = tl.load(WH+f[None, :]*H+h[:, None])
        heads = tl.dot(h1, wh, input_precision='ieee')+tl.load(BH+f)[None, :]
        tl.store(Z0+edge[:, None]*H+h[None, :], z0, edge[:, None] < E)
        tl.store(Z1+edge[:, None]*H+h[None, :], z1, edge[:, None] < E)
        # Store both heads without a padded E*16 intermediate.
        phase = f % F
        tl.store(T+edge[:, None]*F+phase[None, :], heads, (edge[:, None] < E) & (f[None, :] < F))
        tl.store(A+edge[:, None]*F+phase[None, :], heads, (edge[:, None] < E) & (f[None, :] >= F) & (f[None, :] < 2*F))


def _native(d, s, radial, w0, b0, w1, b1, wh, bh, src, dst, scale0, scale1, phases):
    phases, preprojected = phases if isinstance(phases, tuple) else (phases, False)
    h0 = F.silu(d[dst]+s[src]+(radial if preprojected else F.linear(radial, w0, b0)))*scale0
    h1 = F.silu(F.linear(h0, w1, b1))*scale1
    return F.linear(h1, wh[:phases], bh[:phases]), F.linear(h1, wh[phases:2*phases], bh[phases:2*phases])


class _PhaseHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, s, radial, w0, b0, w1, b1, wh, bh, src, dst, scale0, scale1, phases):
        if any(v.requires_grad for v in (w0, b0, w1, b1, wh, bh)):
            raise ValueError('Frozen phase fusion cannot discard parameter gradients')
        d, s, radial = d.contiguous(), s.contiguous(), radial.contiguous()
        phases, preprojected = phases if isinstance(phases, tuple) else (phases, False)
        theta, raw = radial.new_empty(radial.shape[0], phases), radial.new_empty(radial.shape[0], phases)
        z0, z1 = d.new_empty(radial.shape[0], d.shape[1]), d.new_empty(radial.shape[0], d.shape[1])
        ctx.scales, ctx.phases = (scale0, scale1), phases
        ctx.preprojected = preprojected
        if radial.shape[0]:
            _phase_mlp[(triton.cdiv(radial.shape[0], 16),)](d, s, radial, w0, b0, w1, b1, wh, bh, src, dst,
                theta, raw, theta, raw, d, s, radial, z0, z1,
                radial.shape[0], radial.shape[1], d.shape[1], phases, scale0, scale1,
                False, False, False, False, preprojected, 16, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(d, s, radial, w0, b0, w1, b1, wh, bh, src, dst, z0, z1)
        ctx.set_materialize_grads(False)
        return theta, raw

    @staticmethod
    def backward(ctx, gt, ga):
        d, s, radial, w0, b0, w1, b1, wh, bh, src, dst, z0, z1 = ctx.saved_tensors
        from mace_ictc.models.adjoint_requirements import required_gradients
        needed = required_gradients((d, s, radial), ctx.needs_input_grad[:3])
        shape = (radial.shape[0], ctx.phases)
        gt = radial.new_zeros(shape) if gt is None else gt.contiguous()
        ga = radial.new_zeros(shape) if ga is None else ga.contiguous()
        if torch.is_grad_enabled():
            local = tuple(v.clone() for v in (d, s, radial))
            output = _native(*local, w0, b0, w1, b1, wh, bh, src, dst, *ctx.scales, (ctx.phases, ctx.preprojected))
            active = [v for v, need in zip(local, needed) if need]
            values = iter(torch.autograd.grad(output, active, (gt, ga), create_graph=True, allow_unused=True))
            dd, ds, dr = [next(values) if need else None for need in needed]
        else:
            dd = torch.zeros_like(d) if needed[0] else None
            ds = torch.zeros_like(s) if needed[1] else None
            dr = torch.empty_like(radial) if needed[2] else None
            if radial.shape[0] and any(needed):
                _phase_mlp[(triton.cdiv(radial.shape[0], 16),)](d, s, radial, w0, b0, w1, b1, wh, bh, src, dst,
                    gt, ga, gt, ga, dd if needed[0] else d, ds if needed[1] else s,
                    dr if needed[2] else radial, z0, z1,
                    radial.shape[0], radial.shape[1], d.shape[1], ctx.phases, *ctx.scales,
                    True, *needed, ctx.preprojected, 16, num_warps=4, enable_fp_fusion=False)
        return dd, ds, dr, *(None for _ in range(11))


class PhaseHeadPair:
    _hp_phase_heads = True
    def __init__(self, theta, raw):self.theta, self.raw = theta, raw


class FrozenPhaseNetwork:
    def __init__(self, layer, *, preprojected=False):
        self.layer, self.signature, self.weights = layer, None, None
        self.preprojected = preprojected

    def eligible(self, radial):
        layer = self.layer
        return (not layer.training and not any(p.requires_grad for p in layer.phase_trunk.parameters())
            and not any(p.requires_grad for p in layer.phase_head.parameters())
            and layer.phase_amplitude_head is not None
            and not any(p.requires_grad for p in layer.phase_amplitude_head.parameters())
            and layer.phase_trunk[0].out_features == 32
            and layer.phase_trunk[2].in_features == layer.phase_trunk[2].out_features == 32
            and layer.phase_trunk[0].bias is not None and layer.phase_trunk[2].bias is not None
            and layer.phase_amplitude_head.bias is not None
            and layer.phase_head.out_features <= 4 and radial.shape[1] <= 16)

    def __call__(self, nodes, radial, src, dst):
        if not self.eligible(radial):raise ValueError('Original phase MLP is ineligible for frozen fusion')
        layer, width = self.layer, self.layer.phase_node_norm.normalized_shape[0]
        first, second = layer.phase_trunk[0], layer.phase_trunk[2]
        theta, amplitude = layer.phase_head, layer.phase_amplitude_head
        parameters = (*layer.phase_trunk.parameters(), *theta.parameters(), *amplitude.parameters())
        signature = tuple((id(p), p._version, p.dtype, p.device) for p in parameters)
        if signature != self.signature:
            with torch.no_grad():
                wh, bh = first.weight.new_zeros(16, 32), first.weight.new_zeros(16)
                phases = theta.out_features
                wh[:phases].copy_(theta.weight);wh[phases:2*phases].copy_(amplitude.weight)
                if theta.bias is not None:bh[:phases].copy_(theta.bias)
                bh[phases:2*phases].copy_(amplitude.bias)
                self.weights = (first.weight[:, 2*width:].contiguous(), first.bias,
                                second.weight.contiguous(), second.bias, wh, bh)
            self.signature = signature
        d, s = F.linear(nodes, first.weight[:, :width]), F.linear(nodes, first.weight[:, width:2*width])
        if self.preprojected:
            radial = F.linear(radial, first.weight[:, 2*width:], first.bias)
        value = _PhaseHeads.apply(d, s, radial, *self.weights, src, dst,
            layer.phase_trunk[1]._scale, layer.phase_trunk[3]._scale, (theta.out_features, self.preprojected))
        return PhaseHeadPair(*value)
