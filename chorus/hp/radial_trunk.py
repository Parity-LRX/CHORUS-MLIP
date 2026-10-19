"""Frozen radial 8->64->64->64 trunk, IEEE FP32 value and coordinate VJP.

The wide original final linear stays a GEMM. Only intermediate launch/memory
boundaries are fused; all weights, normalized activations and higher derivatives
are the original ones. This is an explicit candidate, not an automatic policy.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _trunk(X, W0, W1, W2, Z0, Z1, Z2, G, Y, DX,
           E: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
           S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
           BACKWARD: tl.constexpr, BLOCK_E: tl.constexpr):
    e = tl.program_id(0)*BLOCK_E+tl.arange(0, BLOCK_E)
    h, k = tl.arange(0, H), tl.arange(0, 16)
    valid = e[:, None] < E
    if BACKWARD:
        z2 = tl.load(Z2+e[:, None]*H+h[None, :], valid, 0.)
        grad = tl.load(G+e[:, None]*H+h[None, :], valid, 0.)
        sig = 1./(1.+tl.exp(-z2))
        dz2 = grad*S2*sig*(1.+z2*(1.-sig))
        w2 = tl.load(W2+h[:, None]*H+h[None, :])
        dh1 = tl.dot(dz2, w2, input_precision='ieee')
        z1 = tl.load(Z1+e[:, None]*H+h[None, :], valid, 0.)
        sig = 1./(1.+tl.exp(-z1))
        dz1 = dh1*S1*sig*(1.+z1*(1.-sig))
        w1 = tl.load(W1+h[:, None]*H+h[None, :])
        dh0 = tl.dot(dz1, w1, input_precision='ieee')
        z0 = tl.load(Z0+e[:, None]*H+h[None, :], valid, 0.)
        sig = 1./(1.+tl.exp(-z0))
        dz0 = dh0*S0*sig*(1.+z0*(1.-sig))
        w0 = tl.load(W0+h[:, None]*K+k[None, :], k[None, :] < K, 0.)
        dx = tl.dot(dz0, w0, input_precision='ieee')
        tl.store(DX+e[:, None]*K+k[None, :], dx, valid & (k[None, :] < K))
    else:
        x = tl.load(X+e[:, None]*K+k[None, :], valid & (k[None, :] < K), 0.)
        w0 = tl.load(W0+h[None, :]*K+k[:, None], k[:, None] < K, 0.)
        z0 = tl.dot(x, w0, input_precision='ieee')
        sig = 1./(1.+tl.exp(-z0))
        h0 = z0*sig*S0
        w1 = tl.load(W1+h[None, :]*H+h[:, None])
        z1 = tl.dot(h0, w1, input_precision='ieee')
        sig = 1./(1.+tl.exp(-z1))
        h1 = z1*sig*S1
        w2 = tl.load(W2+h[None, :]*H+h[:, None])
        z2 = tl.dot(h1, w2, input_precision='ieee')
        sig = 1./(1.+tl.exp(-z2))
        tl.store(Y+e[:, None]*H+h[None, :], z2*sig*S2, valid)
        tl.store(Z0+e[:, None]*H+h[None, :], z0, valid)
        tl.store(Z1+e[:, None]*H+h[None, :], z1, valid)
        tl.store(Z2+e[:, None]*H+h[None, :], z2, valid)


def _native(x, w0, w1, w2, scales):
    for weight, scale in zip((w0, w1, w2), scales):
        x = F.silu(F.linear(x, weight))*scale
    return x


class _Trunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2, scales):
        if any(w.requires_grad for w in (w0, w1, w2)):
            raise ValueError('Frozen radial trunk cannot omit parameter gradients')
        x = x.contiguous()
        tile = int(scales[3]) if len(scales) == 4 else 16
        scales = scales[:3]
        if tile not in (16, 32, 64, 128):
            raise ValueError('Radial trunk tile must be 16, 32, 64 or 128')
        shape = (x.shape[0], w0.shape[0])
        y, z0, z1, z2 = (x.new_empty(shape) for _ in range(4))
        if x.shape[0]:
            _trunk[(triton.cdiv(x.shape[0], tile),)](x, w0, w1, w2, z0, z1, z2, y, y, x,
                x.shape[0], x.shape[1], w0.shape[0], *scales, False, tile,
                num_warps=4 if tile < 64 else 8, enable_fp_fusion=False)
        ctx.save_for_backward(x, w0, w1, w2, z0, z1, z2)
        ctx.scales = scales
        ctx.tile = tile
        return y

    @staticmethod
    def backward(ctx, grad):
        x, w0, w1, w2, z0, z1, z2 = ctx.saved_tensors
        if torch.is_grad_enabled():
            local = x.clone()
            output = _native(local, w0, w1, w2, ctx.scales)
            dx, = torch.autograd.grad(output, local, grad, create_graph=True)
        else:
            dx = torch.empty_like(x)
            if x.shape[0]:
                _trunk[(triton.cdiv(x.shape[0], ctx.tile),)](x, w0, w1, w2, z0, z1, z2,
                    grad.contiguous(), grad, dx, x.shape[0], x.shape[1], w0.shape[0],
                    *ctx.scales, True, ctx.tile, num_warps=4 if ctx.tile < 64 else 8, enable_fp_fusion=False)
        return dx, None, None, None, None


class FrozenRadialTrunk:
    def __init__(self, tile=16):self.tile = tile

    @staticmethod
    def eligible(fc, x):
        return (not fc.training and len(fc) == 7 and 1 <= x.shape[1] <= 16
            and x.dtype == torch.float32 and x.is_cuda
            and all(isinstance(fc[k], torch.nn.Linear) and fc[k].bias is None for k in (0, 2, 4, 6))
            and not any(p.requires_grad for p in fc.parameters())
            and fc[0].out_features == 64 and fc[2].in_features == fc[2].out_features == 64
            and fc[4].in_features == fc[4].out_features == fc[6].in_features == 64
            and all(hasattr(fc[k], '_scale') for k in (1, 3, 5)))

    def __call__(self, fc, x):
        if not self.eligible(fc, x):
            raise ValueError('Original radial network is ineligible for frozen trunk fusion')
        return _Trunk.apply(x, *(fc[k].weight for k in (0, 2, 4)),
                            (*tuple(fc[k]._scale for k in (1, 3, 5)), self.tile))
