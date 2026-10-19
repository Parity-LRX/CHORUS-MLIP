"""Frozen narrow IEEE-FP32 affine/normalized-SiLU and fused input VJP."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _affine_silu(X, W, B, G, Y, Z, E: tl.constexpr, K: tl.constexpr,
                 H: tl.constexpr, SCALE: tl.constexpr, BACKWARD: tl.constexpr,
                 BLOCK_E: tl.constexpr):
    row = tl.program_id(0)*BLOCK_E + tl.arange(0, BLOCK_E)
    k = tl.arange(0, K)
    h = tl.arange(0, H)
    if BACKWARD:
        z = tl.load(Z + row[:, None]*H + h[None, :], row[:, None] < E, 0.)
    else:
        x = tl.load(X + row[:, None]*K + k[None, :], row[:, None] < E, 0.)
        w = tl.load(W + h[None, :]*K + k[:, None])
        z = tl.dot(x, w, input_precision='ieee') + tl.load(B+h)[None, :]
        tl.store(Z + row[:, None]*H + h[None, :], z, row[:, None] < E)
    sigmoid = 1./(1.+tl.exp(-z))
    if BACKWARD:
        grad = tl.load(G + row[:, None]*H + h[None, :], row[:, None] < E, 0.)
        dz = grad*SCALE*sigmoid*(1.+z*(1.-sigmoid))
        wt = tl.load(W + h[:, None]*K + k[None, :])
        dx = tl.dot(dz, wt, input_precision='ieee')
        tl.store(Y + row[:, None]*K + k[None, :], dx, row[:, None] < E)
    else:
        tl.store(Y + row[:, None]*H + h[None, :], z*sigmoid*SCALE, row[:, None] < E)


class FrozenAffineSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, scale):
        if weight.requires_grad or bias.requires_grad:
            raise ValueError('Frozen affine activation cannot discard parameter gradients')
        x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
        ctx.scale = scale
        out = x.new_empty(x.shape[0], weight.shape[0])
        z = torch.empty_like(out)
        if x.shape[0]:
            _affine_silu[(triton.cdiv(x.shape[0], 16),)](x, weight, bias, x, out, z,
                x.shape[0], weight.shape[1], weight.shape[0], scale, False, 16,
                num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, bias, z)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, bias, z = ctx.saved_tensors
        if torch.is_grad_enabled():
            z = F.linear(x, weight, bias)
            sigmoid = z.sigmoid()
            dx = (grad*ctx.scale*sigmoid*(1.+z*(1.-sigmoid)))@weight
        else:
            dx = torch.empty_like(x)
            if x.shape[0]:
                _affine_silu[(triton.cdiv(x.shape[0], 16),)](x, weight, bias, grad.contiguous(), dx, z,
                    x.shape[0], weight.shape[1], weight.shape[0], ctx.scale, True, 16,
                    num_warps=4, enable_fp_fusion=False)
        return dx, None, None, None
