"""Pack neutral/real/imag radial gates without per-path Python tensor kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(G, C, S, I, O, TOTAL: tl.constexpr, CHANNELS: tl.constexpr,
          PATHS: tl.constexpr, HEADS: tl.constexpr, PHASES: tl.constexpr,
          BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    channel = offset % CHANNELS
    stream = (offset // CHANNELS) % 3
    path = (offset // (3*CHANNELS)) % PATHS
    edge = offset // (3*CHANNELS*PATHS)
    phase = tl.load(I + path*HEADS + channel//(CHANNELS//HEADS))
    value = tl.load(G + (edge*PATHS+path)*CHANNELS+channel, offset < TOTAL, other=0)
    cosine = tl.load(C + edge*PHASES+phase, offset < TOTAL, other=0)
    sine = tl.load(S + edge*PHASES+phase, offset < TOTAL, other=0)
    value = value * tl.where(stream == 0, 1., tl.where(stream == 1, cosine, sine))
    tl.store(O + offset, value, offset < TOTAL)


@triton.jit
def _unpack_grad(G, C, S, I, DY, DG, DC, DS, ROWS: tl.constexpr,
                 CHANNELS: tl.constexpr, PATHS: tl.constexpr, HEADS: tl.constexpr,
                 PHASES: tl.constexpr, RC: tl.constexpr, RR: tl.constexpr):
    row = tl.program_id(0)*RR + tl.arange(0, RR)
    cc = tl.arange(0, RC)
    head = row % HEADS
    path = (row // HEADS) % PATHS
    edge = row // (HEADS*PATHS)
    channel = head[:, None]*(CHANNELS//HEADS)+cc[None, :]
    mask = (row[:, None] < ROWS) & (cc[None, :] < CHANNELS//HEADS)
    phase = tl.load(I + path*HEADS+head)
    base = (edge[:, None]*PATHS+path[:, None])*3*CHANNELS+channel
    g = tl.load(G + (edge[:, None]*PATHS+path[:, None])*CHANNELS+channel, mask, other=0)
    d0 = tl.load(DY+base, mask, other=0)
    dr = tl.load(DY+base+CHANNELS, mask, other=0)
    di = tl.load(DY+base+2*CHANNELS, mask, other=0)
    cosine = tl.load(C+edge*PHASES+phase, row < ROWS, other=0)
    sine = tl.load(S+edge*PHASES+phase, row < ROWS, other=0)
    tl.store(DG + (edge[:, None]*PATHS+path[:, None])*CHANNELS+channel,
             d0+dr*cosine[:, None]+di*sine[:, None], mask)
    tl.atomic_add(DC+edge*PHASES+phase, tl.sum(g*dr, 1), row < ROWS, sem="relaxed")
    tl.atomic_add(DS+edge*PHASES+phase, tl.sum(g*di, 1), row < ROWS, sem="relaxed")


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, cosine, sine, indices, channels, heads):
        e, p = gates.shape[0], indices.numel()//heads
        phases = cosine.shape[1]
        output = gates.new_empty((e, p*3*channels))
        gates, cosine, sine = gates.contiguous(), cosine.contiguous(), sine.contiguous()
        ctx.save_for_backward(gates, cosine, sine, indices)
        ctx.channels, ctx.heads, ctx.paths = channels, heads, p
        if output.numel():
            _pack[(triton.cdiv(output.numel(), 256),)](
                gates, cosine, sine, indices, output, output.numel(), channels,
                p, heads, phases, 256, enable_fp_fusion=False)
        return output

    @staticmethod
    def backward(ctx, grad):
        gates, cosine, sine, indices = ctx.saved_tensors
        e, p, c, h = gates.shape[0], ctx.paths, ctx.channels, ctx.heads
        if torch.is_grad_enabled():
            # Differentiable exact adjoint for force-loss training and all higher
            # derivatives. Inference uses the fused first-adjoint kernel below.
            grouped = gates.reshape(e, p, h, c//h)
            dy = grad.reshape(e, p, 3, h, c//h)
            cos = cosine.squeeze(-1).index_select(1, indices).reshape(e, p, h, 1)
            sin = sine.squeeze(-1).index_select(1, indices).reshape(e, p, h, 1)
            dg = (dy[:, :, 0]+dy[:, :, 1]*cos+dy[:, :, 2]*sin).reshape_as(gates)
            dc = torch.zeros_like(cosine.squeeze(-1)).index_add(
                1, indices, (grouped*dy[:, :, 1]).sum(-1).reshape(e, p*h)).unsqueeze(-1)
            ds = torch.zeros_like(sine.squeeze(-1)).index_add(
                1, indices, (grouped*dy[:, :, 2]).sum(-1).reshape(e, p*h)).unsqueeze(-1)
        else:
            dg = torch.empty_like(gates)
            dc, ds = torch.zeros_like(cosine), torch.zeros_like(sine)
            rows = e*p*h
            if rows:
                _unpack_grad[(triton.cdiv(rows, 4),)](
                    gates, cosine, sine, indices, grad.contiguous(), dg, dc, ds,
                    rows, c, p, h, cosine.shape[1], triton.next_power_of_2(c//h), 4,
                    enable_fp_fusion=False)
        return dg, dc, ds, None, None, None


def pack_phase_gates(gates, cosine, sine, indices, channels, heads):
    return _Pack.apply(gates, cosine, sine, indices, channels, heads)


@triton.jit
def _modulate(G, C, I, O, TOTAL: tl.constexpr, CHANNELS: tl.constexpr,
              PATHS: tl.constexpr, HEADS: tl.constexpr, PHASES: tl.constexpr,
              BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    channel, path = offset % CHANNELS, (offset//CHANNELS) % PATHS
    edge = offset//(CHANNELS*PATHS)
    phase = tl.load(I+path*HEADS+channel//(CHANNELS//HEADS))
    gate = tl.load(G+offset, offset < TOTAL, 0.)
    coefficient = tl.load(C+edge*PHASES+phase, offset < TOTAL, 0.)
    tl.store(O+offset, gate*coefficient, offset < TOTAL)


@triton.jit
def _modulate_grad(G, C, I, DY, DG, DC, ROWS: tl.constexpr, CHANNELS: tl.constexpr,
                   PATHS: tl.constexpr, HEADS: tl.constexpr, PHASES: tl.constexpr,
                   RC: tl.constexpr, RR: tl.constexpr):
    row = tl.program_id(0)*RR+tl.arange(0, RR)
    cc = tl.arange(0, RC)
    head, path = row % HEADS, (row//HEADS) % PATHS
    edge = row//(HEADS*PATHS)
    channel = head[:, None]*(CHANNELS//HEADS)+cc[None, :]
    mask = (row[:, None] < ROWS) & (cc[None, :] < CHANNELS//HEADS)
    phase = tl.load(I+path*HEADS+head)
    offset = (edge[:, None]*PATHS+path[:, None])*CHANNELS+channel
    g, dy = tl.load(G+offset, mask, 0.), tl.load(DY+offset, mask, 0.)
    coefficient = tl.load(C+edge*PHASES+phase, row < ROWS, 0.)
    tl.store(DG+offset, dy*coefficient[:, None], mask)
    tl.atomic_add(DC+edge*PHASES+phase, tl.sum(g*dy, 1), row < ROWS, sem="relaxed")


class _Modulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gates, coefficient, indices, channels, heads):
        gates, coefficient = gates.contiguous(), coefficient.contiguous()
        ctx.save_for_backward(gates, coefficient, indices)
        ctx.channels, ctx.heads = channels, heads
        out = torch.empty_like(gates)
        if gates.numel():
            _modulate[(triton.cdiv(gates.numel(), 256),)](
                gates, coefficient, indices, out, gates.numel(), channels,
                indices.numel()//heads, heads, coefficient.shape[1], 256,
                enable_fp_fusion=False)
        return out

    @staticmethod
    def backward(ctx, grad):
        gates, coefficient, indices = ctx.saved_tensors
        c, h = ctx.channels, ctx.heads
        e, p = gates.shape[0], indices.numel()//h
        if torch.is_grad_enabled():
            grouped = gates.reshape(e, p, h, c//h)
            dy = grad.reshape_as(grouped)
            factor = coefficient.squeeze(-1).index_select(1, indices).reshape(e, p, h, 1)
            dg = (dy*factor).reshape_as(gates)
            dc = torch.zeros_like(coefficient.squeeze(-1)).index_add(
                1, indices, (grouped*dy).sum(-1).reshape(e, p*h)).unsqueeze(-1)
        else:
            dg, dc = torch.empty_like(gates), torch.zeros_like(coefficient)
            if e*p*h:
                _modulate_grad[(triton.cdiv(e*p*h, 4),)](
                    gates, coefficient, indices, grad.contiguous(), dg, dc,
                    e*p*h, c, p, h, coefficient.shape[1],
                    triton.next_power_of_2(c//h), 4, enable_fp_fusion=False)
        return dg, dc, None, None, None


def modulate_gates(gates, coefficient, indices, channels, heads):
    """One charged stream; never replicate coefficients across all channels."""
    return _Modulate.apply(gates, coefficient, indices, channels, heads)
