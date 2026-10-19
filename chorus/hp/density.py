"""Fused full-L quadratic density with first and mixed-second adjoints.

Plans contain the existing module's exact CG entries, including nonzero roundoff
coefficients. Projection and output weights remain live PyTorch parameters.
No approximation, rank change, basis change or learned activation is cached.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quadratic(X, H, A, B, C, D, W, S, Y, TOTAL: tl.constexpr,
               XDIM: tl.constexpr, WIDTH: tl.constexpr, K: tl.constexpr,
               BLOCK: tl.constexpr, JVP: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = tl.arange(0, K)
    slot = row % WIDTH
    batch = row // WIDTH
    active = row < TOTAL
    table = slot[:, None] * K + k[None, :]
    a = tl.load(A + table)
    b = tl.load(B + table)
    c = tl.load(C + table)
    d = tl.load(D + table)
    w = tl.load(W + table)
    sign = tl.load(S + slot)[:, None]
    base = batch[:, None] * XDIM
    xa = tl.load(X + base + a, active[:, None], other=0)
    xb = tl.load(X + base + b, active[:, None], other=0)
    xc = tl.load(X + base + c, active[:, None], other=0)
    xd = tl.load(X + base + d, active[:, None], other=0)
    if JVP:
        ha = tl.load(H + base + a, active[:, None], other=0)
        hb = tl.load(H + base + b, active[:, None], other=0)
        hc = tl.load(H + base + c, active[:, None], other=0)
        hd = tl.load(H + base + d, active[:, None], other=0)
        value = (ha*xb + xa*hb + sign*(hc*xd + xc*hd)) * w
    else:
        value = (xa*xb + sign*xc*xd) * w
    tl.store(Y + row, tl.sum(value, 1), active)


@triton.jit
def _adjoint(X, G, A, B, C, D, W, S, DX, TOTAL: tl.constexpr,
             XDIM: tl.constexpr, WIDTH: tl.constexpr, K: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = tl.arange(0, K)
    slot, batch = row % WIDTH, row // WIDTH
    table = slot[:, None] * K + k[None, :]
    a, b = tl.load(A + table), tl.load(B + table)
    c, d = tl.load(C + table), tl.load(D + table)
    w = tl.load(W + table)
    active = (row[:, None] < TOTAL) & (w != 0.)
    base = batch[:, None] * XDIM
    scale = tl.load(G + row, row < TOTAL, other=0)[:, None] * w
    sign = tl.load(S + slot)[:, None]
    xa = tl.load(X + base + a, active, other=0)
    xb = tl.load(X + base + b, active, other=0)
    xc = tl.load(X + base + c, active, other=0)
    xd = tl.load(X + base + d, active, other=0)
    tl.atomic_add(DX + base + a, scale*xb, active, sem="relaxed")
    tl.atomic_add(DX + base + b, scale*xa, active, sem="relaxed")
    tl.atomic_add(DX + base + c, scale*sign*xd, active, sem="relaxed")
    tl.atomic_add(DX + base + d, scale*sign*xc, active, sem="relaxed")


@triton.jit
def _adjoint_gather(X, G, ROW, OTHER, WEIGHT, DX, TOTAL: tl.constexpr,
                    XDIM: tl.constexpr, WIDTH: tl.constexpr, K: tl.constexpr,
                    BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    k = tl.arange(0, K)
    slot, batch = index % XDIM, index // XDIM
    table = slot[:, None]*K + k[None, :]
    row = tl.load(ROW + table)
    other = tl.load(OTHER + table)
    weight = tl.load(WEIGHT + table)
    active = (index[:, None] < TOTAL) & (weight != 0.)
    scale = tl.load(G + batch[:, None]*WIDTH + row, active, other=0)*weight
    value = tl.load(X + batch[:, None]*XDIM + other, active, other=0)
    tl.store(DX + index, tl.sum(scale*value, 1), index < TOTAL)


class _DensityVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad, plan):
        ctx.save_for_backward(x, grad)
        ctx.plan = plan
        return plan.vjp(x, grad)

    @staticmethod
    def backward(ctx, h):
        x, grad = ctx.saved_tensors
        plan = ctx.plan
        # J(x) is linear in x. Its mixed derivative is the same VJP evaluated
        # at h, and the cotangent derivative is the exact quadratic JVP.
        if torch.is_grad_enabled():
            dx = plan.native_vjp(h, grad) if ctx.needs_input_grad[0] else None
            dg = plan.native_jvp(x, h) if ctx.needs_input_grad[1] else None
        else:
            dx = plan.vjp(h, grad) if ctx.needs_input_grad[0] else None
            dg = plan.forward(x, h) if ctx.needs_input_grad[1] else None
        return dx, dg, None


class _Density(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan):
        ctx.save_for_backward(x)
        ctx.plan = plan
        return plan.forward(x)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        if torch.is_grad_enabled():
            return _DensityVJP.apply(x, grad, ctx.plan), None
        return ctx.plan.vjp(x, grad), None


class DensityPlan:
    """One kernel for all retained real/imag CG paths, plus fused adjoints."""
    def __init__(self, owner):
        self.owner = owner
        self.rank, self.lmax = owner.density_rank, owner.lmax
        self.xhalf = self.rank * (self.lmax + 1)**2
        self.sizes = [owner.input_multiplicities[l] * (2*l+1)
                      for l in range(self.lmax + 1)]
        self.width = sum(self.sizes)
        rows, signs, constants = [], [], []
        for out_l in range(self.lmax + 1):
            for l1, l2, target, component, name in owner.paths:
                if target != out_l:
                    continue
                cg = getattr(owner, name)
                constants.append((name, id(cg), cg._version))
                values = cg.detach().cpu().tolist()
                for rank in range(self.rank):
                    for m in range(2*out_l+1):
                        terms = []
                        for i in range(2*l1+1):
                            for j in range(2*l2+1):
                                value = values[i][j][m]
                                if value == 0.0:
                                    continue
                                a = self.rank*l1*l1 + rank*(2*l1+1) + i
                                b = self.rank*l2*l2 + rank*(2*l2+1) + j
                                if component == "real":
                                    indices = (a, b, a+self.xhalf, b+self.xhalf)
                                elif owner.quadratic_form == "hermitian":
                                    indices = (a+self.xhalf, b, a, b+self.xhalf)
                                else:
                                    indices = (a, b+self.xhalf, a+self.xhalf, b)
                                terms.append((*indices, value))
                        rows.append(terms)
                        signs.append(1.0 if ((component == "real") ==
                                            (owner.quadratic_form == "hermitian")) else -1.0)
        self.constants = tuple(dict.fromkeys(constants))
        self.form = owner.quadratic_form
        if len(rows) != self.width:
            raise ValueError("Hermitian output path layout mismatch")
        self.k = triton.next_power_of_2(max(1, max(map(len, rows))))
        padded = [terms + [(0, 0, 0, 0, 0.0)]*(self.k-len(terms)) for terms in rows]
        device = next(owner.parameters()).device
        self.indices = [torch.tensor([[term[i] for term in row] for row in padded],
                                     device=device, dtype=torch.int32) for i in range(4)]
        self.weights = torch.tensor([[term[4] for term in row] for row in padded],
                                    device=device, dtype=torch.float32)
        self.signs = torch.tensor(signs, device=device, dtype=torch.float32)
        # Transpose the exact sparse quadratic derivative. Each input owns its
        # whole sum, removing four contended atomic scatters per CG term. Keep
        # duplicate entries and every nonzero CG coefficient without pruning.
        derivative = [[] for _ in range(2*self.xhalf)]
        for row, (terms, sign) in enumerate(zip(rows, signs)):
            for a, b, c, d, weight in terms:
                derivative[a].append((row, b, weight))
                derivative[b].append((row, a, weight))
                derivative[c].append((row, d, sign*weight))
                derivative[d].append((row, c, sign*weight))
        self.derivative_k = triton.next_power_of_2(max(1, max(map(len, derivative))))
        padded_derivative = [terms + [(0, 0, 0.0)]*(self.derivative_k-len(terms))
                             for terms in derivative]
        self.derivative_rows, self.derivative_other = [torch.tensor(
            [[term[i] for term in row] for row in padded_derivative],
            device=device, dtype=torch.int32) for i in range(2)]
        self.derivative_weights = torch.tensor(
            [[term[2] for term in row] for row in padded_derivative],
            device=device, dtype=torch.float32)

    def _args(self):
        return (*self.indices, self.weights, self.signs)

    def forward(self, x, h=None):
        x = x.contiguous()
        y = x.new_empty((x.shape[0], self.width))
        if y.numel():
            _quadratic[(triton.cdiv(y.numel(), 32),)](
                x, x if h is None else h.contiguous(), *self._args(), y,
                y.numel(), x.shape[1], self.width, self.k, 32, h is not None,
                enable_fp_fusion=False)
        return y

    def vjp(self, x, grad):
        return self.gather_vjp(x, grad)

    def atomic_vjp(self, x, grad):
        dx = torch.zeros_like(x, memory_format=torch.contiguous_format)
        if grad.numel():
            _adjoint[(triton.cdiv(grad.numel(), 32),)](
                x.contiguous(), grad.contiguous(), *self._args(), dx,
                grad.numel(), x.shape[1], self.width, self.k, 32,
                enable_fp_fusion=False)
        return dx

    def gather_vjp(self, x, grad):
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        if dx.numel():
            _adjoint_gather[(triton.cdiv(dx.numel(), 16),)](
                x.contiguous(), grad.contiguous(), self.derivative_rows,
                self.derivative_other, self.derivative_weights, dx,
                dx.numel(), x.shape[1], self.width, self.derivative_k, 16,
                enable_fp_fusion=False)
        return dx

    def native_jvp(self, x, h):
        a, b, c, d = [index.long().reshape(-1) for index in self.indices]
        shape = (x.shape[0], self.width, self.k)
        xa, xb, xc, xd = [x[:, idx].reshape(shape) for idx in (a, b, c, d)]
        ha, hb, hc, hd = [h[:, idx].reshape(shape) for idx in (a, b, c, d)]
        return ((ha*xb+xa*hb+self.signs[None, :, None]*(hc*xd+xc*hd))
                * self.weights[None]).sum(-1)

    def native_vjp(self, x, grad):
        indices = [index.long().reshape(-1) for index in self.indices]
        scale = (grad[..., None]*self.weights).reshape(x.shape[0], -1)
        sign = self.signs[:, None].expand(-1, self.k).reshape(-1)
        result = torch.zeros_like(x)
        for dest, other, factor in ((0, 1, 1.0), (1, 0, 1.0),
                                    (2, 3, sign), (3, 2, sign)):
            value = x[:, indices[other]] * scale * factor
            result = result.scatter_add(1, indices[dest][None].expand(x.shape[0], -1), value)
        return result

    def __call__(self, blocks):
        from mace_ictc.hp.operators import _eager
        sample = blocks[0]
        _eager(sample)
        if sample.dtype != torch.float32 or sample.device != self.weights.device:
            raise RuntimeError("HP density dtype/device changed: reconfigure the model")
        if self.owner.quadratic_form != self.form or any(
                id(getattr(self.owner, name)) != identity or getattr(self.owner, name)._version != version
                for name, identity, version in self.constants):
            raise RuntimeError("Hermitian constants changed: reconfigure the HP density plan")
        batch_shape = sample.shape[:-3]
        x = torch.cat([blocks[l].flatten(-2) for l in range(self.lmax+1)], -1)
        result = _Density.apply(x.reshape(-1, 2*self.xhalf), self)
        output, offset = {}, 0
        for l, size in enumerate(self.sizes):
            output[l] = result[:, offset:offset+size].reshape(
                *batch_shape, self.owner.input_multiplicities[l], 2*l+1)
            offset += size
        return output
