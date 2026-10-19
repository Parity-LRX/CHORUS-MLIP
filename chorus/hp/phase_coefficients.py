"""Dedicated CHORUS scalar-doublet kernel with differentiable exact adjoints."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


_MODES = {"polar": 0, "aligned": 1, "positive": 2, "signed": 3, "cartesian": 4}


@triton.jit
def _polar_mixed(T, R, GC, GS, HT, HR, DT, DR, DGC, DGS,
                 N: tl.constexpr, HAS_R: tl.constexpr, HAS_T: tl.constexpr, HAS_A: tl.constexpr,
                 NEED_T: tl.constexpr, NEED_R: tl.constexpr, NEED_C: tl.constexpr, NEED_S: tl.constexpr,
                 BLOCK: tl.constexpr):
    j = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = j<N
    theta = tl.load(T+j, valid, 0.)
    raw = tl.load(R+j, valid, 0.) if HAS_R else tl.full((BLOCK,), 0., tl.float32)
    amplitude = tl.where(raw>20., raw, tl.maximum(raw, 0.)+
                         libdevice.log1p(tl.exp(-tl.abs(raw)))) if HAS_R else 1.
    sigmoid = tl.where(raw>20., 1., 1./(1.+tl.exp(-raw)))
    second = tl.where(raw>20., 0., sigmoid*(1.-sigmoid))
    ct, st = libdevice.cos(theta), libdevice.sin(theta)
    ht = tl.load(HT+j, valid, 0.) if HAS_T else tl.full((BLOCK,), 0., tl.float32)
    hr = tl.load(HR+j, valid, 0.) if HAS_A else tl.full((BLOCK,), 0., tl.float32)
    if NEED_T or NEED_R:
        gc, gs = tl.load(GC+j, valid, 0.), tl.load(GS+j, valid, 0.)
        radial, angular = gc*ct+gs*st, -gc*st+gs*ct
    if NEED_T:
        tl.store(DT+j, -(ht*amplitude)*radial+(hr*sigmoid)*angular, valid)
    if NEED_R:
        tl.store(DR+j, (ht*sigmoid)*angular+(hr*second)*radial, valid)
    if NEED_C:
        tl.store(DGC+j, -(ht*amplitude)*st+(hr*sigmoid)*ct, valid)
    if NEED_S:
        tl.store(DGS+j, (ht*amplitude)*ct+(hr*sigmoid)*st, valid)


class _PolarAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, raw, gc, gs, has_raw, needed):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(theta, raw, gc, gs)
        ctx.has_raw = has_raw
        dt, dr = torch.empty_like(theta), torch.empty_like(raw) if has_raw else None
        if theta.numel():
            _phase[(triton.cdiv(theta.numel(), 256),)](
                theta, raw, gc, gs, theta, theta, dt, raw if dr is None else dr,
                theta.numel(), 0, has_raw, True, 256, enable_fp_fusion=False)
        return dt if needed[0] else None, dr if needed[1] else None

    @staticmethod
    def backward(ctx, ht, hr):
        from mace_ictc.models.adjoint_requirements import required_gradients
        theta, raw, gc, gs = ctx.saved_tensors
        needs = required_gradients((theta, raw, gc, gs), ctx.needs_input_grad[:4])
        if ht is None and hr is None:
            return (None,)*6
        if torch.is_grad_enabled():
            amplitude = F.softplus(raw) if ctx.has_raw else 1.
            sigmoid = torch.where(raw>20., 1., raw.sigmoid()) if ctx.has_raw else 1.
            second = torch.where(raw>20., 0., sigmoid*(1.-sigmoid)) if ctx.has_raw else 0.
            ct, st = theta.cos(), theta.sin()
            ht = torch.zeros_like(theta) if ht is None else ht
            hr = torch.zeros_like(theta) if hr is None else hr
            radial, angular = gc*ct+gs*st, -gc*st+gs*ct
            derivatives = (-(ht*amplitude)*radial+(hr*sigmoid)*angular,
                           (ht*sigmoid)*angular+(hr*second)*radial,
                           -(ht*amplitude)*st+(hr*sigmoid)*ct,
                           (ht*amplitude)*ct+(hr*sigmoid)*st)
            result = tuple(value if need else None for value, need in zip(derivatives, needs))
        else:
            result = tuple(torch.empty_like(value) if need else None
                           for value, need in zip((theta, raw, gc, gs), needs))
            if theta.numel():
                _polar_mixed[(triton.cdiv(theta.numel(), 256),)](
                    theta, raw, gc, gs, theta if ht is None else ht.contiguous(),
                    theta if hr is None else hr.contiguous(),
                    *(theta if value is None else value for value in result),
                    theta.numel(), ctx.has_raw, ht is not None, hr is not None,
                    *needs, 256, enable_fp_fusion=False)
        return *result, None, None


@triton.jit
def _phase(T, R, GC, GS, C, S, DT, DR, N: tl.constexpr, MODE: tl.constexpr,
           HAS_R: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < N
    theta = tl.load(T+i, valid, 0.)
    raw = tl.load(R+i, valid, 0.) if HAS_R else tl.full((BLOCK,), 0., tl.float32)
    amplitude = (tl.where(raw > 20., raw, tl.maximum(raw, 0.)+
                         libdevice.log1p(tl.exp(-tl.abs(raw))))) if HAS_R else 1.
    sigmoid = tl.where(raw > 20., 1., 1./(1.+tl.exp(-raw)))
    cosine, sine = libdevice.cos(theta), libdevice.sin(theta)
    if BACKWARD:
        gc, gs = tl.load(GC+i, valid, 0.), tl.load(GS+i, valid, 0.)
        if MODE == 0:
            dt = amplitude*(-gc*sine+gs*cosine)
            dr = sigmoid*(gc*cosine+gs*sine)
        elif MODE == 1:
            dt, dr = tl.full((BLOCK,), 0., tl.float32), sigmoid*gc
        elif MODE == 2:
            value = raw+theta
            dt = tl.where(value > 20., 1., 1./(1.+tl.exp(-value)))*gc
            dr = dt
        elif MODE == 3:
            dt, dr = amplitude*gc, sigmoid*(1.+theta)*gc
        else:
            dt, dr = gc, gs
        tl.store(DT+i, dt, valid)
        if HAS_R:
            tl.store(DR+i, dr, valid)
    else:
        if MODE == 0:
            real, imag = amplitude*cosine, amplitude*sine
        elif MODE == 1:
            real, imag = amplitude, theta*0.
        elif MODE == 2:
            value = raw+theta
            real = tl.where(value > 20., value, tl.maximum(value, 0.)+
                            libdevice.log1p(tl.exp(-tl.abs(value))))
            imag = tl.full((BLOCK,), 0., tl.float32)
        elif MODE == 3:
            real, imag = amplitude*(1.+theta), tl.full((BLOCK,), 0., tl.float32)
        else:
            real, imag = 1.+theta, raw
        tl.store(C+i, real, valid)
        tl.store(S+i, imag, valid)


class _Coefficients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, raw, mode, mixed_adjoint):
        has_raw = raw is not None
        theta = theta.contiguous()
        raw = raw.contiguous() if has_raw else theta.new_empty(0)
        ctx.save_for_backward(theta, raw)
        ctx.has_raw, ctx.mode = has_raw, mode
        ctx.mixed_adjoint = mixed_adjoint
        real, imag = torch.empty_like(theta), torch.empty_like(theta)
        if theta.numel():
            _phase[(triton.cdiv(theta.numel(), 256),)](
                theta, raw, theta, theta, real, imag, theta, raw,
                theta.numel(), mode, has_raw, False, 256, enable_fp_fusion=False)
        return real, imag

    @staticmethod
    def backward(ctx, gc, gs):
        theta, raw = ctx.saved_tensors
        gc = torch.zeros_like(theta) if gc is None else gc.contiguous()
        gs = torch.zeros_like(theta) if gs is None else gs.contiguous()
        if torch.is_grad_enabled() and ctx.mode == 0 and ctx.mixed_adjoint:
            from mace_ictc.models.adjoint_requirements import required_gradients
            needed = required_gradients((theta, raw), ctx.needs_input_grad[:2])
            dt, dr = _PolarAdjoint.apply(theta, raw, gc, gs, ctx.has_raw, needed)
            return dt, dr, None, None
        if torch.is_grad_enabled():
            amplitude = F.softplus(raw) if ctx.has_raw else 1.
            sigmoid = torch.where(raw > 20., 1., torch.sigmoid(raw)) if ctx.has_raw else None
            if ctx.mode == 0:
                cosine, sine = torch.cos(theta), torch.sin(theta)
                dt = amplitude*(-gc*sine+gs*cosine)
                dr = sigmoid*(gc*cosine+gs*sine) if ctx.has_raw else None
            elif ctx.mode == 1:
                dt, dr = theta*0., sigmoid*gc
            elif ctx.mode == 2:
                value = raw+theta
                dt = torch.where(value > 20., 1., torch.sigmoid(value))*gc
                dr = dt
            elif ctx.mode == 3:
                dt, dr = amplitude*gc, sigmoid*(1.+theta)*gc
            else:
                dt, dr = gc, gs
        else:
            dt = torch.empty_like(theta)
            dr = torch.empty_like(raw) if ctx.has_raw else None
            if theta.numel():
                _phase[(triton.cdiv(theta.numel(), 256),)](
                    theta, raw, gc, gs, theta, theta, dt, raw if dr is None else dr,
                    theta.numel(), ctx.mode, ctx.has_raw, True, 256, enable_fp_fusion=False)
        return dt, dr, None, None


def phase_coefficients(theta, raw, mode, *, mixed_adjoint=False):
    if mode not in _MODES or (mode != "polar" and raw is None):
        raise ValueError("Unsupported CHORUS phase coefficient configuration")
    if not theta.is_cuda or theta.dtype != torch.float32 or (
            raw is not None and (raw.shape != theta.shape or raw.device != theta.device
                                 or raw.dtype != theta.dtype)):
        raise ValueError("Phase coefficients require matching CUDA FP32 heads")
    real, imag = _Coefficients.apply(theta, raw, _MODES[mode], mixed_adjoint)
    return real.unsqueeze(-1), imag.unsqueeze(-1)
