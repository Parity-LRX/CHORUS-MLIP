"""Single-pass FP32 periodic edge vectors and force accumulation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _vectors(P, BATCH, SRC, DST, SHIFT, CELL, V, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < TOTAL
    edge, component = i//3, i%3
    src, dst = tl.load(SRC+edge, valid, 0), tl.load(DST+edge, valid, 0)
    batch = tl.load(BATCH+src, valid, 0)
    shift = tl.load(SHIFT+edge*3, valid, 0.)*tl.load(CELL+batch*9+component, valid, 0.)
    shift += tl.load(SHIFT+edge*3+1, valid, 0.)*tl.load(CELL+batch*9+3+component, valid, 0.)
    shift += tl.load(SHIFT+edge*3+2, valid, 0.)*tl.load(CELL+batch*9+6+component, valid, 0.)
    value = tl.load(P+dst*3+component, valid, 0.)-tl.load(P+src*3+component, valid, 0.)
    tl.store(V+i, value+shift, valid)


@triton.jit
def _forces(D, SRC, DST, F, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < TOTAL
    edge, component = i//3, i%3
    src, dst = tl.load(SRC+edge, valid, 0), tl.load(DST+edge, valid, 0)
    derivative = tl.load(D+i, valid, 0.)
    tl.atomic_add(F+src*3+component, derivative, valid, sem='relaxed')
    tl.atomic_add(F+dst*3+component, -derivative, valid, sem='relaxed')


def edge_vectors(positions, batch, src, dst, shifts, cell):
    n, e = positions.shape[0], src.numel()
    if (positions.shape != (n, 3) or batch.shape != (n,) or src.shape != (e,) or dst.shape != (e,)
            or shifts.shape != (e, 3) or cell.ndim != 3 or cell.shape[1:] != (3, 3)):
        raise ValueError('Invalid fused periodic graph shapes')
    if any(value.dtype != torch.int64 or value.device != positions.device for value in (batch, src, dst)):
        raise ValueError('Fused geometry requires same-device int64 graph indices')
    if not all(value.is_contiguous() for value in (positions, batch, src, dst, shifts, cell)):
        raise ValueError('Fused geometry requires contiguous live input tensors')
    out = positions.new_empty(src.numel(), 3)
    if out.numel():
        _vectors[(triton.cdiv(out.numel(), 256),)](positions, batch, src, dst, shifts,
            cell, out, out.numel(), 256, enable_fp_fusion=False)
    return out


def scatter_forces(derivative, src, dst, positions):
    force = torch.zeros_like(positions)
    if derivative.numel():
        _forces[(triton.cdiv(derivative.numel(), 256),)](derivative.contiguous(), src,
            dst, force, derivative.numel(), 256, enable_fp_fusion=False)
    return force


@triton.jit
def _gather_forces(D, SPTR, SORDER, DPTR, DORDER, F, BLOCK: tl.constexpr):
    node = tl.program_id(0)
    neighbor = tl.arange(0, BLOCK)[:, None]
    component = tl.arange(0, 4)[None, :]
    sb, se = tl.load(SPTR+node), tl.load(SPTR+node+1)
    db, de = tl.load(DPTR+node), tl.load(DPTR+node+1)
    count = tl.maximum(se-sb, de-db)
    total = tl.full((4,), 0., tl.float32)
    correction = tl.full((4,), 0., tl.float32)
    for start in range(0, count, BLOCK):
        si, di = sb+start+neighbor, db+start+neighbor
        source = tl.load(SORDER+si, si<se, 0)
        destination = tl.load(DORDER+di, di<de, 0)
        positive = tl.load(D+source*3+component, (si<se)&(component<3), 0.)
        negative = tl.load(D+destination*3+component, (di<de)&(component<3), 0.)
        value = tl.sum(positive-negative, axis=0)
        adjusted = value-correction
        updated = total+adjusted
        correction = (updated-total)-adjusted
        total = updated
    c = tl.arange(0, 4)
    tl.store(F+node*3+c, total, c<3)


def gather_forces(derivative, src, dst, positions):
    """Atom-owned live CSR force reduction, entirely FP32 and without atomics."""
    if (derivative.shape != (src.numel(), 3) or src.shape != dst.shape
            or derivative.dtype != torch.float32 or derivative.device != positions.device
            or positions.dtype != torch.float32 or not positions.is_cuda):
        raise ValueError('CSR force reduction requires matching CUDA FP32 [E,3] and [N,3]')
    from mace_ictc.models.message_topology import stable_topology
    force = torch.empty_like(positions)
    if src.numel():
        sptr, sorder = stable_topology(src, positions.shape[0])
        dptr, dorder = stable_topology(dst, positions.shape[0])
        _gather_forces[(positions.shape[0],)](derivative.contiguous(), sptr, sorder,
            dptr, dorder, force, 64, num_warps=4, enable_fp_fusion=False)
    else:force.zero_()
    return force
