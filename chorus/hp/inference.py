"""Conservative short-range E/F/virial through the live ICTC edge adjoint."""
import torch
from .runtime import _core


def energy_forces_virial(model, positions, atomic_numbers, batch, src, dst,
                         shifts, cell, *, fused_geometry=None, fused_forces=True,
                         force_reduction='atomic'):
    """Return E[B], F[N,3], virial[B,3,3] and atomic E[N], without E0.

    Virial is minus the derivative under row-coordinate deformation r->r@(I+eps),
    at fixed neighbor identities. This native short-range interface does not
    support additive long-range or dispersion modules. FP32 frozen inference
    only; virial reduction/output use FP64, as in DPA4, to avoid cancellation
    and atomic-sum error for an extensive tensor. The neural model and forces
    remain FP32. Use coordinate-autograd for force training.
    """
    core = _core(model)
    if core is not model or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("E/F/virial requires an unwrapped, already frozen eval backbone")
    if core.long_range_module is not None or core.dispersion is not None:
        raise ValueError("Edge-only virial does not cover additive long-range/dispersion")
    sample = next(core.parameters())
    if sample.dtype != torch.float32 or not sample.is_cuda or any(
            value.device != sample.device or value.dtype != torch.float32
            for value in (positions, shifts, cell)):
        raise ValueError("E/F/virial requires model and geometry to be CUDA FP32")
    if torch.is_inference_mode_enabled():
        raise ValueError("Forces require autograd; do not use inference_mode")
    if force_reduction not in ('atomic', 'csr'):
        raise ValueError('Force reduction must be atomic or csr')
    if fused_geometry is None:
        fused_geometry = bool(getattr(core, '_hp_large_graph_fusion', False) and src.numel() >= 327680
            and all(value.is_contiguous() for value in (positions, batch, src, dst, shifts, cell)))
    with torch.enable_grad():
        if fused_geometry:
            from .edge_geometry import edge_vectors, scatter_forces
            vec = edge_vectors(positions, batch, src, dst, shifts, cell).detach().requires_grad_(True)
            edge_batch = batch.index_select(0, src) if cell.shape[0] > 1 else None
        else:
            edge_batch = batch.index_select(0, src)
            cells = cell.index_select(0, edge_batch)
            vec = positions.index_select(0, dst)-positions.index_select(0, src)
            vec = (vec+torch.einsum("ei,eij->ej", shifts, cells)).detach().requires_grad_(True)
        value = core(positions, atomic_numbers, batch, src, dst, shifts, cell,
                     precomputed_edge_vec=vec)
        atomic = (value[0] if isinstance(value, tuple) else value).reshape(positions.shape[0])
        energy = atomic.new_zeros(cell.shape[0]).index_add(0, batch, atomic)
        derivative = torch.autograd.grad(energy.sum()+vec.sum()*0.0, vec)[0]
        if force_reduction == 'csr':
            from .edge_geometry import gather_forces
            force = gather_forces(derivative, src, dst, positions)
        else:
            force = (scatter_forces(derivative, src, dst, positions) if fused_geometry and fused_forces else
                torch.zeros_like(positions).index_add(0, src, derivative).index_add(0, dst, -derivative))
        # One outer product per edge, then reduction by structure. Its sign
        # matches -dE/dstrain, not stress (no division by cell volume).
        if cell.shape[0] == 1:
            virial = -(vec.double().T@derivative.double())[None]
        else:
            edge_virial = -vec.double()[:, :, None]*derivative.double()[:, None, :]
            virial = edge_virial.new_zeros(cell.shape[0], 3, 3).index_add(0, edge_batch, edge_virial)
        return {"energy": energy.detach(), "forces": force.detach(),
                "atomic_energy": atomic.detach(), "virial": virial.detach()}
