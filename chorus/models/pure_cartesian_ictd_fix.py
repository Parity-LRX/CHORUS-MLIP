from __future__ import annotations

import math
from typing import Dict, List

import opt_einsum_fx
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from chorus.models.ictd_irreps import (
    build_cg_tensor,
    EdgeWeightedPathPreservingTensorProduct,
    EquivariantChannelLinearSO3,
    EquivariantChannelLinearSO3Rect,
    HarmonicElementwiseProduct,
    HarmonicPathWeightedTensorProduct,
    direction_harmonics_all,
    ictd_u_matrix_so3,
)
from chorus.models.radial_basis import mace_radial_embedding, mace_polynomial_cutoff
from chorus.models.pure_cartesian_ictd_layers import (
    EquivariantScalarReadoutSO3,
    SO3BlockRMSNorm,
    _irreps_total_dim,
    _merge_irreps,
    _split_irreps,
    apply_channel_adapter_per_l,
    resolve_save_multiple_mix_channels,
)
from chorus.models._mace_symmetric_contraction import MaceSymmetricContraction
from chorus.models.mlp import MainNet
from chorus.utils.scatter import scatter
from chorus.models.long_range import build_long_range_module
from chorus.models.dispersion import build_long_range_dispersion, normalize_dispersion_mode


_CONTRACTION_BATCH_EXAMPLE = 10
_CONTRACTION_ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


def _node_type_indices(node_attrs: torch.Tensor) -> torch.Tensor:
    if node_attrs.dim() == 1:
        return node_attrs.long()
    return node_attrs.argmax(dim=-1).long()


def _init_contraction_basis_logits_(logits: torch.Tensor, first_order_logit: float = 4.0) -> None:
    """Start from a stable order-1-dominant contraction instead of free large mixing."""
    with torch.no_grad():
        logits.zero_()
        logits[:, 0, :].fill_(float(first_order_logit))


def _init_contraction_basis_weight_(weight: torch.Tensor, higher_order_std: float = 0.02) -> None:
    """Free ablation: order-1 starts as passthrough, higher orders start small."""
    with torch.no_grad():
        weight.zero_()
        weight[:, 0, :].fill_(1.0)
        if weight.shape[1] > 1 and higher_order_std > 0:
            weight[:, 1:, :].normal_(mean=0.0, std=float(higher_order_std))


def _init_contraction_path_weight_(weight: torch.Tensor, std: float = 0.02) -> None:
    with torch.no_grad():
        weight.fill_(1.0)
        if std > 0:
            weight.add_(torch.randn_like(weight) * float(std))


def _init_path_tp_weight_to_one_(module: nn.Module | None) -> None:
    """ICTC path TPs multiply radial/contraction weights; do not start them near zero."""
    if module is not None and hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        with torch.no_grad():
            module.weight.fill_(1.0)


def _init_linear_identity_(module: nn.Module | None) -> None:
    if not isinstance(module, nn.Linear):
        return
    if module.weight.shape[0] != module.weight.shape[1]:
        return
    with torch.no_grad():
        nn.init.eye_(module.weight)
        if module.bias is not None:
            module.bias.zero_()


def _add_product_self_connection(
    out: torch.Tensor,
    sc: torch.Tensor,
    *,
    channels: int,
    target_lmax: int,
    input_lmax: int,
) -> torch.Tensor:
    """Add a MACE-style skip path to an SO(3) flattened product output.

    MACE's first residual interaction can produce an l=0-only self connection from
    scalar element embeddings, while the product output may still contain higher-l
    blocks. In that case the skip is added only to the output l=0 block.
    """
    if sc.shape[-1] == out.shape[-1]:
        return out + sc
    if sc.shape[-1] == channels:
        if target_lmax == 0:
            return out + sc
        out_blocks = _split_irreps(out, channels, target_lmax)
        out_blocks[0] = out_blocks[0] + sc.reshape(*sc.shape[:-1], channels, 1)
        return _merge_irreps(out_blocks, channels, target_lmax)
    if target_lmax == 0:
        return out + _split_irreps(sc, channels, input_lmax)[0].squeeze(-1)
    raise ValueError(f"Cannot add sc shape {tuple(sc.shape)} to product output {tuple(out.shape)}")


def _init_so3_linear_identity_(module: nn.Module | None) -> None:
    adapters = getattr(module, "adapters", None)
    if adapters is None:
        return
    for adapter in adapters.values():
        _init_linear_identity_(adapter)


def _init_so3_linear_mace_style_(module: nn.Module | None) -> None:
    adapters = getattr(module, "adapters", None)
    if adapters is None:
        return
    for adapter in adapters.values():
        with torch.no_grad():
            nn.init.normal_(adapter.weight, mean=0.0, std=1.0 / math.sqrt(float(adapter.in_features)))
            if adapter.bias is not None:
                adapter.bias.zero_()


def _init_element_conditioned_identity_(module: nn.Module | None) -> None:
    weights = getattr(module, "weights", None)
    if weights is None:
        return
    with torch.no_grad():
        for weight in weights.values():
            if weight.shape[-2] != weight.shape[-1]:
                continue
            weight.zero_()
            eye = torch.eye(weight.shape[-1], dtype=weight.dtype, device=weight.device)
            weight.copy_(eye.unsqueeze(0).expand_as(weight))
        bias = getattr(module, "bias", None)
        if bias is not None:
            for value in bias.values():
                value.zero_()


def _init_element_conditioned_mace_style_(module: nn.Module | None) -> None:
    weights = getattr(module, "weights", None)
    if weights is None:
        return
    with torch.no_grad():
        for weight in weights.values():
            fan_in = float(weight.shape[-1])
            nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(max(fan_in, 1.0)))
        bias = getattr(module, "bias", None)
        if bias is not None:
            for value in bias.values():
                value.zero_()


def _hidden_irreps(channels: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{int(channels)}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(int(lmax) + 1)))


def _mace_like_conv_tp_path_scales(
    tp: EdgeWeightedPathPreservingTensorProduct,
    *,
    channels: int,
    input_lmax: int,
    edge_lmax: int,
    n_probe: int = 6,
    seed: int = 12345,
) -> list[float]:
    """Recover per-path scalars mapping ICTC conv-TP paths to e3nn/MACE paths.

    MACE's convolution tensor product keeps every `(l_node, l_edge) -> l_out`
    path as a separate multiplicity block and applies e3nn's internal path
    normalization. ICTC's irreducible Cartesian CG tensors are equivalent but not
    necessarily in the same scalar convention. The converter calibrates the same
    constants from an actual MACE block; this helper builds the matching e3nn
    TensorProduct directly so from-scratch ICTC training can start in the same
    effective path scale.
    """
    from chorus.mace_basis import orthogonal_Q_blocks

    param_dtype = tp.weight.dtype
    calib_dtype = torch.float64
    device = tp.weight.device
    C = int(channels)
    paths = [tuple(int(v) for v in p) for p in tp.paths]
    edge_lmax = int(edge_lmax)
    input_lmax = int(input_lmax)

    irreps_in1 = _hidden_irreps(C, input_lmax)
    irreps_in2 = o3.Irreps.spherical_harmonics(edge_lmax)
    irreps_out = o3.Irreps(
        [
            (
                C,
                o3.Irrep(int(l3), 1 if int(l3) % 2 == 0 else -1),
            )
            for _l1, _l2, l3 in paths
        ]
    )
    instructions = []
    for out_idx, (l1, l2, _l3) in enumerate(paths):
        if l1 > input_lmax:
            continue
        instructions.append((int(l1), int(l2), int(out_idx), "uvu", True))
    ref_tp = o3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        irrep_normalization="component",
        path_normalization="element",
        internal_weights=False,
        shared_weights=False,
    ).to(device=device, dtype=calib_dtype)

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    N = int(n_probe)
    q_blocks = orthogonal_Q_blocks(edge_lmax, dtype=calib_dtype, device=device)
    h_ictd = {
        l: torch.randn(N, C, 2 * l + 1, generator=g, dtype=calib_dtype).to(device)
        for l in range(input_lmax + 1)
    }
    h_e3nn = torch.cat(
        [
            torch.einsum("ncm,mp->ncp", h_ictd[l], q_blocks[l]).reshape(N, C * (2 * l + 1))
            for l in range(input_lmax + 1)
        ],
        dim=-1,
    )
    ndir = torch.randn(N, 3, generator=g, dtype=calib_dtype).to(device)
    ndir = ndir / ndir.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y_ictd = direction_harmonics_all(ndir, edge_lmax)
    y_e3nn = torch.cat([y_ictd[l] @ q_blocks[l] for l in range(edge_lmax + 1)], dim=-1)

    weights = torch.randn(N, len(paths), C, generator=g, dtype=calib_dtype).to(device)
    ref_weights = weights.reshape(N, len(paths) * C)
    old_weight = tp.weight.detach().clone()
    old_compute_dtype = tp.internal_compute_dtype
    with torch.no_grad():
        try:
            tp.internal_compute_dtype = calib_dtype
            tp.weight.fill_(1.0)
            ref_out = ref_tp(h_e3nn, y_e3nn, ref_weights)
            edge_attrs = {l: y_ictd[l].unsqueeze(-2) for l in range(edge_lmax + 1)}
            ictd_out = tp(h_ictd, edge_attrs, ref_weights)
        finally:
            tp.weight.copy_(old_weight.to(dtype=param_dtype, device=device))
            tp.internal_compute_dtype = old_compute_dtype
            tp._cg_cache_by_dev_dtype.clear()
            tp._proj_group_cache_by_dev_dtype.clear()
            tp._proj_group_view_cache_by_dev_dtype.clear()

    scales = [1.0 for _ in paths]
    idx = 0
    max_resid = 0.0
    for path_idx, (_l1, _l2, l3) in enumerate(paths):
        d = C * (2 * int(l3) + 1)
        if int(_l1) > input_lmax:
            idx += d
            continue
        ref_block = ref_out[:, idx : idx + d].reshape(N, C, 2 * int(l3) + 1)
        off = int(tp.path_offset[path_idx])
        ictd_block = ictd_out[int(l3)][:, off * C : (off + 1) * C, :]
        ictd_block = torch.einsum("ncm,mp->ncp", ictd_block, q_blocks[int(l3)])
        num = (ref_block * ictd_block).sum()
        den = (ictd_block * ictd_block).sum().clamp_min(1e-30)
        scale = float((num / den).item())
        scales[path_idx] = scale
        max_resid = max(max_resid, float((ref_block - scale * ictd_block).abs().max().item()))
        idx += d
    if max_resid > 1e-6:
        raise RuntimeError(
            f"e3nn conv-TP path calibration residual too large ({max_resid:.2e}); "
            "ICTC and e3nn path conventions do not match."
        )
    return scales


def _so3_flat_to_mace_features(x: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    blocks = _split_irreps(x, int(channels), int(lmax))
    return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)


def _merge_blocks_subset(blocks: Dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    return torch.cat([blocks[l].reshape(blocks[l].shape[0], int(channels) * (2 * l + 1)) for l in range(int(lmax) + 1)], dim=-1)


def _concat_so3_states_by_l(states: List[torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    """
    Concatenate multiple SO3-flat states by channel within each l-block.

    Each state is laid out as [l0 | l1 | ...]. Directly concatenating states
    along the flat dimension would produce [s0_l0 | s0_l1 | ... | s1_l0 | ...],
    which is not a valid SO3-flat layout for a larger channel count. Equivariant
    operators expect [all_l0_channels | all_l1_channels | ...].
    """
    if len(states) == 0:
        raise ValueError("states must contain at least one SO3-flat tensor")
    split_states = [_split_irreps(state, int(channels), int(lmax)) for state in states]
    parts = []
    for l in range(int(lmax) + 1):
        block = torch.cat([split_state[l] for split_state in split_states], dim=-2)
        parts.append(block.reshape(*block.shape[:-2], block.shape[-2] * block.shape[-1]))
    return torch.cat(parts, dim=-1)


def _so3_block_rmsnorm(
    x: torch.Tensor,
    channels: int,
    lmax: int,
    gamma: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Apply an equivariant RMS normalization independently inside each l block."""
    blocks = _split_irreps(x, int(channels), int(lmax))
    parts = []
    gamma = gamma.to(dtype=x.dtype, device=x.device)
    for l in range(int(lmax) + 1):
        block = blocks[l]
        rms = block.square().mean(dim=(-2, -1), keepdim=True).add(float(eps)).sqrt()
        block = block / rms * gamma[l].view(*([1] * (block.ndim - 2)), 1, 1)
        parts.append(block.reshape(*block.shape[:-2], block.shape[-2] * block.shape[-1]))
    return torch.cat(parts, dim=-1)


def _tp_allowed_paths_from_target_lmax(lmax_in1: int, lmax_in2: int, lmax_target: int) -> List[tuple[int, int, int]]:
    """
    Mirror MACE/e3nn instruction pruning at the SO3 level:
    keep only paths (l1, l2, l3) whose output irrep l3 is present in the target set.

    For our current ICTC fix baseline, target irreps are exactly all l=0..lmax_target.
    This helper still makes the path set explicit and keeps interaction TP
    aligned with the same contract used by MACE TensorProduct instructions.
    """
    paths: List[tuple[int, int, int]] = []
    target_ls = set(range(int(lmax_target) + 1))
    for l1 in range(int(lmax_in1) + 1):
        for l2 in range(int(lmax_in2) + 1):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                if l3 not in target_ls:
                    continue
                if l3 > int(lmax_target):
                    continue
                if (l1 + l2 + l3) % 2 == 1:
                    continue
                paths.append((l1, l2, l3))
    return paths


def _tp_allowed_paths_to_output_l(lmax_in1: int, lmax_in2: int, output_l: int) -> List[tuple[int, int, int]]:
    paths: List[tuple[int, int, int]] = []
    l3 = int(output_l)
    for l1 in range(int(lmax_in1) + 1):
        for l2 in range(int(lmax_in2) + 1):
            if not (abs(l1 - l2) <= l3 <= l1 + l2):
                continue
            if (l1 + l2 + l3) % 2 == 1:
                continue
            paths.append((l1, l2, l3))
    return paths


class ElementConditionedLinearSO3(nn.Module):
    def __init__(self, num_elements: int, channels: int, lmax: int, bias: bool = False):
        super().__init__()
        self.num_elements = int(num_elements)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.dim = _irreps_total_dim(self.channels, self.lmax)
        self.weights = nn.ParameterDict(
            {
                str(l): nn.Parameter(torch.randn(self.num_elements, self.channels, self.channels) * 0.02)
                for l in range(self.lmax + 1)
            }
        )
        if bias:
            self.bias = nn.ParameterDict(
                {
                    str(l): nn.Parameter(torch.zeros(self.num_elements, self.channels))
                    for l in range(self.lmax + 1)
                }
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        attrs = node_attrs.to(dtype=x.dtype)
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            weight = self.weights[str(l)].to(dtype=x.dtype)
            mixed_weight = torch.einsum("ne,eoi->noi", attrs, weight)
            out_block = torch.einsum("noi,nid->nod", mixed_weight, blocks[l])
            if self.bias is not None:
                mixed_bias = torch.einsum("ne,eo->no", attrs, self.bias[str(l)].to(dtype=x.dtype))
                out_block = out_block + mixed_bias.unsqueeze(-1)
            out_blocks[l] = out_block
        return _merge_irreps(out_blocks, self.channels, self.lmax)

    def forward_type_idx(self, x: torch.Tensor, node_type_idx: torch.Tensor) -> torch.Tensor:
        idx = node_type_idx.to(device=x.device, dtype=torch.long)
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            weight = self.weights[str(l)].to(dtype=x.dtype, device=x.device)
            mixed_weight = weight.index_select(0, idx)
            out_block = torch.einsum("noi,nid->nod", mixed_weight, blocks[l])
            if self.bias is not None:
                bias = self.bias[str(l)].to(dtype=x.dtype, device=x.device)
                mixed_bias = bias.index_select(0, idx)
                out_block = out_block + mixed_bias.unsqueeze(-1)
            out_blocks[l] = out_block
        return _merge_irreps(out_blocks, self.channels, self.lmax)


class ElementConditionedScalarLinear(nn.Module):
    """Element-conditioned rectangular linear map between invariant features."""

    def __init__(self, num_elements: int, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.num_elements = int(num_elements)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(
            torch.empty(self.num_elements, self.out_features, self.in_features)
        )
        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=1.0 / math.sqrt(float(max(self.in_features, 1))),
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.num_elements, self.out_features)) if bias else None
        )

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_type_idx is not None:
            idx = node_type_idx.to(device=x.device, dtype=torch.long)
            weight = self.weight.to(dtype=x.dtype, device=x.device).index_select(0, idx)
            out = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)
            if self.bias is not None:
                out = out + self.bias.to(dtype=x.dtype, device=x.device).index_select(0, idx)
            return out
        if node_attrs is None:
            raise ValueError("node_attrs is required when node_type_idx is not provided")
        attrs = node_attrs.to(dtype=x.dtype, device=x.device)
        weight = torch.einsum(
            "ne,eoi->noi", attrs, self.weight.to(dtype=x.dtype, device=x.device)
        )
        out = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)
        if self.bias is not None:
            out = out + torch.einsum(
                "ne,eo->no", attrs, self.bias.to(dtype=x.dtype, device=x.device)
            )
        return out


class SO3DoubletRMSNorm(nn.Module):
    """Joint RMS norm for a real pair representing one complex SO(3) feature.

    A shared norm and gain are essential here: independently normalizing the real
    and imaginary streams would not commute with a global U(1) rotation.
    """

    def __init__(self, channels: int, lmax: int, eps: float = 1.0e-8):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.eps = float(eps)
        self.gain = nn.Parameter(
            torch.ones(self.lmax + 1, self.channels, dtype=torch.get_default_dtype())
        )

    def forward(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_blocks = _split_irreps(real, self.channels, self.lmax)
        imag_blocks = _split_irreps(imag, self.channels, self.lmax)
        real_out: Dict[int, torch.Tensor] = {}
        imag_out: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            re = real_blocks[l]
            im = imag_blocks[l]
            rms = (
                (re.square() + im.square())
                .mean(dim=(-1, -2), keepdim=True)
                .add(self.eps)
                .sqrt()
            )
            gain = self.gain[l].view(
                *([1] * (re.ndim - 2)), self.channels, 1
            ).to(dtype=re.dtype, device=re.device)
            real_out[l] = re * (gain / rms)
            imag_out[l] = im * (gain / rms)
        return (
            _merge_irreps(real_out, self.channels, self.lmax),
            _merge_irreps(imag_out, self.channels, self.lmax),
        )


class PersistentChargedUpdate(nn.Module):
    """Mix a previous and an incoming q=1 SO(3) doublet across network depth.

    Every channel map and per-l gate is shared by the real and imaginary
    components.  Consequently a common U(1) rotation applied to both inputs
    commutes with this update.  This is a global-U(1) charged memory update; it
    does not implement independently transforming node gauges or link transport.
    """

    def __init__(self, *, channels: int, lmax: int):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.previous_linear = EquivariantChannelLinearSO3(
            self.channels, self.lmax, bias=False
        )
        self.incoming_linear = EquivariantChannelLinearSO3(
            self.channels, self.lmax, bias=False
        )
        _init_so3_linear_identity_(self.previous_linear)
        _init_so3_linear_identity_(self.incoming_linear)
        # sigmoid(0)=0.5 starts as an equal-depth mixture while keeping the
        # charged magnitude bounded when another interaction is added.
        self.memory_logits = nn.Parameter(
            torch.zeros(self.lmax + 1, dtype=torch.get_default_dtype())
        )

    def forward(
        self,
        previous_real: torch.Tensor,
        previous_imag: torch.Tensor,
        incoming_real: torch.Tensor,
        incoming_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous = torch.stack((previous_real, previous_imag), dim=-2)
        incoming = torch.stack((incoming_real, incoming_imag), dim=-2)
        charged = self.forward_doublet(previous, incoming)
        return charged[..., 0, :], charged[..., 1, :]

    def forward_doublet(
        self,
        previous: torch.Tensor,
        incoming: torch.Tensor,
    ) -> torch.Tensor:
        previous_blocks = _split_irreps(previous, self.channels, self.lmax)
        incoming_blocks = _split_irreps(incoming, self.channels, self.lmax)
        gates = self.memory_logits.to(
            dtype=previous.dtype, device=previous.device
        ).sigmoid()
        out_blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            gate = gates[l]
            # Concatenate the two charged sources along the channel axis and
            # their gated SO(3) channel maps along the input axis.  This is
            # algebraically identical to
            #
            #   g * W_prev(previous) + (1-g) * W_in(incoming),
            #
            # but launches one GEMM per l instead of two and avoids an
            # intermediate merge/split of both complete irreps layouts.
            source = torch.cat((previous_blocks[l], incoming_blocks[l]), dim=-2)
            previous_weight = self.previous_linear.adapters[str(l)].weight.to(
                dtype=source.dtype, device=source.device
            )
            incoming_weight = self.incoming_linear.adapters[str(l)].weight.to(
                dtype=source.dtype, device=source.device
            )
            fused_weight = torch.cat(
                (gate * previous_weight, (1.0 - gate) * incoming_weight), dim=-1
            )
            out_blocks[l] = F.linear(
                source.movedim(-2, -1), fused_weight
            ).movedim(-1, -2)
        return _merge_irreps(out_blocks, self.channels, self.lmax)


class PhaseHermitianScalarResidual(nn.Module):
    """Contract a complex SO(3) doublet to a real, element-conditioned scalar residual.

    ``real`` and ``imag`` encode a complex atomic environment without using a
    complex dtype.  The contraction

        <real, real>_l + <imag, imag>_l

    is exactly invariant under a shared U(1) rotation of the doublet and under
    SO(3).  It also exposes the intended pair factor cos(theta_j - theta_k).
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        residual_scale_init: float = 0.05,
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.hermitian_product = HarmonicElementwiseProduct(
            lmax=self.lmax,
            mul=self.channels,
            irreps_out="0e",
            normalization="component",
            internal_compute_dtype=internal_compute_dtype,
        )
        # Non-persistent because it is fully determined by lmax and the
        # normalization convention.  Keeping hermitian_product above preserves
        # the old module structure/checkpoint contract, while the fused forward
        # avoids evaluating the same scalar product once for real and once for
        # imaginary features.
        self.register_buffer(
            "hermitian_0e_factors",
            torch.as_tensor(
                self.hermitian_product._0e_factors,
                # Preserve the Python/CG constants at full precision; module.to
                # will downcast for float32 training without making a later
                # float64 conversion irreversibly inherit float32 rounding.
                dtype=torch.float64,
            ).repeat_interleave(self.channels),
            persistent=False,
        )
        self.scalar_linear = ElementConditionedScalarLinear(
            num_elements=num_elements,
            in_features=self.channels * (self.lmax + 1),
            out_features=self.channels,
            bias=False,
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.get_default_dtype())
        )

    def hermitian_features(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return self.hermitian_features_doublet(torch.stack((real, imag), dim=-2))

    def hermitian_features_doublet(self, doublet: torch.Tensor) -> torch.Tensor:
        doublet_blocks = _split_irreps(doublet, self.channels, self.lmax)
        # Concatenating the U(1) doublet along m lets one square/reduction form
        # ||real_l||^2 + ||imag_l||^2.  Compared with two independent harmonic
        # products this halves reductions and removes one output concat/add.
        rho = torch.cat(
            [
                doublet_blocks[l].square().sum(dim=(-3, -1))
                for l in range(self.lmax + 1)
            ],
            dim=-1,
        )
        return rho * self.hermitian_0e_factors.to(dtype=rho.dtype, device=rho.device)

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        *,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rho = self.hermitian_features(real, imag)
        delta = self.scalar_linear(rho, node_attrs, node_type_idx)
        return self.residual_scale.to(dtype=delta.dtype, device=delta.device) * delta

    def forward_doublet(
        self,
        doublet: torch.Tensor,
        *,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rho = self.hermitian_features_doublet(doublet)
        delta = self.scalar_linear(rho, node_attrs, node_type_idx)
        return self.residual_scale.to(dtype=delta.dtype, device=delta.device) * delta


class PhaseHermitianFullLResidual(nn.Module):
    """Low-rank, full-L quadratic density mapped back to the SO(3) message layout.

    ``real`` and ``imag`` are a real doublet for one complex equivariant atomic
    environment.  A shared (real) channel projection first reduces every l block
    from ``channels`` to ``density_rank`` latent orbitals, which commutes with a
    U(1) rotation of the doublet.  For each natural-parity path

        (l1, l2) -> L <= lmax

    ``quadratic_form="hermitian"`` contracts ``z tensor z*`` and therefore
    produces U(1)-neutral blocks. ``quadratic_form="charge2"`` is the
    parameter-matched non-Hermitian control ``z tensor z``: it replaces
    ``(xx+yy, yx-xy)`` by ``(xx-yy, xy+yx)`` while retaining exactly the same
    channel projections, CG paths, output weights, and residual injection.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        density_rank: int = 8,
        residual_scale_init: float = 0.05,
        coherence_gate: bool = False,
        adaptive_coherence: bool = False,
        environment_adaptive_coherence: bool = False,
        adaptive_coherence_init: float = 0.1,
        quadratic_form: str = "hermitian",
        output_gate: bool = False,
        species_mode: str = "onehot-full",
        species_embedding_dim: int = 16,
        species_rank: int = 16,
    ):
        super().__init__()
        self.num_elements = int(num_elements)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.density_rank = int(density_rank)
        self.coherence_gate = bool(coherence_gate)
        self.adaptive_coherence = bool(adaptive_coherence)
        self.environment_adaptive_coherence = bool(environment_adaptive_coherence)
        self.adaptive_coherence_init = float(adaptive_coherence_init)
        self.quadratic_form = str(quadratic_form)
        self.output_gate = bool(output_gate)
        self.species_mode = str(species_mode)
        self.species_embedding_dim = int(species_embedding_dim)
        self.species_rank = int(species_rank)
        if self.quadratic_form not in {"hermitian", "charge2"}:
            raise ValueError(
                "quadratic_form must be 'hermitian' or 'charge2', "
                f"got {self.quadratic_form!r}"
            )
        if self.quadratic_form == "charge2" and (
            self.coherence_gate
            or self.adaptive_coherence
            or self.environment_adaptive_coherence
        ):
            raise ValueError(
                "charge2 is a full post-aggregation quadratic control and does "
                "not support Hermitian coherence gating"
            )
        if sum(
            (
                self.coherence_gate,
                self.adaptive_coherence,
                self.environment_adaptive_coherence,
            )
        ) > 1:
            raise ValueError(
                "coherence_gate, adaptive_coherence, and "
                "environment_adaptive_coherence are mutually exclusive"
            )
        if self.density_rank <= 0:
            raise ValueError(f"density_rank must be positive, got {self.density_rank}")
        if self.species_mode not in {"onehot-full", "embedded-lowrank"}:
            raise ValueError(
                "species_mode must be 'onehot-full' or 'embedded-lowrank', "
                f"got {self.species_mode!r}"
            )
        if self.species_embedding_dim <= 0:
            raise ValueError("species_embedding_dim must be positive")
        if self.species_rank <= 0:
            raise ValueError("species_rank must be positive")

        self.rank_projections = nn.ModuleDict(
            {
                str(l): nn.Linear(self.channels, self.density_rank, bias=False)
                for l in range(self.lmax + 1)
            }
        )
        for projection in self.rank_projections.values():
            nn.init.normal_(
                projection.weight,
                mean=0.0,
                std=1.0 / math.sqrt(float(max(self.channels, 1))),
            )

        # A path is (l1, l2, L, component, cg_buffer_name).  l1 <= l2 avoids
        # storing both Hermitian-conjugate channel blocks.  The imaginary part
        # is independent only for an off-diagonal (l1, l2) block.
        self.paths: list[tuple[int, int, int, str, str]] = []
        multiplicities = [0 for _ in range(self.lmax + 1)]
        path_idx = 0
        for l1 in range(self.lmax + 1):
            for l2 in range(l1, self.lmax + 1):
                for out_l in range(abs(l1 - l2), min(l1 + l2, self.lmax) + 1):
                    # The backbone carries natural-parity irreps only.  The
                    # Hermitian density parity is p1*p2, so keep exactly paths
                    # with (-1)^out_l = (-1)^(l1+l2).
                    if (l1 + l2 + out_l) % 2 == 1:
                        continue
                    cg = build_cg_tensor(l1, l2, out_l).to(dtype=torch.get_default_dtype())
                    cg_norm = cg.square().sum().sqrt().clamp_min(1.0e-30)
                    cg = cg * (math.sqrt(float(2 * out_l + 1)) / cg_norm)
                    cg_name = f"full_l_cg_{path_idx}"
                    self.register_buffer(cg_name, cg.contiguous(), persistent=False)
                    self.paths.append((l1, l2, out_l, "real", cg_name))
                    multiplicities[out_l] += self.density_rank
                    if l1 < l2:
                        self.paths.append((l1, l2, out_l, "imag", cg_name))
                        multiplicities[out_l] += self.density_rank
                    path_idx += 1

        self.input_multiplicities = tuple(int(value) for value in multiplicities)
        if any(value <= 0 for value in self.input_multiplicities):
            raise RuntimeError(
                "full-L Hermitian construction produced an empty output block: "
                f"lmax={self.lmax}, multiplicities={self.input_multiplicities}"
            )

        # Fuse every CG tensor that shares an (l1, l2) input pair.  The eager
        # implementation used to launch a separate einsum for every real/imag
        # term (22 tiny CUDA kernels for lmax=2).  A doublet contraction computes
        # all four complex components at once for off-diagonal pairs, while a
        # component-diagonal contraction directly forms xx + yy for l1 == l2.
        # Concatenating the output CG axes also folds multiple L targets for the
        # same input pair into one kernel.  Individual CG buffers and parameter
        # names are retained, so existing checkpoints remain strictly loadable.
        grouped_couplings: Dict[tuple[int, int], List[tuple[int, str]]] = {}
        for l1, l2, out_l, component, cg_name in self.paths:
            if component == "real":
                grouped_couplings.setdefault((l1, l2), []).append((out_l, cg_name))
        self.coupling_groups: list[
            tuple[int, int, str, tuple[tuple[int, str, int, int], ...]]
        ] = []
        for group_idx, ((l1, l2), specs) in enumerate(grouped_couplings.items()):
            pieces = [getattr(self, cg_name) for _, cg_name in specs]
            fused_cg = torch.cat(pieces, dim=-1).contiguous()
            fused_name = f"full_l_fused_cg_{group_idx}"
            self.register_buffer(fused_name, fused_cg, persistent=False)
            offset = 0
            slices: list[tuple[int, str, int, int]] = []
            for out_l, cg_name in specs:
                width = 2 * out_l + 1
                slices.append((out_l, cg_name, offset, offset + width))
                offset += width
            self.coupling_groups.append((l1, l2, fused_name, tuple(slices)))

        if self.species_mode == "onehot-full":
            self.output_weights = nn.ParameterDict()
            for out_l, in_mul in enumerate(self.input_multiplicities):
                weight = nn.Parameter(
                    torch.empty(self.num_elements, self.channels, int(in_mul))
                )
                nn.init.normal_(
                    weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(float(max(in_mul, 1))),
                )
                self.output_weights[str(out_l)] = weight
            self.shared_output_weights = None
            self.species_left = None
            self.species_right = None
            self.species_gate = None
        else:
            self.output_weights = None
            self.shared_output_weights = nn.ParameterDict()
            self.species_left = nn.ParameterDict()
            self.species_right = nn.ParameterDict()
            for out_l, in_mul in enumerate(self.input_multiplicities):
                in_mul = int(in_mul)
                shared = nn.Parameter(torch.empty(self.channels, in_mul))
                left = nn.Parameter(
                    torch.empty(self.channels, self.species_rank)
                )
                right = nn.Parameter(torch.empty(self.species_rank, in_mul))
                nn.init.normal_(
                    shared,
                    mean=0.0,
                    std=1.0 / math.sqrt(float(max(in_mul, 1))),
                )
                nn.init.normal_(
                    left,
                    mean=0.0,
                    std=1.0 / math.sqrt(float(self.species_rank)),
                )
                nn.init.normal_(
                    right,
                    mean=0.0,
                    std=1.0 / math.sqrt(float(max(in_mul, 1))),
                )
                self.shared_output_weights[str(out_l)] = shared
                self.species_left[str(out_l)] = left
                self.species_right[str(out_l)] = right
            self.species_gate = nn.Linear(
                self.species_embedding_dim, self.species_rank, bias=True
            )
            nn.init.normal_(
                self.species_gate.weight,
                mean=0.0,
                std=1.0 / math.sqrt(float(self.species_embedding_dim)),
            )
            nn.init.zeros_(self.species_gate.bias)
        self.residual_scale = nn.Parameter(
            torch.full(
                (self.lmax + 1,),
                float(residual_scale_init),
                dtype=torch.get_default_dtype(),
            )
        )
        if self.output_gate:
            gate_hidden_channels = max(16, min(self.channels, 64))
            self.output_gate_context = nn.Sequential(
                nn.LayerNorm(self.channels, elementwise_affine=False),
                nn.Linear(self.channels, gate_hidden_channels),
                _Normalize2MomSiLU(),
                nn.Linear(
                    gate_hidden_channels,
                    (self.lmax + 1) * self.channels,
                ),
            )
            gate_output = self.output_gate_context[-1]
            if not isinstance(gate_output, nn.Linear):
                raise RuntimeError("Hermitian output gate must end in nn.Linear")
            # 2*sigmoid(0)=1: the nonlinear mode starts as the ordinary
            # Hermitian residual and only adds environment selectivity after it
            # receives gradient signal.
            nn.init.zeros_(gate_output.weight)
            nn.init.zeros_(gate_output.bias)
        else:
            self.output_gate_context = None
        if self.coherence_gate:
            # rho = D + gamma_L C, where D contains j=k edge self-densities and
            # C = rho_full - D contains exactly the j!=k coherent cross terms.
            # gamma=1 reproduces the ordinary full Hermitian density at
            # initialization; only lmax+1 scalar parameters are added.
            self.coherence_scale = nn.Parameter(
                torch.ones(self.lmax + 1, dtype=torch.get_default_dtype())
            )
            self.register_parameter("coherence_logit", None)
            self.register_buffer("coherence_base_logit", None)
            self.coherence_context = None
        elif self.adaptive_coherence:
            if not 0.0 < self.adaptive_coherence_init < 1.0:
                raise ValueError(
                    "adaptive_coherence_init must lie strictly between zero and one, "
                    f"got {self.adaptive_coherence_init}"
                )
            # A bounded convex mixture preserves the diagonal/full Hermitian
            # endpoints while allowing every output L to select its own amount
            # of j!=k coherence.  A diagonal-biased initialization prevents the
            # noisier cross terms from dominating before they earn gradient signal.
            initial_logit = math.log(
                self.adaptive_coherence_init / (1.0 - self.adaptive_coherence_init)
            )
            self.register_parameter("coherence_scale", None)
            self.coherence_logit = nn.Parameter(
                torch.full(
                    (self.lmax + 1,),
                    initial_logit,
                    dtype=torch.get_default_dtype(),
                )
            )
            self.register_buffer("coherence_base_logit", None)
            self.coherence_context = None
        elif self.environment_adaptive_coherence:
            if not 0.0 < self.adaptive_coherence_init < 1.0:
                raise ValueError(
                    "adaptive_coherence_init must lie strictly between zero and one, "
                    f"got {self.adaptive_coherence_init}"
                )
            initial_logit = math.log(
                self.adaptive_coherence_init / (1.0 - self.adaptive_coherence_init)
            )
            self.register_parameter("coherence_scale", None)
            self.register_parameter("coherence_logit", None)
            self.register_buffer(
                "coherence_base_logit",
                torch.full(
                    (self.lmax + 1,),
                    initial_logit,
                    dtype=torch.get_default_dtype(),
                ),
            )
            hidden_channels = max(16, min(self.channels, 64))
            self.coherence_context = nn.Sequential(
                nn.LayerNorm(self.channels, elementwise_affine=False),
                nn.Linear(self.channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, self.lmax + 1),
            )
            final = self.coherence_context[-1]
            if not isinstance(final, nn.Linear):
                raise RuntimeError("environment coherence head must end in nn.Linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            self.register_parameter("coherence_scale", None)
            self.register_parameter("coherence_logit", None)
            self.register_buffer("coherence_base_logit", None)
            self.coherence_context = None

    def effective_coherence_scale(
        self, gate_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return gamma_L or gamma_iL used in D + gamma (rho_full - D)."""
        if self.coherence_context is not None:
            if gate_features is None:
                raise ValueError(
                    "environment-adaptive coherence requires invariant gate_features"
                )
            if gate_features.ndim != 2 or gate_features.shape[-1] != self.channels:
                raise ValueError(
                    "gate_features must have shape (num_nodes, channels), got "
                    f"{tuple(gate_features.shape)}"
                )
            if self.coherence_base_logit is None:
                raise RuntimeError("environment coherence base logit is missing")
            delta_logit = self.coherence_context(gate_features)
            base_logit = self.coherence_base_logit.to(
                dtype=delta_logit.dtype, device=delta_logit.device
            )
            return torch.sigmoid(base_logit[None, :] + delta_logit)
        if self.coherence_logit is not None:
            return torch.sigmoid(self.coherence_logit)
        if self.coherence_scale is not None:
            return self.coherence_scale
        raise RuntimeError("this Hermitian residual has no coherence gate")

    @staticmethod
    def _coherence_for_block(
        coherence_scale: torch.Tensor,
        out_l: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if coherence_scale.ndim == 1:
            scale = coherence_scale[out_l]
        elif coherence_scale.ndim == 2:
            scale = coherence_scale[:, out_l, None, None]
        else:
            raise ValueError(
                "coherence scale must have shape (L,) or (num_nodes, L), got "
                f"{tuple(coherence_scale.shape)}"
            )
        return scale.to(dtype=reference.dtype, device=reference.device)

    @staticmethod
    def _project_block(x: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        return projection(x.movedim(-2, -1)).movedim(-1, -2)

    @staticmethod
    def _couple(a: torch.Tensor, b: torch.Tensor, cg: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...rm,...rn,mnk->...rk", a, b, cg)

    def hermitian_blocks(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Backward-compatible alias for the configured quadratic blocks."""
        return self.quadratic_blocks(real, imag)

    def hermitian_blocks_doublet(
        self, doublet: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Backward-compatible alias for the configured quadratic blocks."""
        return self.quadratic_blocks_doublet(doublet)

    def quadratic_blocks(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        return self.quadratic_blocks_doublet(torch.stack((real, imag), dim=-2))

    def quadratic_blocks_doublet(
        self, doublet: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        doublet_in = _split_irreps(doublet, self.channels, self.lmax)
        # One projection kernel per l instead of separate real/imag kernels.
        # Shape: (..., U(1)=2, density_rank, 2l+1).
        rank_doublet = {
            l: self._project_block(
                doublet_in[l],
                self.rank_projections[str(l)],
            )
            for l in range(self.lmax + 1)
        }
        coupled_values: Dict[tuple[str, str], torch.Tensor] = {}
        for l1, l2, fused_name, slices in self.coupling_groups:
            left = rank_doublet[l1]
            right = rank_doublet[l2]
            cg = getattr(self, fused_name).to(dtype=left.dtype, device=left.device)
            if self.quadratic_form == "charge2":
                # Re[z tensor z] = xx - yy and Im[z tensor z] = xy + yx.
                # Computing all four components keeps the implementation
                # independent of basis/CG special cases. The retained path set
                # is identical to the Hermitian operator, so parameter count and
                # output layout are exactly matched.
                pair_value = torch.einsum(
                    "...arm,...brn,mnk->...abrk", left, right, cg
                )
                real_value = (
                    pair_value[..., 0, 0, :, :]
                    - pair_value[..., 1, 1, :, :]
                )
            elif l1 == 0:
                # Coupling a scalar to l has one path, 0 x l -> l, and its CG
                # matrix is the identity (up to normalization already applied
                # above).  Broadcasting avoids a disproportionately expensive
                # tiny einsum for the three scalar-input groups at lmax=2.
                diagonal = cg[0].diagonal()
                if l2 == 0:
                    real_value = (left * right).sum(dim=-3) * diagonal
                    pair_value = None
                else:
                    pair_value = (
                        left[..., 0].unsqueeze(-2).unsqueeze(-1)
                        * right.unsqueeze(-4)
                        * diagonal
                    )
                    real_value = (
                        pair_value[..., 0, 0, :, :]
                        + pair_value[..., 1, 1, :, :]
                    )
            elif l1 == l2:
                # Sum over the shared doublet index: xx + yy.
                real_value = torch.einsum(
                    "...arm,...arn,mnk->...rk", left, right, cg
                )
                pair_value = None
            else:
                # Compute xx, xy, yx, yy together.  Their Hermitian real and
                # imaginary combinations are selected below without new CG
                # contractions.
                pair_value = torch.einsum(
                    "...arm,...brn,mnk->...abrk", left, right, cg
                )
                real_value = pair_value[..., 0, 0, :, :] + pair_value[..., 1, 1, :, :]
            for _, cg_name, start, end in slices:
                coupled_values[(cg_name, "real")] = real_value[..., start:end]
                if pair_value is not None:
                    if self.quadratic_form == "charge2":
                        coupled_values[(cg_name, "imag")] = (
                            pair_value[..., 0, 1, :, start:end]
                            + pair_value[..., 1, 0, :, start:end]
                        )
                    else:
                        coupled_values[(cg_name, "imag")] = (
                            pair_value[..., 1, 0, :, start:end]
                            - pair_value[..., 0, 1, :, start:end]
                        )
        outputs: Dict[int, List[torch.Tensor]] = {
            l: [] for l in range(self.lmax + 1)
        }
        for l1, l2, out_l, component, cg_name in self.paths:
            outputs[out_l].append(coupled_values[(cg_name, component)])
        return {l: torch.cat(outputs[l], dim=-2) for l in range(self.lmax + 1)}

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        *,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
        gate_features: torch.Tensor | None = None,
        species_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        density_blocks = self.hermitian_blocks(real, imag)
        return self._mix_density_blocks(
            density_blocks,
            reference=real,
            node_attrs=node_attrs,
            node_type_idx=node_type_idx,
            gate_features=gate_features,
            species_embedding=species_embedding,
        )

    def _mix_output_block(
        self,
        density: torch.Tensor,
        out_l: int,
        *,
        type_idx: torch.Tensor | None,
        node_attrs: torch.Tensor | None,
        species_gates: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the configured species-conditioned map without dense node weights."""
        if self.species_mode == "onehot-full":
            if self.output_weights is None:
                raise RuntimeError("onehot-full Hermitian output weights are missing")
            weight = self.output_weights[str(out_l)].to(density)
            if type_idx is not None:
                mixed_weight = weight.index_select(0, type_idx)
            else:
                if node_attrs is None:
                    raise ValueError(
                        "node_attrs is required when node_type_idx is not provided"
                    )
                mixed_weight = torch.einsum(
                    "ne,eoi->noi", node_attrs.to(density), weight
                )
            return torch.bmm(mixed_weight, density)

        if species_gates is None:
            raise ValueError(
                "embedded-lowrank Hermitian writeback requires species gates"
            )
        if (
            self.shared_output_weights is None
            or self.species_left is None
            or self.species_right is None
        ):
            raise RuntimeError("embedded-lowrank Hermitian parameters are missing")
        shared = torch.einsum(
            "oi,nim->nom",
            self.shared_output_weights[str(out_l)].to(density),
            density,
        )
        low_rank_density = torch.einsum(
            "si,nim->nsm",
            self.species_right[str(out_l)].to(density),
            density,
        )
        residual = torch.einsum(
            "os,ns,nsm->nom",
            self.species_left[str(out_l)].to(density),
            species_gates.to(density),
            low_rank_density,
        )
        return shared + residual

    def _species_gates(
        self,
        reference: torch.Tensor,
        species_embedding: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.species_mode == "onehot-full":
            return None
        if species_embedding is None:
            raise ValueError(
                "embedded-lowrank Hermitian writeback requires species_embedding"
            )
        if species_embedding.ndim != 2 or species_embedding.shape != (
            reference.shape[0],
            self.species_embedding_dim,
        ):
            raise ValueError(
                "species_embedding must have shape "
                f"({reference.shape[0]}, {self.species_embedding_dim}), got "
                f"{tuple(species_embedding.shape)}"
            )
        if self.species_gate is None:
            raise RuntimeError("embedded-lowrank Hermitian gate is missing")
        return torch.tanh(self.species_gate(species_embedding.to(reference)))

    def forward_diagonal_edges_doublet(
        self,
        edge_doublet: torch.Tensor,
        *,
        edge_dst: torch.Tensor,
        num_nodes: int,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
        species_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build only j=k Hermitian terms, then aggregate the neutral edge densities."""
        edge_density_blocks = self.hermitian_blocks_doublet(edge_doublet)
        density_blocks = {
            out_l: scatter(
                block,
                edge_dst,
                dim=0,
                dim_size=int(num_nodes),
                reduce="sum",
            )
            for out_l, block in edge_density_blocks.items()
        }
        out_blocks: Dict[int, torch.Tensor] = {}
        if node_type_idx is not None:
            type_idx = node_type_idx.to(device=edge_doublet.device, dtype=torch.long)
        else:
            type_idx = None
            if node_attrs is None:
                raise ValueError("node_attrs is required when node_type_idx is not provided")
        species_gates = self._species_gates(edge_doublet, species_embedding)
        for out_l in range(self.lmax + 1):
            out = self._mix_output_block(
                density_blocks[out_l],
                out_l,
                type_idx=type_idx,
                node_attrs=node_attrs,
                species_gates=species_gates,
            )
            scale = self.residual_scale[out_l].to(dtype=out.dtype, device=out.device)
            out_blocks[out_l] = scale * out
        return _merge_irreps(out_blocks, self.channels, self.lmax)

    def _diagonal_blocks_factorized(
        self,
        edge_orbital: torch.Tensor,
        edge_norm_sq: torch.Tensor,
        *,
        edge_dst: torch.Tensor,
        num_nodes: int,
    ) -> Dict[int, torch.Tensor]:
        """Form edgewise self-density without materializing a charged doublet.

        The edge coefficient is one scalar doublet shared by all channels and
        irreps, so an edge charged orbital has the factorized form

            z_e = (c_{e,x}, c_{e,y}) b_e .

        Its Hermitian self-density is therefore

            z_e z_e^dagger = (c_{e,x}^2 + c_{e,y}^2) b_e b_e^T,

        and every same-edge imaginary block vanishes exactly.  Coupling each
        real orbital pair and scattering each small rank-R result directly to
        nodes avoids both the 2x edge doublet and the concatenated edge-density
        tensor used by ``forward_diagonal_edges_doublet``.
        """
        if edge_orbital.ndim != 2:
            raise ValueError(
                "edge_orbital must have shape (num_edges, irreps_dim), got "
                f"{tuple(edge_orbital.shape)}"
            )
        edge_weight = edge_norm_sq.reshape(-1).to(
            dtype=edge_orbital.dtype, device=edge_orbital.device
        )
        if edge_weight.shape[0] != edge_orbital.shape[0]:
            raise ValueError(
                "edge_norm_sq and edge_orbital disagree on num_edges: "
                f"{edge_weight.shape[0]} vs {edge_orbital.shape[0]}"
            )

        orbital_in = _split_irreps(edge_orbital, self.channels, self.lmax)
        rank_orbital = {
            l: self._project_block(
                orbital_in[l],
                self.rank_projections[str(l)],
            )
            for l in range(self.lmax + 1)
        }
        coupled_nodes: Dict[tuple[str, str], torch.Tensor] = {}
        for l1, l2, fused_name, slices in self.coupling_groups:
            left = rank_orbital[l1]
            right = rank_orbital[l2]
            cg = getattr(self, fused_name).to(dtype=left.dtype, device=left.device)
            if l1 == 0:
                diagonal = cg[0].diagonal()
                if l2 == 0:
                    real_value = (left * right) * diagonal
                else:
                    real_value = left[..., 0].unsqueeze(-1) * right * diagonal
            else:
                real_value = torch.einsum(
                    "...rm,...rn,mnk->...rk", left, right, cg
                )
            real_value = real_value * edge_weight[:, None, None]
            for _, cg_name, start, end in slices:
                node_value = scatter(
                    real_value[..., start:end],
                    edge_dst,
                    dim=0,
                    dim_size=int(num_nodes),
                    reduce="sum",
                )
                coupled_nodes[(cg_name, "real")] = node_value
                if l1 < l2:
                    coupled_nodes[(cg_name, "imag")] = torch.zeros_like(node_value)

        outputs: Dict[int, List[torch.Tensor]] = {
            l: [] for l in range(self.lmax + 1)
        }
        for _, _, out_l, component, cg_name in self.paths:
            outputs[out_l].append(coupled_nodes[(cg_name, component)])
        return {
            out_l: torch.cat(outputs[out_l], dim=-2)
            for out_l in range(self.lmax + 1)
        }

    def _mix_density_blocks(
        self,
        density_blocks: Dict[int, torch.Tensor],
        *,
        reference: torch.Tensor,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None,
        gate_features: torch.Tensor | None = None,
        species_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_blocks: Dict[int, torch.Tensor] = {}
        output_gate = None
        if self.output_gate_context is not None:
            if gate_features is None:
                raise ValueError(
                    "nonlinear Hermitian output gating requires invariant "
                    "gate_features"
                )
            if gate_features.ndim != 2 or gate_features.shape[-1] != self.channels:
                raise ValueError(
                    "gate_features must have shape (num_nodes, channels), got "
                    f"{tuple(gate_features.shape)}"
                )
            gate_logits = self.output_gate_context(gate_features)
            output_gate = 2.0 * torch.sigmoid(
                gate_logits.reshape(
                    gate_logits.shape[0],
                    self.lmax + 1,
                    self.channels,
                )
            )
        if node_type_idx is not None:
            type_idx = node_type_idx.to(device=reference.device, dtype=torch.long)
        else:
            type_idx = None
            if node_attrs is None:
                raise ValueError(
                    "node_attrs is required when node_type_idx is not provided"
                )
        species_gates = self._species_gates(reference, species_embedding)
        for out_l in range(self.lmax + 1):
            out = self._mix_output_block(
                density_blocks[out_l],
                out_l,
                type_idx=type_idx,
                node_attrs=node_attrs,
                species_gates=species_gates,
            )
            if output_gate is not None:
                out = out * output_gate[:, out_l, :, None].to(
                    dtype=out.dtype, device=out.device
                )
            scale = self.residual_scale[out_l].to(
                dtype=out.dtype, device=out.device
            )
            out_blocks[out_l] = scale * out
        return _merge_irreps(out_blocks, self.channels, self.lmax)

    def forward_diagonal_edges_factorized(
        self,
        edge_orbital: torch.Tensor,
        edge_norm_sq: torch.Tensor,
        *,
        edge_dst: torch.Tensor,
        num_nodes: int,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
        species_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the exact j=k density from a real orbital and coefficient norm."""
        density_blocks = self._diagonal_blocks_factorized(
            edge_orbital,
            edge_norm_sq,
            edge_dst=edge_dst,
            num_nodes=num_nodes,
        )
        return self._mix_density_blocks(
            density_blocks,
            reference=edge_orbital,
            node_attrs=node_attrs,
            node_type_idx=node_type_idx,
            species_embedding=species_embedding,
        )

    def forward_coherence_gated_factorized(
        self,
        doublet: torch.Tensor,
        edge_orbital: torch.Tensor,
        edge_norm_sq: torch.Tensor,
        *,
        gate_features: torch.Tensor | None = None,
        edge_dst: torch.Tensor,
        num_nodes: int,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
        species_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mix D and j!=k coherence using an exact factorized diagonal path."""
        if (
            self.coherence_scale is None
            and self.coherence_logit is None
            and self.coherence_context is None
        ):
            raise RuntimeError(
                "forward_coherence_gated_factorized requires coherence_gate=True"
            )
        coherence_scale = self.effective_coherence_scale(gate_features)
        full_blocks = self.hermitian_blocks_doublet(doublet)
        diagonal_blocks = self._diagonal_blocks_factorized(
            edge_orbital,
            edge_norm_sq,
            edge_dst=edge_dst,
            num_nodes=num_nodes,
        )
        density_blocks = {
            out_l: diagonal_blocks[out_l]
            + self._coherence_for_block(
                coherence_scale,
                out_l,
                full_blocks[out_l],
            )
            * (full_blocks[out_l] - diagonal_blocks[out_l])
            for out_l in range(self.lmax + 1)
        }
        return self._mix_density_blocks(
            density_blocks,
            reference=doublet,
            node_attrs=node_attrs,
            node_type_idx=node_type_idx,
            species_embedding=species_embedding,
        )

    def forward_pair_count_balanced_factorized(
        self,
        doublet: torch.Tensor,
        edge_orbital: torch.Tensor,
        edge_norm_sq: torch.Tensor,
        effective_coordination: torch.Tensor,
        *,
        reference_coordination: float,
        edge_dst: torch.Tensor,
        num_nodes: int,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
        species_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Balance diagonal and coherent terms by their local pair counts.

        For an effective coordination ``n_i``, diagonal self terms scale as
        O(n_i), while ordered off-diagonal pairs scale as O(n_i(n_i-1)).
        We preserve the original operator at ``n_i == reference_coordination``
        and remove only local coordination drift:

            D_i <- (n_ref / n_i) D_i
            C_i <- ((n_ref(n_ref-1)+1) / (n_i(n_i-1)+1)) C_i.

        The +1 regularizes one-neighbor environments without a discontinuous
        branch. ``effective_coordination`` is formed from the smooth cutoff
        envelope, so its position derivatives preserve continuous forces.
        """
        full_blocks = self.hermitian_blocks_doublet(doublet)
        diagonal_blocks = self._diagonal_blocks_factorized(
            edge_orbital,
            edge_norm_sq,
            edge_dst=edge_dst,
            num_nodes=num_nodes,
        )
        coordination = effective_coordination.reshape(-1).to(
            dtype=doublet.dtype, device=doublet.device
        ).clamp_min(1.0)
        if coordination.shape[0] != int(num_nodes):
            raise ValueError(
                "effective_coordination and num_nodes disagree: "
                f"{coordination.shape[0]} vs {int(num_nodes)}"
            )
        reference = max(float(reference_coordination), 1.0)
        diagonal_scale = reference / coordination
        reference_pairs = reference * (reference - 1.0)
        local_pairs = coordination * (coordination - 1.0)
        coherent_scale = (reference_pairs + 1.0) / (local_pairs + 1.0)
        density_blocks = {
            out_l: (
                diagonal_scale[:, None, None] * diagonal_blocks[out_l]
                + coherent_scale[:, None, None]
                * (full_blocks[out_l] - diagonal_blocks[out_l])
            )
            for out_l in range(self.lmax + 1)
        }
        return self._mix_density_blocks(
            density_blocks,
            reference=doublet,
            node_attrs=node_attrs,
            node_type_idx=node_type_idx,
            species_embedding=species_embedding,
        )

    def forward_coherence_gated_doublet(
        self,
        doublet: torch.Tensor,
        edge_doublet: torch.Tensor,
        *,
        gate_features: torch.Tensor | None = None,
        edge_dst: torch.Tensor,
        num_nodes: int,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
        species_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mix diagonal and coherent density as D + gamma_L (rho_full - D)."""
        if (
            self.coherence_scale is None
            and self.coherence_logit is None
            and self.coherence_context is None
        ):
            raise RuntimeError(
                "forward_coherence_gated_doublet requires coherence_gate=True"
            )
        coherence_scale = self.effective_coherence_scale(gate_features)
        full_blocks = self.hermitian_blocks_doublet(doublet)
        edge_density_blocks = self.hermitian_blocks_doublet(edge_doublet)
        diagonal_blocks = {
            out_l: scatter(
                block,
                edge_dst,
                dim=0,
                dim_size=int(num_nodes),
                reduce="sum",
            )
            for out_l, block in edge_density_blocks.items()
        }
        density_blocks = {
            out_l: diagonal_blocks[out_l]
            + self._coherence_for_block(
                coherence_scale,
                out_l,
                full_blocks[out_l],
            )
            * (full_blocks[out_l] - diagonal_blocks[out_l])
            for out_l in range(self.lmax + 1)
        }
        out_blocks: Dict[int, torch.Tensor] = {}
        if node_type_idx is not None:
            type_idx = node_type_idx.to(device=doublet.device, dtype=torch.long)
        else:
            type_idx = None
            if node_attrs is None:
                raise ValueError(
                    "node_attrs is required when node_type_idx is not provided"
                )
        species_gates = self._species_gates(doublet, species_embedding)
        for out_l in range(self.lmax + 1):
            out = self._mix_output_block(
                density_blocks[out_l],
                out_l,
                type_idx=type_idx,
                node_attrs=node_attrs,
                species_gates=species_gates,
            )
            scale = self.residual_scale[out_l].to(
                dtype=out.dtype, device=out.device
            )
            out_blocks[out_l] = scale * out
        return _merge_irreps(out_blocks, self.channels, self.lmax)

    def forward_doublet(
        self,
        doublet: torch.Tensor,
        *,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
        gate_features: torch.Tensor | None = None,
        species_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        density_blocks = self.hermitian_blocks_doublet(doublet)
        return self._mix_density_blocks(
            density_blocks,
            reference=doublet,
            node_attrs=node_attrs,
            node_type_idx=node_type_idx,
            gate_features=gate_features,
            species_embedding=species_embedding,
        )


class PerLScaleSO3(nn.Module):
    def __init__(self, channels: int, lmax: int, init_scales: list[float] | tuple[float, ...]):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        if len(init_scales) != self.lmax + 1:
            raise ValueError(f"Expected {self.lmax + 1} init scales, got {len(init_scales)}")
        scales = torch.as_tensor(init_scales, dtype=torch.get_default_dtype()).clamp_min(1e-6)
        self.log_scale = nn.Parameter(scales.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = _split_irreps(x, self.channels, self.lmax)
        out_blocks: Dict[int, torch.Tensor] = {}
        scales = self.log_scale.to(dtype=x.dtype, device=x.device).exp()
        for l in range(self.lmax + 1):
            out_blocks[l] = blocks[l] * scales[l]
        return _merge_irreps(out_blocks, self.channels, self.lmax)


class _Normalize2MomSiLU(nn.Module):
    _scale = 1.6791767923989418

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) * self._scale


def _init_mace_style_linear_(linear: nn.Linear) -> None:
    with torch.no_grad():
        nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(float(linear.in_features)))
        if linear.bias is not None:
            linear.bias.zero_()


class PathPreservingLinearSO3(nn.Module):
    def __init__(self, in_channels_by_l: Dict[int, int], out_channels: int, lmax: int):
        super().__init__()
        self.in_channels_by_l = {int(k): int(v) for k, v in in_channels_by_l.items()}
        self.out_channels = int(out_channels)
        self.lmax = int(lmax)
        self.weights = nn.ParameterDict()
        for l in range(self.lmax + 1):
            in_channels = self.in_channels_by_l.get(l, 0)
            weight = nn.Parameter(torch.empty(self.out_channels, in_channels))
            if in_channels > 0:
                nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(float(in_channels)))
            else:
                nn.init.zeros_(weight)
            self.weights[str(l)] = weight

    def forward(self, blocks: Dict[int, torch.Tensor]) -> torch.Tensor:
        out_blocks: Dict[int, torch.Tensor] = {}
        sample = next(iter(blocks.values()))
        for l in range(self.lmax + 1):
            x_l = blocks[l]
            weight = self.weights[str(l)].to(dtype=x_l.dtype, device=x_l.device)
            if x_l.shape[-2] == 0:
                out_blocks[l] = torch.zeros(
                    *x_l.shape[:-2], self.out_channels, 2 * l + 1, dtype=x_l.dtype, device=x_l.device
                )
            else:
                out_blocks[l] = torch.einsum("oc,...cm->...om", weight, x_l)
        return _merge_irreps(out_blocks, self.out_channels, self.lmax)


class ICTDSymmetricContractionSO3(nn.Module):
    """
    MACE-style symmetric contraction implemented with ICTC-SO3 operators.

    This keeps the higher-order ICTC-SO3 paths explicit as a basis list and then
    combines those basis terms with compact element-conditioned coefficients.
    That is closer in spirit to MACE's explicit product basis than the previous
    shared-contraction-plus-output-gating implementation.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        self.num_elements = int(num_elements)
        if contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(f"contraction_combine must be 'softmax', 'free', or 'path-free', got {contraction_combine!r}")
        self.contraction_combine = str(contraction_combine)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.reduce)
        self.order_mix = nn.ModuleList(
            [EquivariantChannelLinearSO3(self.hidden_channels, self.lmax, bias=False) for _ in range(self.correlation)]
        )
        _init_so3_linear_identity_(self.order_mix[0])
        self.tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        for tp in self.tp_layers:
            _init_path_tp_weight_to_one_(tp)
        self.tp_path_weight = nn.ParameterList()
        if self.contraction_combine == "path-free":
            for tp in self.tp_layers:
                weight = nn.Parameter(torch.empty(self.num_elements, tp.num_paths, self.hidden_channels))
                _init_contraction_path_weight_(weight)
                self.tp_path_weight.append(weight)
        self.out_linear = EquivariantChannelLinearSO3(
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.out_linear)
        if self.contraction_combine == "softmax":
            self.basis_logits = nn.ParameterDict(
                {
                    str(l): nn.Parameter(
                        torch.zeros(self.num_elements, self.correlation, self.hidden_channels)
                    )
                    for l in range(self.lmax + 1)
                }
            )
            self.basis_weight = None
            for logits in self.basis_logits.values():
                _init_contraction_basis_logits_(logits)
        else:
            self.basis_logits = None
            self.basis_weight = nn.ParameterDict(
                {
                    str(l): nn.Parameter(
                        torch.empty(self.num_elements, self.correlation, self.hidden_channels)
                    )
                    for l in range(self.lmax + 1)
                }
            )
            for weight in self.basis_weight.values():
                _init_contraction_basis_weight_(weight)

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        element_index = _node_type_indices(node_attrs)

        basis_terms = [self.order_mix[0](base)]
        if self.correlation > 1:
            base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
            current_blocks = base_blocks
            for order_idx, tp in enumerate(self.tp_layers, start=1):
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.tp_path_weight[order_idx - 1][element_index].to(dtype=base.dtype)
                current_blocks = tp(current_blocks, base_blocks, path_channel_weights=path_weight)
                current_flat = _merge_irreps(current_blocks, self.hidden_channels, self.lmax)
                basis_terms.append(self.order_mix[order_idx](current_flat))

        basis_blocks = [_split_irreps(term, self.hidden_channels, self.lmax) for term in basis_terms]
        combined_blocks: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            if self.contraction_combine == "softmax":
                coeff = torch.softmax(self.basis_logits[str(l)][element_index].to(dtype=base.dtype), dim=1)
            else:
                coeff = self.basis_weight[str(l)][element_index].to(dtype=base.dtype)
            stack = torch.stack([term_blocks[l] for term_blocks in basis_blocks], dim=1)
            combined_blocks[l] = torch.sum(stack * coeff.unsqueeze(-1), dim=1)
        combined = _merge_irreps(combined_blocks, self.hidden_channels, self.lmax)
        return self.out_linear(combined)


class ICTDProductBasisBlock(nn.Module):
    """
    MACE-style product block:
      h_{t+1} = linear( symmetric_contraction_ictd(message, node_attrs) ) + sc
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.symmetric_contractions = ICTDSymmetricContractionSO3(
            num_elements=num_elements,
            in_channels=channels,
            hidden_channels=channels,
            lmax=lmax,
            correlation=correlation,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
            contraction_combine=contraction_combine,
        )
        self.linear = EquivariantChannelLinearSO3(channels, lmax, bias=False)
        _init_so3_linear_identity_(self.linear)
        self.output_norm = nn.Identity()

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        contracted = self.symmetric_contractions(node_feats, node_attrs)
        out = self.linear(contracted)
        if sc is not None:
            out = out + sc
        return self.output_norm(out)


class ICTDScalarSymmetricContractionSO3(nn.Module):
    """
    Scalar-target MACE-style contraction.

    This mirrors MACE's final product target `64x0e`: the last TP step is
    instruction-pruned to l_out=0, and all order mixing/output projection happens
    only in scalar channel space.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        self.num_elements = int(num_elements)
        if contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(f"contraction_combine must be 'softmax', 'free', or 'path-free', got {contraction_combine!r}")
        self.contraction_combine = str(contraction_combine)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        _init_so3_linear_identity_(self.reduce)
        self.scalar_order_mix = nn.ModuleList(
            [nn.Linear(self.hidden_channels, self.hidden_channels, bias=False) for _ in range(self.correlation)]
        )
        _init_linear_identity_(self.scalar_order_mix[0])
        self.full_tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 2, 0))
            ]
        )
        scalar_paths = _tp_allowed_paths_to_output_l(self.lmax, self.lmax, 0)
        self.final_scalar_tp = (
            HarmonicPathWeightedTensorProduct(
                channels=self.hidden_channels,
                lmax=self.lmax,
                allowed_paths=scalar_paths,
                path_policy=ictd_tp_path_policy,
                max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=internal_compute_dtype,
            )
            if self.correlation > 1
            else None
        )
        for tp in self.full_tp_layers:
            _init_path_tp_weight_to_one_(tp)
        _init_path_tp_weight_to_one_(self.final_scalar_tp)
        self.full_tp_path_weight = nn.ParameterList()
        self.final_scalar_path_weight = None
        if self.contraction_combine == "path-free":
            for tp in self.full_tp_layers:
                weight = nn.Parameter(torch.empty(self.num_elements, tp.num_paths, self.hidden_channels))
                _init_contraction_path_weight_(weight)
                self.full_tp_path_weight.append(weight)
            if self.final_scalar_tp is not None:
                self.final_scalar_path_weight = nn.Parameter(
                    torch.empty(self.num_elements, self.final_scalar_tp.num_paths, self.hidden_channels)
                )
                _init_contraction_path_weight_(self.final_scalar_path_weight)
        self.out_linear = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        _init_linear_identity_(self.out_linear)
        if self.contraction_combine == "softmax":
            self.basis_logits = nn.Parameter(
                torch.zeros(self.num_elements, self.correlation, self.hidden_channels)
            )
            self.basis_weight = None
            _init_contraction_basis_logits_(self.basis_logits)
        else:
            self.basis_logits = None
            self.basis_weight = nn.Parameter(
                torch.empty(self.num_elements, self.correlation, self.hidden_channels)
            )
            _init_contraction_basis_weight_(self.basis_weight)

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
        element_index = _node_type_indices(node_attrs)

        basis_terms = [self.scalar_order_mix[0](base_blocks[0].squeeze(-1))]
        current_blocks = base_blocks
        for order_idx in range(1, self.correlation):
            if order_idx == self.correlation - 1:
                if self.final_scalar_tp is None:
                    raise RuntimeError("final_scalar_tp unexpectedly missing")
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.final_scalar_path_weight[element_index].to(dtype=base.dtype)
                current_blocks = self.final_scalar_tp(current_blocks, base_blocks, path_channel_weights=path_weight)
            else:
                path_weight = None
                if self.contraction_combine == "path-free":
                    path_weight = self.full_tp_path_weight[order_idx - 1][element_index].to(dtype=base.dtype)
                current_blocks = self.full_tp_layers[order_idx - 1](
                    current_blocks,
                    base_blocks,
                    path_channel_weights=path_weight,
                )
            scalar = current_blocks[0].squeeze(-1)
            basis_terms.append(self.scalar_order_mix[order_idx](scalar))

        if self.contraction_combine == "softmax":
            coeff = torch.softmax(self.basis_logits[element_index].to(dtype=base.dtype), dim=1)
        else:
            coeff = self.basis_weight[element_index].to(dtype=base.dtype)
        stack = torch.stack(basis_terms, dim=1)
        combined = torch.sum(stack * coeff, dim=1)
        return self.out_linear(combined)


class ICTDScalarProductBasisBlock(nn.Module):
    """
    MACE-style final product block for keep_last_layer_irreps=False.

    Native MACE changes the last product target irreps to scalar-only. This
    block uses a scalar-target ICTC contraction instead of building the full
    output irreps and slicing l=0 afterward.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        contraction_combine: str = "softmax",
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.symmetric_contractions = ICTDScalarSymmetricContractionSO3(
            num_elements=num_elements,
            in_channels=channels,
            hidden_channels=channels,
            lmax=lmax,
            correlation=correlation,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
            contraction_combine=contraction_combine,
        )
        self.linear = nn.Linear(self.channels, self.channels, bias=False)
        _init_linear_identity_(self.linear)
        self.output_norm = nn.Identity()

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        contracted = self.symmetric_contractions(node_feats, node_attrs)
        out = self.linear(contracted)
        if sc is not None:
            if sc.shape[-1] == self.channels:
                sc_scalar = sc
            else:
                sc_scalar = _split_irreps(sc, self.channels, self.lmax)[0].squeeze(-1)
            out = out + sc_scalar
        return self.output_norm(out)


class SO3ToE3NNBasisBridge(nn.Module):
    """
    Fixed per-l orthogonal bridge between the ICTC SO3 basis and e3nn/MACE basis.

    `direction_harmonics_all` now matches e3nn component normalization in RMS, but
    each l block can still differ by an orthogonal basis convention. Native MACE
    contraction assumes the e3nn convention, while ICTC interaction emits ICTC
    convention features. The bridge uses a deterministic least-squares/SVD fit
    from sampled directions to construct Q_l such that:

        Y_ictd_l @ Q_l ~= Y_e3nn_l
    """

    def __init__(self, channels: int, lmax: int, num_samples: int = 8192):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260426)
        dirs = torch.randn(int(num_samples), 3, generator=generator, dtype=torch.float64)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y_ictd = direction_harmonics_all(dirs, self.lmax)
        y_e3nn = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(self.lmax),
            dirs,
            normalize=True,
            normalization="component",
        )
        offset = 0
        for l in range(self.lmax + 1):
            width = 2 * l + 1
            a = y_ictd[l].to(dtype=torch.float64)
            b = y_e3nn[:, offset : offset + width].to(dtype=torch.float64)
            offset += width
            u, _, vh = torch.linalg.svd(a.T @ b)
            q = (u @ vh).to(dtype=torch.get_default_dtype())
            self.register_buffer(f"q_{l}", q, persistent=True)

    def _q(self, l: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return getattr(self, f"q_{int(l)}").to(dtype=dtype, device=device)

    def ictd_flat_to_e3nn_blocks(self, x: torch.Tensor, lmax: int) -> Dict[int, torch.Tensor]:
        blocks = _split_irreps(x, self.channels, int(lmax))
        out: Dict[int, torch.Tensor] = {}
        for l, block in blocks.items():
            out[l] = torch.einsum("ncm,mp->ncp", block, self._q(l, dtype=x.dtype, device=x.device))
        return out

    def e3nn_flat_to_ictd_blocks(self, x: torch.Tensor, lmax: int) -> Dict[int, torch.Tensor]:
        blocks = _split_irreps(x, self.channels, int(lmax))
        out: Dict[int, torch.Tensor] = {}
        for l, block in blocks.items():
            out[l] = torch.einsum("ncm,pm->ncp", block, self._q(l, dtype=x.dtype, device=x.device))
        return out

    def ictd_flat_to_e3nn_features(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.ictd_flat_to_e3nn_blocks(x, int(lmax))
        return torch.cat([blocks[l] for l in range(int(lmax) + 1)], dim=-1)

    def ictd_flat_to_e3nn_flat(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.ictd_flat_to_e3nn_blocks(x, int(lmax))
        return _merge_blocks_subset(blocks, self.channels, int(lmax))

    def e3nn_flat_to_ictd_flat(self, x: torch.Tensor, lmax: int) -> torch.Tensor:
        blocks = self.e3nn_flat_to_ictd_blocks(x, int(lmax))
        return _merge_blocks_subset(blocks, self.channels, int(lmax))


class NativeMACEProductBasisBlockSO3(nn.Module):
    """
    Hybrid product block: ICTC-SO3 interaction features are interpreted in
    MACE/e3nn mul-ir layout, then contracted by the native MACE symmetric
    contraction implementation.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=self.use_reduced_cg,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, max(self.lmax, self.target_lmax))

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        x = self.basis_bridge.ictd_flat_to_e3nn_features(node_feats, self.lmax)
        out = self.linear(self.symmetric_contractions(x, node_attrs))
        if sc is not None:
            if sc.shape[-1] != self.channels:
                sc = self.basis_bridge.ictd_flat_to_e3nn_flat(sc, self.target_lmax)
            out = _add_product_self_connection(
                out,
                sc,
                channels=self.channels,
                target_lmax=self.target_lmax,
                input_lmax=self.lmax,
            )
        if self.target_lmax > 0:
            out = self.basis_bridge.e3nn_flat_to_ictd_flat(out, self.target_lmax)
        return out


def _cueq_o3_e3nn_group():
    try:
        from mace.tools.cg import O3_e3nn
    except Exception:
        try:
            from chorus.models._mace_cg import O3_e3nn
        except Exception as exc:  # pragma: no cover - optional accelerator dependency
            raise RuntimeError(
                "ictd_fix_product_backend='cueq' requires the e3nn-compatible "
                "cuequivariance O3 group from mace.tools.cg or chorus.models._mace_cg"
            ) from exc
    return O3_e3nn


def _cueq_o3_irreps(channels: int, lmax: int):
    """Build a cuequivariance O3 irreps string matching the local MACE/e3nn convention."""
    try:
        import cuequivariance as cue
    except Exception as exc:  # pragma: no cover - optional accelerator dependency
        raise RuntimeError(
            "ictd_fix_product_backend='cueq' requires cuequivariance and "
            "cuequivariance_torch to be installed"
        ) from exc
    terms = []
    for l in range(int(lmax) + 1):
        parity = "e" if l % 2 == 0 else "o"
        terms.append(f"{int(channels)}x{l}{parity}")
    return cue.Irreps(_cueq_o3_e3nn_group(), " + ".join(terms))


class CueqMaceSymmetricContractionSO3(nn.Module):
    """MACE symmetric contraction accelerated by cuEquivariance.

    The inner ``symmetric_contractions`` module keeps the exact MACE parameter
    layout used by the converter and by existing checkpoints. The cuEquivariance
    modules hold frozen mirror weights for eval/inference. CUDA training uses
    cuEquivariance's contraction graph with weights computed differentiably from
    the authoritative MACE parameters, so gradients still land on checkpoint-
    compatible tensors.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.correlation = int(correlation)
        self.use_reduced_cg = bool(use_reduced_cg)
        if self.correlation not in {1, 2, 3, 4}:
            raise NotImplementedError(
                "cuEquivariance product backend currently supports correlation=1,2,3,4"
            )

        try:
            import cuequivariance as cue
            import cuequivariance_torch as cuet
        except Exception as exc:  # pragma: no cover - optional accelerator dependency
            raise RuntimeError(
                "ictd_fix_product_backend='cueq' requires cuequivariance and "
                "cuequivariance_torch to be installed"
            ) from exc

        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=self.correlation,
            num_elements=int(num_elements),
            use_reduced_cg=self.use_reduced_cg,
        )
        self._cueq_weight_specs = [
            self._make_cueq_weight_spec(ref)
            for ref in self.symmetric_contractions.contractions
        ]

        irreps_in = _cueq_o3_irreps(self.channels, self.lmax)
        method = "uniform_1d" if torch.cuda.is_available() else "naive"
        self.cueq_contractions = nn.ModuleList()
        self._cueq_weight_projection_names: list[str | None] = []
        weight_projection_fn = None
        if self.use_reduced_cg:
            try:
                from chorus.models._cueq_cg_tools import symmetric_contraction_proj as weight_projection_fn
            except Exception as exc:  # pragma: no cover - optional conversion dependency
                raise RuntimeError(
                    "ictd_fix_product_backend='cueq' with reduced CG requires "
                    "chorus.models._cueq_cg_tools.symmetric_contraction_proj to convert "
                    "MACE/e3nn symmetric-contraction weights into cueq weight space"
                ) from exc
        for out_idx, (_mul, ir) in enumerate(self.target_irreps):
            out_l = int(ir.l)
            parity = "e" if out_l % 2 == 0 else "o"
            irreps_out = cue.Irreps(_cueq_o3_e3nn_group(), f"{self.channels}x{out_l}{parity}")
            projection_name = None
            if weight_projection_fn is not None:
                _, projection = weight_projection_fn(
                    irreps_in,
                    irreps_out,
                    tuple(range(1, self.correlation + 1)),
                )
                projection_name = f"_cueq_weight_projection_{out_idx}"
                self.register_buffer(
                    projection_name,
                    torch.tensor(projection, dtype=torch.get_default_dtype()),
                    persistent=False,
                )
            self._cueq_weight_projection_names.append(projection_name)
            self.cueq_contractions.append(
                cuet.SymmetricContraction(
                    irreps_in,
                    irreps_out,
                    contraction_degree=self.correlation,
                    num_elements=int(num_elements),
                    layout_in=cue.ir_mul,
                    layout_out=cue.mul_ir,
                    dtype=torch.get_default_dtype(),
                    math_dtype=torch.get_default_dtype(),
                    original_mace=not self.use_reduced_cg,
                    method=method,
                )
            )
            self.cueq_contractions[-1].weight.requires_grad_(False)
        self.refresh_cueq_weights()
        self.register_load_state_dict_post_hook(lambda module, _incompatible: module.refresh_cueq_weights())

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.refresh_cueq_weights()
        return self

    def _make_cueq_weight_spec(self, ref: nn.Module) -> list[int | str]:
        spec: list[int | str] = ["max"]
        spec.extend(range(len(ref.weights)))
        if self.use_reduced_cg:
            return spec
        active: list[int | str] = []
        for item in spec:
            flag_name = "weights_max_zeroed" if item == "max" else f"weights_{int(item)}_zeroed"
            zeroed = bool(getattr(ref, flag_name).detach().cpu().item()) if hasattr(ref, flag_name) else False
            if not zeroed:
                active.append(item)
        return active

    def _mace_weights_for_cueq(self, ref: nn.Module) -> torch.Tensor:
        """Pack local MACE symmetric-contraction weights in cuEq path order."""
        try:
            spec_idx = list(self.symmetric_contractions.contractions).index(ref)
            spec = self._cueq_weight_specs[spec_idx]
        except ValueError:
            spec = self._make_cueq_weight_spec(ref)
        parts = [
            ref.weights_max if item == "max" else ref.weights[int(item)]
            for item in spec
        ]
        if parts:
            return torch.cat(parts, dim=1)
        return ref.weights_max.new_zeros((ref.weights_max.shape[0], 0, ref.weights_max.shape[-1]))

    def refresh_cueq_weights(self) -> None:
        """Mirror MACE-contraction weights into cuEquivariance path-weight tensors."""
        with torch.no_grad():
            for ref, fast, projection_name in zip(
                self.symmetric_contractions.contractions,
                self.cueq_contractions,
                self._cueq_weight_projection_names,
            ):
                weights = self._mace_weights_for_cueq(ref)
                weights = weights.to(dtype=fast.weight.dtype, device=fast.weight.device)
                if projection_name is not None:
                    projection = getattr(self, projection_name).to(
                        dtype=fast.weight.dtype,
                        device=fast.weight.device,
                    )
                    weights = torch.einsum("zau,ab->zbu", weights, projection)
                if weights.shape == fast.weight.shape:
                    fast.weight.copy_(weights)
                    continue
                raise ValueError(
                    "Cannot mirror MACE contraction weights into cueq contraction: "
                    f"source={tuple(weights.shape)} target={tuple(fast.weight.shape)} "
                    f"projection={None if getattr(fast, 'projection', None) is None else tuple(fast.projection.shape)}"
                )

    def _cueq_forward_with_weight(
        self,
        fast: nn.Module,
        flat: torch.Tensor,
        idx: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        projection = getattr(fast, "projection", None)
        if projection is not None:
            projection = projection.to(dtype=weight.dtype, device=weight.device)
            weight = torch.einsum("zau,ab->zbu", weight, projection)
        output = fast.f([weight.flatten(1), fast.transpose_in(flat)], input_indices={0: idx})
        return fast.transpose_out(output[0])

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            if node_attrs is None:
                raise ValueError("node_attrs is required for the reference MACE contraction path")
            return self.symmetric_contractions(x, node_attrs)
        if node_type_idx is None:
            if node_attrs is None:
                raise ValueError("node_type_idx or node_attrs is required for cueq contraction")
            idx = _node_type_indices(node_attrs).to(device=x.device, dtype=torch.int32)
        else:
            idx = node_type_idx.to(device=x.device, dtype=torch.int32)
        flat = x.transpose(1, 2).reshape(x.shape[0], -1)
        if self.training:
            outs = []
            for ref, fast, projection_name in zip(
                self.symmetric_contractions.contractions,
                self.cueq_contractions,
                self._cueq_weight_projection_names,
            ):
                weights = self._mace_weights_for_cueq(ref).to(dtype=flat.dtype, device=flat.device)
                if projection_name is not None:
                    projection = getattr(self, projection_name).to(dtype=weights.dtype, device=weights.device)
                    weights = torch.einsum("zau,ab->zbu", weights, projection)
                outs.append(self._cueq_forward_with_weight(fast, flat, idx, weights))
        else:
            outs = [contraction(flat, idx) for contraction in self.cueq_contractions]
        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]


class CueqMACEProductBasisBlockSO3(nn.Module):
    """Native-MACE product block using cuEquivariance for the contraction."""

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self._e3nn_basis = False
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, max(self.lmax, self.target_lmax))
        self.symmetric_contractions = CueqMaceSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
            use_reduced_cg=self.use_reduced_cg,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def refresh_cueq_weights(self) -> None:
        self.symmetric_contractions.refresh_cueq_weights()

    def enable_e3nn_basis(self, q_blocks: "List[torch.Tensor] | None" = None) -> None:
        """Consume and return e3nn-basis features directly.

        cuEquivariance's MACE contraction already uses the e3nn/O3 convention, so
        when the rest of the model has folded its interaction CGs into that same
        basis we can skip the per-l ICTC<->e3nn bridge around the product block.
        """
        del q_blocks
        self._e3nn_basis = True

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor | None,
        node_attrs: torch.Tensor | None,
        node_type_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._e3nn_basis:
            x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        else:
            x = self.basis_bridge.ictd_flat_to_e3nn_features(node_feats, self.lmax)
        out = self.linear(self.symmetric_contractions(x, node_attrs, node_type_idx=node_type_idx))
        if sc is not None:
            if (not self._e3nn_basis) and sc.shape[-1] != self.channels:
                sc = self.basis_bridge.ictd_flat_to_e3nn_flat(sc, self.target_lmax)
            out = _add_product_self_connection(
                out,
                sc,
                channels=self.channels,
                target_lmax=self.target_lmax,
                input_lmax=self.lmax,
            )
        if self.target_lmax > 0 and not self._e3nn_basis:
            out = self.basis_bridge.e3nn_flat_to_ictd_flat(out, self.target_lmax)
        return out


class ICTDBridgeUSymmetricContractionSO3(nn.Module):
    """
    Bridge-U symmetric contraction expressed directly in the ICTC basis.

    This is algebraically equivalent to:
      ICTC features -> e3nn basis -> MACE SymmetricContraction -> ICTC basis
    but the per-l basis change is folded into the stored U tensors once at
    initialization. The forward path therefore consumes and returns ICTC-basis
    flat SO3 features. This backend is the stable high-l bridge used when
    pure ICTC U generation is not numerically reliable.
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.basis_bridge = SO3ToE3NNBasisBridge(self.channels, max(self.lmax, self.target_lmax))
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=self.use_reduced_cg,
        )
        self._fold_basis_change_into_u_tensors()

    def _input_q(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        blocks = [self.basis_bridge._q(l, dtype=dtype, device=device) for l in range(self.lmax + 1)]
        return torch.block_diag(*blocks)

    def _transform_u_tensor(self, u_tensor: torch.Tensor, output_l: int) -> torch.Tensor:
        dtype = u_tensor.dtype
        device = u_tensor.device
        q_in = self._input_q(dtype=dtype, device=device)
        nu = int(u_tensor.dim() - 1) if int(output_l) == 0 else int(u_tensor.dim() - 2)
        if int(output_l) == 0:
            if nu == 1:
                return torch.einsum("ai,ip->ap", q_in, u_tensor)
            if nu == 2:
                return torch.einsum("ai,bj,ijp->abp", q_in, q_in, u_tensor)
            if nu == 3:
                return torch.einsum("ai,bj,ck,ijkp->abcp", q_in, q_in, q_in, u_tensor)
        q_out = self.basis_bridge._q(output_l, dtype=dtype, device=device)
        if nu == 1:
            return torch.einsum("ro,ai,oip->rap", q_out, q_in, u_tensor)
        if nu == 2:
            return torch.einsum("ro,ai,bj,oijp->rabp", q_out, q_in, q_in, u_tensor)
        if nu == 3:
            return torch.einsum("ro,ai,bj,ck,oijkp->rabcp", q_out, q_in, q_in, q_in, u_tensor)
        raise NotImplementedError(f"ICTC bridge-U contraction currently supports correlation<=3, got nu={nu}")

    def _fold_basis_change_into_u_tensors(self) -> None:
        with torch.no_grad():
            for (mul, ir), contraction in zip(self.target_irreps, self.symmetric_contractions.contractions):
                del mul
                output_l = int(ir.l)
                for nu in range(1, int(contraction.correlation) + 1):
                    name = f"U_matrix_{nu}"
                    old = getattr(contraction, name)
                    new = self._transform_u_tensor(old, output_l)
                    old.copy_(new.to(dtype=old.dtype, device=old.device))
                if getattr(contraction, "_use_scalar_corr3_fast", False) and hasattr(
                    contraction, "refresh_scalar_corr3_fast_buffers"
                ):
                    contraction.refresh_scalar_corr3_fast_buffers()

    def forward(self, node_feats: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        return self.symmetric_contractions(x, node_attrs)


class ICTDBridgeUProductBasisBlockSO3(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        use_reduced_cg: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.use_reduced_cg = bool(use_reduced_cg)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = ICTDBridgeUSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
            use_reduced_cg=self.use_reduced_cg,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.symmetric_contractions(node_feats, node_attrs))
        if sc is not None:
            out = _add_product_self_connection(
                out,
                sc,
                channels=self.channels,
                target_lmax=self.target_lmax,
                input_lmax=self.lmax,
            )
        return out


# Backward-compatible aliases for checkpoints/scripts that still refer to the old name.
ICTDMACEUSymmetricContractionSO3 = ICTDBridgeUSymmetricContractionSO3
ICTDMACEUProductBasisBlockSO3 = ICTDBridgeUProductBasisBlockSO3


class _ICTDPureUContraction(nn.Module):
    """MACE contraction recursion over caller-provided ICTC U tensors."""

    def __init__(
        self,
        *,
        u_tensors: Dict[int, torch.Tensor],
        output_l: int,
        num_elements: int,
        num_features: int,
    ):
        super().__init__()
        self.output_l = int(output_l)
        self.correlation = int(max(u_tensors))
        self.num_elements = int(num_elements)
        self.num_features = int(num_features)
        for nu in range(1, self.correlation + 1):
            self.register_buffer(f"U_matrix_{nu}", u_tensors[nu].contiguous())

        self.contractions_weighting = nn.ModuleList()
        self.contractions_features = nn.ModuleList()
        self.weights = nn.ParameterList([])

        for i in range(self.correlation, 0, -1):
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * self.output_l + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == self.correlation:
                parse_subscript_main = (
                    [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum("".join(parse_subscript_main), x, y, w, z)
                )
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn([num_equivariance] + [num_ell] * i + [num_params]).squeeze(0),
                        torch.randn((self.num_elements, num_params, self.num_features)),
                        torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_elements)),
                    ),
                )
                self.weights_max = nn.Parameter(
                    torch.randn((self.num_elements, num_params, self.num_features)) / max(num_params, 1)
                )
            else:
                parse_subscript_weighting = (
                    [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1))]
                    + ["k,ekc,be->bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i + min(self.output_l, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i - 1 + min(self.output_l, 1))]
                    + ["i,bci->bc"]
                    + [_CONTRACTION_ALPHABET[j] for j in range(i - 1 + min(self.output_l, 1))]
                )

                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum("".join(parse_subscript_weighting), x, y, z)
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )
                self.contractions_weighting.append(
                    opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_weighting,
                        example_inputs=(
                            torch.randn([num_equivariance] + [num_ell] * i + [num_params]).squeeze(0),
                            torch.randn((self.num_elements, num_params, self.num_features)),
                            torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_elements)),
                        ),
                    )
                )
                self.contractions_features.append(
                    opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_features,
                        example_inputs=(
                            torch.randn(
                                [_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_equivariance]
                                + [num_ell] * i
                            ).squeeze(2),
                            torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_ell)),
                        ),
                    )
                )
                self.weights.append(
                    nn.Parameter(torch.randn((self.num_elements, num_params, self.num_features)) / max(num_params, 1))
                )

    def U_tensors(self, nu: int) -> torch.Tensor:
        return dict(self.named_buffers())[f"U_matrix_{int(nu)}"]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.graph_opt_main(self.U_tensors(self.correlation), self.weights_max, x, y)
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(self.U_tensors(self.correlation - i - 1), weight, y)
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)
        return out.view(out.shape[0], -1)


class ICTDPureUSymmetricContractionSO3(nn.Module):
    """
    MACE-style symmetric contraction with U tensors generated from ICTC CG only.

    This keeps the optimized MACE contraction/einsum wrapper and trainable
    per-element weights, but replaces every `U_matrix_real` buffer with the
    corresponding ICTC-basis U generated by `ictd_u_matrix_so3`. It is the
    pure-ICTC contraction ablation against `ictd-bridge-u`.

    NATIVE-TRAINING ONLY -- does NOT correspond to an e3nn/MACE checkpoint. The
    ICTC U is obtained by solving CG null-spaces via SVD, whose basis is not unique;
    the resulting path gauge spans the correct intertwiner space but does not align
    with e3nn's Wigner-3j paths. A trained e3nn checkpoint's contraction weights
    therefore have no image here, so this backend cannot reproduce a pretrained
    spherical model -- use `ictd-bridge-u` (which reuses e3nn's own U) for exact
    checkpoint utilization. (Empirically edge_lmax==lmax or random weights look
    close, ~1e-5 in fp64, but real pretrained weights diverge by tens of percent on
    a converted MACE-OFF23: that small residual is a benign-regime artifact, not a
    fidelity floor.)
    """

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.contractions = nn.ModuleList()
        dtype = torch.get_default_dtype()
        for mul, ir in self.target_irreps:
            del mul
            output_l = int(ir.l)
            u_tensors = {
                nu: ictd_u_matrix_so3(
                    lmax=self.lmax,
                    output_l=output_l,
                    correlation=nu,
                    irrep_normalization="component",
                    dtype=dtype,
                )
                for nu in range(1, int(correlation) + 1)
            }
            self.contractions.append(
                _ICTDPureUContraction(
                    u_tensors=u_tensors,
                    output_l=output_l,
                    num_elements=int(num_elements),
                    num_features=self.channels,
                )
            )

        # angular_basis='e3nn': wrap this (order-nu) contraction so it RUNS in the ICTC basis
        # (U tensors stay ICTC) yet consumes/returns e3nn-basis features. Folding Q into the
        # order-3 U is mathematically identical but introduces input-dependent float64 cancellation
        # in the rotated basis (~1e-6 on the l>=1 features); the wrap keeps the contraction output
        # equal to (ictd output) @ Q to MACHINE PRECISION. The interaction TP (order 2) folds
        # exactly, so only the contraction is wrapped.
        self._e3nn_basis = False
        self._e3nn_bridge: SO3ToE3NNBasisBridge | None = None

    def enable_e3nn_basis(self, q_blocks: "List[torch.Tensor] | None" = None) -> None:
        """Make this contraction consume + return e3nn-basis features: rotate the input
        e3nn->ICTC, run the numerically-stable ICTC contraction, rotate the output ICTC->e3nn."""
        del q_blocks  # the bridge rebuilds the identical Q (same deterministic Procrustes fit)
        self._e3nn_basis = True
        if self._e3nn_bridge is None:
            self._e3nn_bridge = SO3ToE3NNBasisBridge(self.channels, self.lmax)

    def forward(self, node_feats: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        if self._e3nn_basis:
            node_feats = self._e3nn_bridge.e3nn_flat_to_ictd_flat(node_feats, self.lmax)
        x = _so3_flat_to_mace_features(node_feats, self.channels, self.lmax)
        out = torch.cat([contraction(x, node_attrs) for contraction in self.contractions], dim=-1)
        if self._e3nn_basis and self.target_lmax > 0:
            out = self._e3nn_bridge.ictd_flat_to_e3nn_flat(out, self.target_lmax)
        return out


class ICTDPureUProductBasisBlockSO3(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.target_irreps = _hidden_irreps(self.channels, self.target_lmax)
        self.symmetric_contractions = ICTDPureUSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.symmetric_contractions(node_feats, node_attrs))
        if sc is not None:
            out = _add_product_self_connection(
                out,
                sc,
                channels=self.channels,
                target_lmax=self.target_lmax,
                input_lmax=self.lmax,
            )
        return out


class MACEStyleScalarReadoutSO3(nn.Module):
    """Native MACE final readout shape: Cx0e -> MLP_irreps scalar width -> 1x0e."""

    def __init__(self, channels: int, hidden_channels: int = 16, output_init_std: float = 0.003):
        """output_init_std: small value (0.003) so initial energy ≈ 0 (MLIP standard practice)."""
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.linear_1 = nn.Linear(self.channels, self.hidden_channels, bias=False)
        self.activation = _Normalize2MomSiLU()
        self.linear_2 = nn.Linear(self.hidden_channels, 1, bias=False)
        _init_mace_style_linear_(self.linear_1)
        _init_mace_style_linear_(self.linear_2)

    def forward(self, scalar_feats: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(scalar_feats)))


class InvariantScalarResidualFFN(nn.Module):
    """Cheap transformer-style FFN acting only on the invariant l=0 block."""

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        self.channels = int(channels)
        hidden_channels = int(expansion) * self.channels
        self.norm = nn.LayerNorm(
            self.channels,
            elementwise_affine=False,
        )
        self.linear_1 = nn.Linear(self.channels, hidden_channels)
        self.activation = _Normalize2MomSiLU()
        self.linear_2 = nn.Linear(hidden_channels, self.channels)
        _init_mace_style_linear_(self.linear_1)
        # Exact identity initialization keeps the original MACE/CHORUS path
        # intact; the output projection receives gradients immediately.
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        if node_feats.shape[-1] < self.channels:
            raise ValueError(
                "scalar FFN requires an l=0 block with at least "
                f"{self.channels} channels, got {tuple(node_feats.shape)}"
            )
        scalar = node_feats[..., : self.channels]
        delta = self.linear_2(
            self.activation(self.linear_1(self.norm(scalar)))
        )
        return torch.cat(
            (scalar + delta, node_feats[..., self.channels :]),
            dim=-1,
        )


class ICTDResidualInteractionBlock(nn.Module):
    """
    ICTC-SO3 interaction block with MACE-like interface.

    Returns:
      - message: scatter-aggregated neighbor message
      - sc:      element-conditioned self-connection
    """

    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        input_lmax: int | None = None,
        target_lmax: int | None = None,
        sc_lmax: int | None = None,
        number_of_basis: int,
        num_elements: int,
        function_type: str = "gaussian",
        ictd_save_tp_mode: str = "fully-connected",
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        equivariant_post_linear: bool = False,
        use_self_connection: bool = True,
        avg_num_neighbors: float | None = None,
        message_scale_init: list[float] | tuple[float, ...] | None = None,
        sc_scale_init: list[float] | tuple[float, ...] | None = None,
        conv_tp_scale_init: str = "none",
        freeze_conv_tp_weight: bool = False,
        interaction_init: str = "identity",
        use_rms_norm: bool = False,
        interaction_attn_heads: int = 0,
        interaction_attn_mode: str = "legacy-softmax",
        phase_enabled: bool = False,
        phase_hidden_channels: int = 32,
        phase_amplitude: str = "unit",
        phase_coefficient: str = "polar",
        phase_context: str = "content",
        phase_normalization: str = "avg-neighbors",
        phase_angular_channels: bool = False,
        phase_heads: int = 1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.use_rms_norm = bool(use_rms_norm)
        self.input_lmax = self.lmax if input_lmax is None else int(input_lmax)
        self.target_lmax = self.lmax if target_lmax is None else int(target_lmax)
        self.sc_lmax = self.input_lmax if sc_lmax is None else int(sc_lmax)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)
        self.equivariant_post_linear = bool(equivariant_post_linear)
        self.use_self_connection = bool(use_self_connection)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.conv_tp_scale_init = str(conv_tp_scale_init)
        if self.conv_tp_scale_init not in {"none", "e3nn"}:
            raise ValueError(f"conv_tp_scale_init must be 'none' or 'e3nn', got {conv_tp_scale_init!r}")
        self.freeze_conv_tp_weight = bool(freeze_conv_tp_weight)
        self.interaction_init = str(interaction_init)
        if self.interaction_init not in {"identity", "mace-random"}:
            raise ValueError(f"interaction_init must be 'identity' or 'mace-random', got {interaction_init!r}")
        allowed_paths = _tp_allowed_paths_from_target_lmax(
            lmax_in1=self.input_lmax,
            lmax_in2=self.lmax,
            lmax_target=self.target_lmax,
        )
        self.linear_up = EquivariantChannelLinearSO3(self.channels, self.input_lmax, bias=False)
        self.tp = EdgeWeightedPathPreservingTensorProduct(
            channels=self.channels,
            lmax=self.lmax,
            allowed_paths=allowed_paths,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
        )
        _init_path_tp_weight_to_one_(self.tp)
        if self.conv_tp_scale_init == "e3nn":
            scales = _mace_like_conv_tp_path_scales(
                self.tp,
                channels=self.channels,
                input_lmax=self.input_lmax,
                edge_lmax=self.lmax,
            )
            with torch.no_grad():
                for path_idx, scale in enumerate(scales):
                    self.tp.weight[path_idx].fill_(float(scale))
        if self.freeze_conv_tp_weight:
            self.tp.weight.requires_grad_(False)
        self.fc = nn.Sequential(
            nn.Linear(self.number_of_basis, 64, bias=False),
            _Normalize2MomSiLU(),
            nn.Linear(64, 64, bias=False),
            _Normalize2MomSiLU(),
            nn.Linear(64, 64, bias=False),
            _Normalize2MomSiLU(),
            nn.Linear(64, self.tp.num_paths * self.channels, bias=False),
        )
        for idx in (0, 2, 4, 6):
            _init_mace_style_linear_(self.fc[idx])
        self.message_linear = PathPreservingLinearSO3(
            {
                l: self.channels * int(self.tp.path_counts_by_l.get(l, 0))
                for l in range(self.target_lmax + 1)
            },
            out_channels=self.channels,
            lmax=self.target_lmax,
        )
        if self.interaction_init == "mace-random":
            _init_so3_linear_mace_style_(self.linear_up)
        else:
            _init_so3_linear_identity_(self.linear_up)
        self.message_selector = (
            ElementConditionedLinearSO3(
                num_elements=num_elements,
                channels=self.channels,
                lmax=self.target_lmax,
                bias=False,
            )
            if not self.use_self_connection
            else None
        )
        if self.interaction_init == "mace-random":
            _init_element_conditioned_mace_style_(self.message_selector)
        else:
            _init_element_conditioned_identity_(self.message_selector)
        self.self_connection = (
            ElementConditionedLinearSO3(
                num_elements=num_elements,
                channels=self.channels,
                lmax=self.sc_lmax,
                bias=False,
            )
            if self.use_self_connection
            else None
        )
        if self.interaction_init == "mace-random":
            _init_element_conditioned_mace_style_(self.self_connection)
        else:
            _init_element_conditioned_identity_(self.self_connection)
        self.message_norm = (
            SO3BlockRMSNorm(self.channels, self.target_lmax) if self.use_rms_norm else nn.Identity()
        )
        self.sc_norm = (
            SO3BlockRMSNorm(self.channels, self.sc_lmax) if (self.use_rms_norm and self.self_connection is not None) else nn.Identity()
        )
        self.message_output_scale = (
            PerLScaleSO3(self.channels, self.target_lmax, message_scale_init)
            if message_scale_init is not None
            else nn.Identity()
        )
        self.sc_output_scale = (
            PerLScaleSO3(self.channels, self.sc_lmax, sc_scale_init)
            if sc_scale_init is not None
            else nn.Identity()
        )
        # --- Optional equivariant neighbor attention. ---
        # Both modes form invariant per-edge, per-head logits from the l=0 node
        # scalars plus a radial bias, and share each scalar weight across every m
        # component in that head. ``legacy-softmax`` is the original DPA-4-style
        # env^2-gated null softmax. ``density-preserving`` instead normalizes the
        # positive gate to cutoff-weighted local mean one and retains the ordinary
        # /avg_num_neighbors density scale. With its zero-initialized logit weights,
        # the latter is exactly the original scatter-sum at initialization.
        self.interaction_attn_heads = int(interaction_attn_heads)
        self.interaction_attn_mode = str(interaction_attn_mode)
        if self.interaction_attn_mode not in {
            "legacy-softmax",
            "density-preserving",
        }:
            raise ValueError(
                "interaction_attn_mode must be 'legacy-softmax' or "
                f"'density-preserving', got {self.interaction_attn_mode!r}"
            )
        self.phase_enabled = bool(phase_enabled)
        self.phase_hidden_channels = int(phase_hidden_channels)
        self.phase_amplitude = str(phase_amplitude)
        self.phase_coefficient = str(phase_coefficient)
        self.phase_context = str(phase_context)
        self.phase_normalization = str(phase_normalization)
        self.phase_angular_channels = bool(phase_angular_channels)
        self.phase_heads = int(phase_heads)
        if self.phase_hidden_channels <= 0:
            raise ValueError(
                f"phase_hidden_channels must be positive, got {self.phase_hidden_channels}"
            )
        if self.phase_heads <= 0 or self.channels % self.phase_heads != 0:
            raise ValueError(
                "phase_heads must be positive and divide channels, got "
                f"phase_heads={self.phase_heads}, channels={self.channels}"
            )
        if self.phase_amplitude not in {"unit", "softplus"}:
            raise ValueError(
                f"phase_amplitude must be 'unit' or 'softplus', got {self.phase_amplitude!r}"
            )
        if self.phase_coefficient not in {"polar", "positive", "signed", "cartesian"}:
            raise ValueError(
                "phase_coefficient must be 'polar', 'positive', 'signed', or "
                f"'cartesian', got {self.phase_coefficient!r}"
            )
        if self.phase_context not in {
            "content",
            "radial",
            "irrep-norm",
            "content-irrep-norm",
        }:
            raise ValueError(
                "phase_context must be 'content', 'radial', 'irrep-norm', or "
                f"'content-irrep-norm', got {self.phase_context!r}"
            )
        if self.phase_normalization not in {"avg-neighbors", "local-effective"}:
            raise ValueError(
                "phase_normalization must be 'avg-neighbors' or 'local-effective', "
                f"got {self.phase_normalization!r}"
            )
        if self.phase_coefficient != "polar" and self.phase_amplitude != "softplus":
            raise ValueError(
                f"phase_coefficient={self.phase_coefficient!r} requires "
                "phase_amplitude='softplus' for a parameter-matched two-head control"
            )
        if (
            self.phase_enabled
            and self.interaction_attn_heads > 0
            and self.interaction_attn_mode != "density-preserving"
        ):
            raise ValueError(
                "phase-enabled interaction can only be combined with density-preserving "
                "neighbor attention; legacy-softmax changes the baseline density scale"
            )
        if self.phase_enabled:
            # Every supported node context is O(3)-invariant.  ``content`` uses
            # signed l=0 channels.  ``irrep-norm`` mirrors the NequIP adapter by
            # using the squared norm of every (l, channel) block, while
            # ``content-irrep-norm`` retains signed l=0 content and appends the
            # squared norms for l>0.  No m component enters the phase MLP
            # directly.
            if self.phase_context in {"irrep-norm", "content-irrep-norm"}:
                phase_node_channels = self.channels * (self.input_lmax + 1)
            else:
                phase_node_channels = self.channels
            self.phase_node_norm = nn.LayerNorm(phase_node_channels)
            self.phase_trunk = nn.Sequential(
                nn.Linear(
                    2 * phase_node_channels + self.number_of_basis,
                    self.phase_hidden_channels,
                ),
                _Normalize2MomSiLU(),
                nn.Linear(self.phase_hidden_channels, self.phase_hidden_channels),
                _Normalize2MomSiLU(),
            )
            phase_output_channels = self.phase_heads * (
                self.target_lmax + 1 if self.phase_angular_channels else 1
            )
            self.phase_head = nn.Linear(
                self.phase_hidden_channels,
                phase_output_channels,
                bias=False,
            )
            for layer in (self.phase_trunk[0], self.phase_trunk[2]):
                _init_mace_style_linear_(layer)
            nn.init.normal_(
                self.phase_head.weight,
                mean=0.0,
                std=0.1 / math.sqrt(float(self.phase_hidden_channels)),
            )
            if self.phase_amplitude == "softplus":
                self.phase_amplitude_head = nn.Linear(
                    self.phase_hidden_channels,
                    phase_output_channels,
                    bias=True,
                )
                with torch.no_grad():
                    self.phase_amplitude_head.weight.zero_()
                    if self.phase_coefficient == "cartesian":
                        # Cartesian two-real-channel control starts at z ~= 1+0i.
                        self.phase_amplitude_head.bias.zero_()
                    else:
                        # softplus^{-1}(1): start as the unit-amplitude model while
                        # retaining a learnable positive amplitude.
                        self.phase_amplitude_head.bias.fill_(math.log(math.expm1(1.0)))
            else:
                self.phase_amplitude_head = None
            self.phase_norm = (
                SO3DoubletRMSNorm(self.channels, self.target_lmax)
                if self.use_rms_norm
                else None
            )
        else:
            self.phase_node_norm = None
            self.phase_trunk = None
            self.phase_head = None
            self.phase_amplitude_head = None
            self.phase_norm = None
        self._fused_selector_message_enabled = False
        self._fused_selector_message_has_bias = False
        if self.interaction_attn_heads > 0:
            if self.channels % self.interaction_attn_heads != 0:
                raise ValueError(
                    f"channels ({self.channels}) must be divisible by interaction_attn_heads ({self.interaction_attn_heads})"
                )
            self.attn_head_dim = self.channels // self.interaction_attn_heads
            self.attn_qk_norm = nn.LayerNorm(self.channels)
            self.attn_q_proj = nn.Linear(self.channels, self.channels, bias=False)
            self.attn_k_proj = nn.Linear(self.channels, self.channels, bias=False)
            self.attn_radial_bias = nn.Linear(self.number_of_basis, self.interaction_attn_heads, bias=False)
            self.attn_logit_w = nn.Parameter(
                torch.empty(self.interaction_attn_heads, self.attn_head_dim)
            )
            if self.interaction_attn_mode == "legacy-softmax":
                self.attn_z_bias_raw = nn.Parameter(
                    torch.zeros(self.interaction_attn_heads)
                )
                # Preserve the historical gentle-start initialization exactly.
                nn.init.normal_(self.attn_logit_w, mean=0.0, std=0.01)
            else:
                self.register_parameter("attn_z_bias_raw", None)
                # score == 0 -> exp(score) == 1 -> local-mean gate == 1.
                # Gradients still reach attn_logit_w on the first optimizer step.
                nn.init.zeros_(self.attn_logit_w)
            nn.init.zeros_(self.attn_radial_bias.weight)
        else:
            self.attn_head_dim = 0
            self.attn_qk_norm = None
            self.attn_q_proj = None
            self.attn_k_proj = None
            self.attn_radial_bias = None
            self.attn_z_bias_raw = None
            self.attn_logit_w = None

    def _phase_effective_coordination(
        self,
        *,
        edge_env: torch.Tensor | None,
        edge_mask: torch.Tensor | None,
        edge_dst: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
        sync_after_scatter: callable | None,
    ) -> torch.Tensor:
        """Return the smooth participation-ratio coordination.

            (sum_j w_ij)^2 / sum_j w_ij^2,

        with the polynomial cutoff envelope as w. It equals the ordinary
        neighbor count for equal weights and changes smoothly at the cutoff.
        """
        if edge_env is None:
            raise ValueError(
                "effective coordination requires the cutoff envelope"
            )
        weight = edge_env.reshape(-1).to(dtype=dtype).clamp_min(0.0)
        if edge_mask is not None:
            weight = weight * edge_mask.reshape(-1).to(dtype=dtype)
        sum_weight = scatter(
            weight,
            edge_dst,
            dim=0,
            dim_size=int(num_nodes),
            reduce="sum",
        )
        sum_weight_sq = scatter(
            weight.square(),
            edge_dst,
            dim=0,
            dim_size=int(num_nodes),
            reduce="sum",
        )
        if sync_after_scatter is not None:
            sum_weight = sync_after_scatter(sum_weight)
            sum_weight_sq = sync_after_scatter(sum_weight_sq)
        return sum_weight.square() / sum_weight_sq.clamp_min(
            1.0e-12
        )

    def _phase_local_normalizer(
        self,
        *,
        edge_env: torch.Tensor | None,
        edge_mask: torch.Tensor | None,
        edge_dst: torch.Tensor,
        num_nodes: int,
        avg_num_neighbors: float,
        dtype: torch.dtype,
        sync_after_scatter: callable | None,
    ) -> torch.Tensor:
        """Smooth 1/sqrt(avg_n * n_eff(i)) charged-stream normalization."""
        effective_coordination = self._phase_effective_coordination(
            edge_env=edge_env,
            edge_mask=edge_mask,
            edge_dst=edge_dst,
            num_nodes=num_nodes,
            dtype=dtype,
            sync_after_scatter=sync_after_scatter,
        )
        anchor = max(float(avg_num_neighbors), 1.0e-8)
        return torch.rsqrt(
            (anchor * effective_coordination.clamp_min(1.0)).to(dtype=dtype)
        )

    def enable_eval_fused_selector_message(self) -> bool:
        """Precompose message_linear and element selector for eval/AOTI inference.

        The first MACE residual block applies, per l:
            scatter -> message_linear -> /avg_num_neighbors -> element selector -> output scale.
        When the selector is present and all these maps are linear, precompose the
        fixed weights into one element-conditioned linear map. This changes only
        fp32 accumulation order and is disabled for training by default.
        """
        if self.interaction_attn_heads > 0 or self.message_selector is None:
            self._fused_selector_message_enabled = False
            return False
        if self.avg_num_neighbors is None:
            self._fused_selector_message_enabled = False
            return False
        if not isinstance(self.message_norm, nn.Identity):
            self._fused_selector_message_enabled = False
            return False

        if isinstance(self.message_output_scale, nn.Identity):
            scales = [1.0 for _ in range(self.target_lmax + 1)]
        elif isinstance(self.message_output_scale, PerLScaleSO3):
            scales = [
                float(v)
                for v in self.message_output_scale.log_scale.detach().exp().cpu().tolist()
            ]
        else:
            self._fused_selector_message_enabled = False
            return False

        selector_bias = getattr(self.message_selector, "bias", None)
        self._fused_selector_message_has_bias = selector_bias is not None
        avg = float(self.avg_num_neighbors)
        with torch.no_grad():
            for l in range(self.target_lmax + 1):
                msg_w = self.message_linear.weights[str(l)].detach()
                sel_w = self.message_selector.weights[str(l)].detach()
                fused = torch.einsum("eoj,jc->eoc", sel_w, msg_w) * (float(scales[l]) / avg)
                name = f"_fused_selector_message_weight_{l}"
                if name in self._buffers:
                    self._buffers[name] = fused.contiguous()
                else:
                    self.register_buffer(name, fused.contiguous(), persistent=False)
                if selector_bias is not None:
                    bias = selector_bias[str(l)].detach() * float(scales[l])
                    bias_name = f"_fused_selector_message_bias_{l}"
                    if bias_name in self._buffers:
                        self._buffers[bias_name] = bias.contiguous()
                    else:
                        self.register_buffer(bias_name, bias.contiguous(), persistent=False)
        self._fused_selector_message_enabled = True
        return True

    def _attention_logit(
        self,
        node_feats_l0: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Return invariant content-plus-radial logits with shape ``(E, H)``."""
        H = self.interaction_attn_heads
        d = self.attn_head_dim
        qk = self.attn_qk_norm(node_feats_l0)
        q = self.attn_q_proj(qk).reshape(-1, H, d)
        k = self.attn_k_proj(qk).reshape(-1, H, d)
        logit = (q[edge_dst] * k[edge_src] * self.attn_logit_w).sum(-1)  # (E, H)
        return logit + self.attn_radial_bias(edge_feats)

    def _attention_weight(
        self,
        node_feats_l0: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_env: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return legacy softmax weights or density-preserving positive gates.

        ``legacy-softmax``:

            alpha_ij = env_ij^2 exp(s_ij)
                       / (softplus(zeta) + sum_k env_ik^2 exp(s_ik)).

        ``density-preserving``:

            g_ij = exp(s_ij)
                   / [sum_k env_ik exp(s_ik) / sum_k env_ik].

        The second identity implies ``sum_j env_ij g_ij = sum_j env_ij``.
        It uses the cutoff only to define a smooth local mean; the edge message
        itself is not multiplied by another envelope.
        """
        H = self.interaction_attn_heads
        logit = self._attention_logit(
            node_feats_l0,
            edge_feats,
            edge_src,
            edge_dst,
        )
        env2 = edge_env.reshape(-1, 1).to(dtype=logit.dtype).clamp_min(0.0).square()  # (E, 1)
        # A destination-wise shift makes every exponential <= 1 and is cancelled
        # exactly by either normalization. torch-native scatter_reduce avoids the
        # non-differentiable argmax returned by torch_scatter.scatter_max and stays
        # compatible with compiled autograd/CUDA graph capture.
        gmax = logit.new_zeros(num_nodes, H).scatter_reduce_(
            0, edge_dst.unsqueeze(-1).expand(-1, H), logit, reduce="amax", include_self=False
        ).clamp_min(0.0)  # (N, H)
        exp_shifted = torch.exp(logit - gmax[edge_dst])
        if self.interaction_attn_mode == "legacy-softmax":
            ex = env2 * exp_shifted
            denom = scatter(
                ex, edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
            )
            if self.attn_z_bias_raw is None:
                raise RuntimeError("legacy-softmax attention requires a null logit")
            zeta = F.softplus(self.attn_z_bias_raw).reshape(1, H).to(
                dtype=logit.dtype
            )
            denom = denom + zeta * torch.exp(-gmax)
            return ex / denom[edge_dst].clamp_min(1.0e-20)

        env = edge_env.reshape(-1, 1).to(dtype=logit.dtype).clamp_min(0.0)
        weighted_exp_sum = scatter(
            env * exp_shifted,
            edge_dst,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        env_sum = scatter(
            env.expand(-1, H),
            edge_dst,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        local_mean = weighted_exp_sum / env_sum.clamp_min(1.0e-12)
        return exp_shifted / local_mean[edge_dst].clamp_min(1.0e-12)

    def _phase_stream_block(
        self,
        edge_block: torch.Tensor,
        phase_cos: torch.Tensor,
        phase_sin: torch.Tensor,
        l: int,
    ) -> torch.Tensor:
        """Return neutral/real/imag edge streams with channel-group phases.

        A head acts on the same base-channel group in every tensor-product
        path.  The representation size is unchanged: this only replaces one
        broadcast scalar multiplication by a grouped broadcast.
        """
        heads = self.phase_heads
        offset = l * heads if self.phase_angular_channels else 0
        cos_l = phase_cos[:, offset : offset + heads]
        sin_l = phase_sin[:, offset : offset + heads]
        edge_count, path_channels, m_count = edge_block.shape
        if path_channels % self.channels != 0:
            raise RuntimeError(
                "phase grouping requires path channels to be a multiple of "
                f"base channels, got {path_channels} and {self.channels}"
            )
        path_count = path_channels // self.channels
        grouped = edge_block.reshape(
            edge_count,
            path_count,
            heads,
            self.channels // heads,
            m_count,
        )
        cos_gate = cos_l.reshape(edge_count, 1, heads, 1, 1)
        sin_gate = sin_l.reshape(edge_count, 1, heads, 1, 1)
        charged_real = (grouped * cos_gate).reshape(
            edge_count, path_channels, m_count
        )
        charged_imag = (grouped * sin_gate).reshape(
            edge_count, path_channels, m_count
        )
        return torch.stack((edge_block, charged_real, charged_imag), dim=1)

    def forward(
        self,
        *,
        node_attrs: torch.Tensor | None,
        node_feats: torch.Tensor,
        edge_attrs: Dict[int, torch.Tensor],
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        edge_env: torch.Tensor | None = None,
        node_type_idx: torch.Tensor | None = None,
        sync_after_scatter: callable | None = None,
        return_phase_doublet: bool = False,
        return_phase_edges: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        num_nodes = node_feats.size(0)

        node_feats_for_sc = node_feats
        node_feats = self.linear_up(node_feats)
        gates = self.fc(edge_feats)
        x1 = _split_irreps(node_feats, self.channels, self.input_lmax)
        x1e = {l: x1[l][edge_src] for l in range(self.input_lmax + 1)}
        edge_blocks = self.tp(x1e, edge_attrs, gates)
        if edge_mask is not None:
            mask = edge_mask.to(dtype=node_feats.dtype)
            edge_blocks = {l: block * mask.unsqueeze(-1) for l, block in edge_blocks.items()}
        phase_cos: torch.Tensor | None = None
        phase_sin: torch.Tensor | None = None
        if self.phase_enabled:
            scalar_nodes = x1[0].squeeze(-1)
            if self.phase_context == "irrep-norm":
                node_context = torch.cat(
                    [x1[l].square().sum(dim=-1) for l in range(self.input_lmax + 1)],
                    dim=-1,
                )
            elif self.phase_context == "content-irrep-norm":
                node_context = torch.cat(
                    [scalar_nodes]
                    + [x1[l].square().sum(dim=-1) for l in range(1, self.input_lmax + 1)],
                    dim=-1,
                )
            else:
                node_context = scalar_nodes
            normalized_node_context = self.phase_node_norm(node_context)
            if self.phase_context == "radial":
                # Preserve the exact trunk shape and nominal parameter budget while
                # removing all chemical-content information from the coefficient.
                normalized_node_context = torch.zeros_like(normalized_node_context)
            phase_context = torch.cat(
                (
                    normalized_node_context[edge_dst],
                    normalized_node_context[edge_src],
                    edge_feats,
                ),
                dim=-1,
            )
            phase_hidden = self.phase_trunk(phase_context)
            theta = self.phase_head(phase_hidden)
            if self.phase_amplitude_head is None:
                amplitude = torch.ones_like(theta)
                amplitude_raw = None
            else:
                amplitude_raw = self.phase_amplitude_head(phase_hidden)
                amplitude = F.softplus(amplitude_raw)
            if self.phase_coefficient == "polar":
                phase_cos = amplitude.mul(torch.cos(theta)).unsqueeze(-1)
                phase_sin = amplitude.mul(torch.sin(theta)).unsqueeze(-1)
            elif self.phase_coefficient == "positive":
                if amplitude_raw is None:
                    raise RuntimeError("positive coefficient requires an amplitude head")
                # Both scalar heads remain active while the coefficient stays positive.
                positive = F.softplus(amplitude_raw + theta)
                phase_cos = positive.unsqueeze(-1)
                phase_sin = torch.zeros_like(phase_cos)
            elif self.phase_coefficient == "signed":
                if amplitude_raw is None:
                    raise RuntimeError("signed coefficient requires an amplitude head")
                # Start close to the same +1 real coefficient as the polar and
                # Cartesian controls, while retaining an unconstrained path through
                # zero to negative weights. Starting at tanh(0)=0 would suppress
                # the entire branch and confound parameterization with initialization.
                signed = amplitude.mul(1.0 + theta)
                phase_cos = signed.unsqueeze(-1)
                phase_sin = torch.zeros_like(phase_cos)
            else:
                if amplitude_raw is None:
                    raise RuntimeError("cartesian coefficient requires a second real head")
                # Unconstrained Cartesian doublet, initialized close to 1+0i.
                phase_cos = (1.0 + theta).unsqueeze(-1)
                phase_sin = amplitude_raw.unsqueeze(-1)
        selector_message_fused = False
        phase_doublet = None
        phase_doublet_blocks: Dict[int, torch.Tensor] | None = None
        message_phase_blocks: Dict[int, torch.Tensor] | None = None
        phase_node_normalizer = None
        if self.interaction_attn_heads > 0 and self.phase_enabled:
            if edge_env is None:
                raise ValueError(
                    "interaction_attn_heads > 0 requires edge_env to be passed to forward()"
                )
            H = self.interaction_attn_heads
            attention_weight = self._attention_weight(
                x1[0].squeeze(-1),
                edge_feats,
                edge_src,
                edge_dst,
                edge_env,
                num_nodes,
            )
            attention_gate = attention_weight.reshape(-1, H, 1, 1)
            attended_edge_blocks: Dict[int, torch.Tensor] = {}
            for l in range(self.target_lmax + 1):
                edge_block = edge_blocks[l]
                edge_count, path_channels, m_count = edge_block.shape
                attended_edge_blocks[l] = (
                    edge_block.reshape(
                        edge_count,
                        H,
                        path_channels // H,
                        m_count,
                    )
                    * attention_gate
                ).reshape(edge_count, path_channels, m_count)
            edge_blocks = attended_edge_blocks
        if self.interaction_attn_heads > 0 and not self.phase_enabled:
            if edge_env is None:
                raise ValueError("interaction_attn_heads > 0 requires edge_env to be passed to forward()")
            H = self.interaction_attn_heads
            attention_weight = self._attention_weight(
                x1[0].squeeze(-1), edge_feats, edge_src, edge_dst, edge_env, num_nodes
            )
            a = attention_weight.reshape(-1, H, 1, 1)
            message_blocks = {}
            for l in range(self.target_lmax + 1):
                eb = edge_blocks[l]  # (E, C_l, 2l+1); C_l = channels * num_paths_l, H | channels => H | C_l
                e_n, c_l, m = eb.shape
                eb = (eb.reshape(e_n, H, c_l // H, m) * a).reshape(e_n, c_l, m)
                message_blocks[l] = scatter(eb, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
            message = self.message_linear(message_blocks)
            if self.interaction_attn_mode == "density-preserving":
                if self.avg_num_neighbors is None:
                    if edge_mask is not None:
                        avg_num_neighbors = float(
                            edge_mask.detach().sum().item()
                        ) / float(max(num_nodes, 1))
                    else:
                        avg_num_neighbors = float(edge_src.numel()) / float(
                            max(num_nodes, 1)
                        )
                else:
                    avg_num_neighbors = self.avg_num_neighbors
                message = message / max(avg_num_neighbors, 1.0e-8)
        else:
            if self.phase_enabled:
                if phase_cos is None or phase_sin is None:
                    raise RuntimeError("phase gates were not constructed")
                # Aggregate the neutral stream and both q=1 components in one
                # scatter.  Their edge geometry is identical; only the scalar
                # gate differs.  Keeping the stream axis as a leading batch
                # dimension also lets the shared SO(3) channel map below use
                # one batched linear operation instead of separate neutral and
                # charged calls.
                message_phase_blocks = {
                    l: scatter(
                        self._phase_stream_block(
                            edge_blocks[l],
                            phase_cos,
                            phase_sin,
                            l,
                        ),
                        edge_dst,
                        dim=0,
                        dim_size=num_nodes,
                        reduce="sum",
                    )
                    for l in range(self.target_lmax + 1)
                }
                message_blocks = {
                    l: block[:, 0] for l, block in message_phase_blocks.items()
                }
                phase_doublet_blocks = {
                    l: block[:, 1:] for l, block in message_phase_blocks.items()
                }
            else:
                message_blocks = {
                    l: scatter(
                        edge_blocks[l], edge_dst, dim=0, dim_size=num_nodes, reduce="sum"
                    )
                    for l in range(self.target_lmax + 1)
                }
            if getattr(self, "_fused_selector_message_enabled", False) and sync_after_scatter is None:
                type_idx = (
                    node_type_idx.to(device=node_feats.device, dtype=torch.long)
                    if node_type_idx is not None
                    else None
                )
                if type_idx is None:
                    if node_attrs is None:
                        raise ValueError("node_attrs is required when node_type_idx is not provided")
                    attrs = node_attrs.to(dtype=node_feats.dtype)
                else:
                    attrs = None
                out_blocks: Dict[int, torch.Tensor] = {}
                for l in range(self.target_lmax + 1):
                    weight = getattr(self, f"_fused_selector_message_weight_{l}").to(
                        dtype=message_blocks[l].dtype,
                        device=message_blocks[l].device,
                    )
                    if type_idx is None:
                        if attrs is None:
                            raise ValueError("node_attrs is required when node_type_idx is not provided")
                        mixed_weight = torch.einsum("ne,eoc->noc", attrs, weight)
                        out_block = torch.einsum("noc,ncm->nom", mixed_weight, message_blocks[l])
                    else:
                        mixed_weight = weight.index_select(0, type_idx)
                        out_block = torch.einsum("noc,ncm->nom", mixed_weight, message_blocks[l])
                    if self._fused_selector_message_has_bias:
                        bias = getattr(self, f"_fused_selector_message_bias_{l}").to(
                            dtype=out_block.dtype,
                            device=out_block.device,
                        )
                        if type_idx is None:
                            if attrs is None:
                                raise ValueError("node_attrs is required when node_type_idx is not provided")
                            mixed_bias = torch.einsum("ne,eo->no", attrs, bias)
                        else:
                            mixed_bias = bias.index_select(0, type_idx)
                        out_block = out_block + mixed_bias.unsqueeze(-1)
                    out_blocks[l] = out_block
                message = _merge_irreps(out_blocks, self.channels, self.target_lmax)
                selector_message_fused = True
            else:
                if self.avg_num_neighbors is None:
                    if edge_mask is not None:
                        avg_num_neighbors = float(edge_mask.detach().sum().item()) / float(max(num_nodes, 1))
                    else:
                        avg_num_neighbors = float(edge_src.numel()) / float(max(num_nodes, 1))
                else:
                    avg_num_neighbors = self.avg_num_neighbors
                if message_phase_blocks is None:
                    message = self.message_linear(message_blocks) / max(
                        avg_num_neighbors, 1e-8
                    )
                else:
                    if self.phase_normalization == "avg-neighbors":
                        message_phase = self.message_linear(
                            message_phase_blocks
                        ) / max(avg_num_neighbors, 1e-8)
                        message = message_phase[:, 0]
                        phase_doublet = message_phase[:, 1:]
                    else:
                        message_phase = self.message_linear(message_phase_blocks)
                        message = message_phase[:, 0] / max(
                            avg_num_neighbors, 1e-8
                        )
                        phase_node_normalizer = self._phase_local_normalizer(
                            edge_env=edge_env,
                            edge_mask=edge_mask,
                            edge_dst=edge_dst,
                            num_nodes=num_nodes,
                            avg_num_neighbors=avg_num_neighbors,
                            dtype=message_phase.dtype,
                            sync_after_scatter=sync_after_scatter,
                        )
                        phase_doublet = (
                            message_phase[:, 1:]
                            * phase_node_normalizer[:, None, None]
                        )
        phase_real = None
        phase_imag = None
        phase_edge_orbital = None
        phase_edge_norm_sq = None
        if self.phase_enabled:
            if phase_cos is None or phase_sin is None:
                raise RuntimeError("phase gates were not constructed")
            if phase_doublet is None:
                if phase_doublet_blocks is None:
                    phase_doublet_blocks = {
                        l: scatter(
                            self._phase_stream_block(
                                edge_blocks[l],
                                phase_cos,
                                phase_sin,
                                l,
                            )[:, 1:],
                            edge_dst,
                            dim=0,
                            dim_size=num_nodes,
                            reduce="sum",
                        )
                        for l in range(self.target_lmax + 1)
                    }
                if self.avg_num_neighbors is None:
                    if edge_mask is not None:
                        phase_avg_num_neighbors = float(
                            edge_mask.detach().sum().item()
                        ) / float(max(num_nodes, 1))
                    else:
                        phase_avg_num_neighbors = float(edge_src.numel()) / float(
                            max(num_nodes, 1)
                        )
                else:
                    phase_avg_num_neighbors = self.avg_num_neighbors
                phase_doublet = self.message_linear(phase_doublet_blocks)
                if self.phase_normalization == "avg-neighbors":
                    phase_doublet = phase_doublet / max(
                        phase_avg_num_neighbors, 1.0e-8
                    )
                else:
                    phase_node_normalizer = self._phase_local_normalizer(
                        edge_env=edge_env,
                        edge_mask=edge_mask,
                        edge_dst=edge_dst,
                        num_nodes=num_nodes,
                        avg_num_neighbors=phase_avg_num_neighbors,
                        dtype=phase_doublet.dtype,
                        sync_after_scatter=sync_after_scatter,
                    )
                    phase_doublet = (
                        phase_doublet * phase_node_normalizer[:, None, None]
                    )
            if return_phase_edges:
                if self.phase_angular_channels or self.phase_heads != 1:
                    raise ValueError(
                        "edge-factorized phase density requires a shared phase; "
                        "per-L or grouped phases are supported by full-nonlinear only"
                    )
                if self.phase_norm is not None:
                    raise ValueError(
                        "diagonal edge density is not defined with interaction RMS norm enabled"
                    )
                if self.avg_num_neighbors is None:
                    if edge_mask is not None:
                        phase_avg_num_neighbors = float(
                            edge_mask.detach().sum().item()
                        ) / float(max(num_nodes, 1))
                    else:
                        phase_avg_num_neighbors = float(edge_src.numel()) / float(
                            max(num_nodes, 1)
                        )
                else:
                    phase_avg_num_neighbors = self.avg_num_neighbors
                # The phase coefficient is a single invariant scalar shared by
                # every channel, path and l block, hence it commutes with the
                # real SO(3) message map.  Return one real edge orbital plus its
                # scalar U(1)-doublet norm instead of a 2x larger edge doublet.
                phase_edge_orbital = self.message_linear(edge_blocks)
                phase_edge_norm_sq = (
                    phase_cos.square() + phase_sin.square()
                ).reshape(-1)
                if self.phase_normalization == "avg-neighbors":
                    denominator = max(phase_avg_num_neighbors, 1.0e-8)
                    phase_edge_norm_sq = phase_edge_norm_sq / (
                        denominator * denominator
                    )
                else:
                    if phase_node_normalizer is None:
                        phase_node_normalizer = self._phase_local_normalizer(
                            edge_env=edge_env,
                            edge_mask=edge_mask,
                            edge_dst=edge_dst,
                            num_nodes=num_nodes,
                            avg_num_neighbors=phase_avg_num_neighbors,
                            dtype=phase_edge_orbital.dtype,
                            sync_after_scatter=sync_after_scatter,
                        )
                    edge_normalizer = phase_node_normalizer.index_select(
                        0, edge_dst
                    )
                    phase_edge_norm_sq = (
                        phase_edge_norm_sq * edge_normalizer.square()
                    )
        if sync_after_scatter is not None:
            message = sync_after_scatter(message)
            if phase_doublet is not None:
                phase_doublet = sync_after_scatter(phase_doublet)
        if not self.use_self_connection and not selector_message_fused:
            if node_attrs is None:
                raise ValueError("node_attrs is required for the unfused message selector path")
            message = self.message_selector(message, node_attrs)
        if not selector_message_fused:
            message = self.message_norm(message)
            message = self.message_output_scale(message)
        if phase_doublet is not None:
            phase_real = phase_doublet[..., 0, :]
            phase_imag = phase_doublet[..., 1, :]
            if self.phase_norm is not None:
                phase_real, phase_imag = self.phase_norm(phase_real, phase_imag)
                phase_doublet = torch.stack((phase_real, phase_imag), dim=-2)
            phase_doublet = self.message_output_scale(phase_doublet)
            phase_real = phase_doublet[..., 0, :]
            phase_imag = phase_doublet[..., 1, :]
        if phase_edge_orbital is not None:
            phase_edge_orbital = self.message_output_scale(phase_edge_orbital)
        sc = None
        if self.self_connection is not None:
            if self.sc_lmax == self.input_lmax:
                sc_input = node_feats_for_sc
            elif self.sc_lmax == 0:
                sc_input = _split_irreps(node_feats_for_sc, self.channels, self.input_lmax)[0].reshape(
                    node_feats_for_sc.shape[0], self.channels
                )
            else:
                raise ValueError(
                    f"Unsupported ICTC self-connection projection input_lmax={self.input_lmax}, sc_lmax={self.sc_lmax}"
                )
            if (not self.training) and node_type_idx is not None:
                sc = self.self_connection.forward_type_idx(sc_input, node_type_idx)
            else:
                if node_attrs is None:
                    raise ValueError("node_attrs is required for the training self-connection path")
                sc = self.self_connection(sc_input, node_attrs)
            sc = self.sc_norm(sc)
            sc = self.sc_output_scale(sc)
        if return_phase_doublet and phase_doublet is not None:
            if return_phase_edges:
                if phase_edge_orbital is None or phase_edge_norm_sq is None:
                    raise RuntimeError(
                        "factorized phase edge state was requested but not constructed"
                    )
                return (
                    message,
                    sc,
                    phase_doublet,
                    phase_edge_orbital,
                    phase_edge_norm_sq,
                )
            return message, sc, phase_doublet
        if phase_real is not None and phase_imag is not None:
            return message, sc, phase_real, phase_imag
        return message, sc


class PureCartesianICTDFix(nn.Module):
    """
    ICTC-SO3 model organized with a MACE-style backbone:

      h_t -> interaction_t(node_attrs, h_t, edge_*) = (m_t, sc_t)
          -> product_t(m_t, sc_t, node_attrs) = h_{t+1}
          -> layer_readout_t(h_{t+1})

    Optional route:
      - baseline: sum(layerwise readouts)
      - fusion:   sum(layerwise readouts) + E_fusion(h_1, ..., h_N)
    """

    def __init__(
        self,
        max_embed_radius: float,
        main_max_radius: float,
        main_number_of_basis: int,
        hidden_dim_conv: int,
        hidden_dim_sh: int,
        hidden_dim: int,
        channel_in2: int = 32,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        atomic_numbers: list[int] | tuple[int, ...] | None = None,
        output_size: int = 8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers: int = 1,
        num_interaction: int = 2,
        device=None,
        function_type_main: str = "gaussian",
        lmax: int = 2,
        ictd_Lmax: int = 6,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        max_rank_other: int = 1,
        k_policy: str = "k0",
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        product5_muls_by_l: dict[int, int] | None = None,
        invariant_channels: int = 32,
        long_range_mode: str = "none",
        long_range_hidden_dim: int = 64,
        long_range_boundary: str = "nonperiodic",
        long_range_neutralize: bool = True,
        long_range_filter_hidden_dim: int = 64,
        long_range_kmax: int = 2,
        long_range_mesh_size: int = 16,
        long_range_slab_padding_factor: int = 2,
        long_range_include_k0: bool = False,
        long_range_source_channels: int = 1,
        long_range_backend: str = "dense_pairwise",
        long_range_reciprocal_backend: str = "direct_kspace",
        long_range_energy_partition: str = "potential",
        long_range_green_mode: str = "poisson",
        long_range_assignment: str = "pcs",
        long_range_mesh_fft_full_ewald: bool = False,
        long_range_mesh_fft_reciprocal_only: bool = False,
        long_range_max_multipole_l: int = 0,
        long_range_multipole_gate_init: float = 0.1,
        long_range_dispersion: bool = False,
        long_range_dispersion_mode: str | None = None,
        dispersion_cutoff: float = 8.0,
        dispersion_max_num_neighbors: int | None = None,
        dispersion_neighbor_method: str = "auto",
        dispersion_bruteforce_threshold: int = 1024,
        dispersion_allow_large_bruteforce_fallback: bool = False,
        dispersion_slq_num_probes: int = 8,
        dispersion_slq_lanczos_steps: int = 16,
        mbd_operator_backend: str = "edge_sparse",
        mbd_pme_mesh_size: int = 32,
        mbd_pme_assignment: str = "pcs",
        mbd_pme_k_norm_floor: float = 1.0e-6,
        mbd_pme_assignment_window_floor: float = 1.0e-6,
        mbd_pme_ewald_alpha_prefactor: float = 5.0,
        mbd_anisotropic_polarizability: bool = False,
        mbd_learnable_energy_scale: bool = True,
        mbd_alpha_floor: float = 1.0e-4,
        dispersion_min_cutoff: float = 0.0,
        dispersion_switch_width: float = 0.5,
        long_range_theta: float = 0.5,
        long_range_leaf_size: int = 32,
        long_range_multipole_order: int = 0,
        long_range_far_source_dim: int = 16,
        long_range_far_num_shells: int = 3,
        long_range_far_shell_growth: float = 2.0,
        long_range_far_tail: bool = True,
        long_range_far_tail_bins: int = 2,
        long_range_far_stats: str = "mean,count,mean_r,rms_r",
        long_range_far_max_radius_multiplier: float | None = None,
        long_range_far_source_norm: bool = True,
        long_range_far_gate_init: float = 0.0,
        feature_spectral_mode: str = "none",
        feature_spectral_bottleneck_dim: int = 8,
        feature_spectral_mesh_size: int = 16,
        feature_spectral_filter_hidden_dim: int = 64,
        feature_spectral_boundary: str = "periodic",
        feature_spectral_slab_padding_factor: int = 2,
        feature_spectral_neutralize: bool = True,
        feature_spectral_include_k0: bool = False,
        feature_spectral_assignment: str = "pcs",
        feature_spectral_gate_init: float = 0.0,
        equivariant_post_linear: bool = False,
        ictd_save_tp_mode: str = "fully-connected",
        ictd_fix_route: str = "baseline",
        ictd_fix_contraction_combine: str = "softmax",
        ictd_fix_product_backend: str = "ictd-bridge-u",
        ictd_fix_use_reduced_cg: bool = False,
        ictd_fix_first_layer_self_connection: bool = False,
        ictd_fix_conv_tp_scale_init: str = "none",
        ictd_fix_freeze_conv_tp_weight: bool = False,
        ictd_fix_interaction_init: str = "identity",
        ictd_fix_edge_lmax: int | None = None,
        angular_basis: str = "ictd",
        ictd_fix_interaction_scale: str = "none",
        ictd_fix_fusion_scale_init: float = 0.1,
        ictd_fix_fusion_heads: int = 1,
        ictd_fix_fusion_head_weight_mode: str = "softmax",
        ictd_fix_fusion_input_scale_init: float = 1.0,
        ictd_fix_fusion_input_scale_trainable: bool = False,
        ictd_fix_fusion_depth_attention: bool = False,
        ictd_fix_gmix_gate_init: float = 1.0,
        ictd_fix_gmix_gate_trainable: bool = False,
        ictd_fix_gmix_block_rmsnorm: bool = False,
        ictd_fix_gmix_block_rmsnorm_gamma_init: float = 1.0,
        ictd_fix_readout_head_scale_init: float = 1.0,
        ictd_fix_readout_head_scale_trainable: bool = False,
        ictd_fix_fusion_readout_mixed_channels: bool = True,
        ictd_fix_fusion_pre_product_norm: bool = True,
        ictd_fix_interaction_rms_norm: bool = False,
        radial_sqrt_num_basis: bool = False,
        ictd_fix_interaction_attn_heads: int = 0,
        ictd_fix_interaction_attn_mode: str = "legacy-softmax",
        ictd_fix_interaction_attn_scope: str = "all",
        ictd_fix_phase_mode: str = "none",
        ictd_fix_phase_hidden_channels: int = 32,
        ictd_fix_phase_residual_scale_init: float = 0.05,
        ictd_fix_phase_amplitude: str = "unit",
        ictd_fix_phase_coefficient: str = "polar",
        ictd_fix_phase_context: str = "content",
        ictd_fix_phase_placement: str = "post-product",
        ictd_fix_phase_density_rank: int = 8,
        ictd_fix_phase_density_species_mode: str = "onehot-full",
        ictd_fix_phase_density_species_embedding_dim: int = 16,
        ictd_fix_phase_density_species_rank: int = 16,
        ictd_fix_phase_density_pairs: str = "full",
        ictd_fix_phase_coherence_init: float = 0.1,
        ictd_fix_phase_normalization: str = "avg-neighbors",
        ictd_fix_phase_scope: str = "final",
        ictd_fix_phase_heads: int = 1,
        ictd_fix_gmix_energy_readout: bool = True,
        ictd_fix_gmix_readout_scale_init: float | None = None,
        ictd_fix_gmix_readout_output_init_std: float = 0.003,
        ictd_fix_gmix_output_lmax: int | None = None,
        ictd_fix_readout_hidden_channels: int = 16,
        ictd_fix_nonlinear_layer_readouts: bool = False,
        ictd_fix_final_layer_readout_only: bool = False,
        ictd_fix_element_energy_correction: bool = False,
        ictd_fix_scalar_ffn: bool = False,
        ictd_fix_layer_readout_output_init_std: float = 0.003,
        polynomial_cutoff_p: int | None = 6,
        save_contraction_order: int = 3,
        save_multiple_mix_channels: int | None = None,
        avg_num_neighbors: float | None = None,
        energy_output_scale: float = 1.0,
        energy_output_scale_enabled: bool = False,
        energy_output_shift: float = 0.0,
        energy_output_shift_enabled: bool = False,
    ):
        super().__init__()
        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if int(num_interaction) < 1:
            raise ValueError(f"num_interaction must be >= 1, got {num_interaction}")
        if ictd_fix_route != "baseline":
            raise ValueError(
                f"this baseline-only MACE-ICTC build supports ictd_fix_route='baseline' only, "
                f"got {ictd_fix_route!r}"
            )
        if ictd_fix_contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(
                f"ictd_fix_contraction_combine must be 'softmax', 'free', or 'path-free', got {ictd_fix_contraction_combine!r}"
            )
        self.angular_basis = str(angular_basis)
        if self.angular_basis not in {"ictd", "e3nn"}:
            raise ValueError(f"angular_basis must be 'ictd' or 'e3nn', got {self.angular_basis!r}")
        # Lazily applied on first forward (folds the fixed angular operators into the e3nn basis).
        self._e3nn_folded = False
        self._e3nn_q_blocks: List[torch.Tensor] = []
        requested_product_backend = str(ictd_fix_product_backend)
        if requested_product_backend == "ictd-mace-u":
            requested_product_backend = "ictd-bridge-u"
        if requested_product_backend not in {"ictd", "native-mace", "ictd-bridge-u", "ictd-pure-u", "cueq"}:
            raise ValueError(
                "ictd_fix_product_backend must be 'ictd', 'native-mace', 'ictd-bridge-u', "
                f"'ictd-mace-u' alias, 'ictd-pure-u', or 'cueq', got {ictd_fix_product_backend!r}"
            )
        if ictd_fix_interaction_scale not in {"none", "mace-rms"}:
            raise ValueError(
                f"ictd_fix_interaction_scale must be 'none' or 'mace-rms', got {ictd_fix_interaction_scale!r}"
            )
        if ictd_fix_conv_tp_scale_init not in {"none", "e3nn"}:
            raise ValueError(
                "ictd_fix_conv_tp_scale_init must be 'none' or 'e3nn', "
                f"got {ictd_fix_conv_tp_scale_init!r}"
            )
        if ictd_fix_interaction_init not in {"identity", "mace-random"}:
            raise ValueError(
                "ictd_fix_interaction_init must be 'identity' or 'mace-random', "
                f"got {ictd_fix_interaction_init!r}"
            )
        if ictd_fix_fusion_head_weight_mode not in {"softmax", "free"}:
            raise ValueError(
                "ictd_fix_fusion_head_weight_mode must be 'softmax' or 'free', "
                f"got {ictd_fix_fusion_head_weight_mode!r}"
            )
        if feature_spectral_mode != "none":
            raise NotImplementedError("pure-cartesian-ictd-fix currently supports only feature_spectral_mode=none (long_range is wired)")

        self.channels = int(hidden_dim_conv)
        self.lmax = int(lmax)
        self.ictd_fix_edge_lmax = self.lmax if ictd_fix_edge_lmax is None else int(ictd_fix_edge_lmax)
        if self.ictd_fix_edge_lmax < 0:
            raise ValueError(f"ictd_fix_edge_lmax must be >= 0, got {self.ictd_fix_edge_lmax}")
        if self.ictd_fix_edge_lmax < self.lmax:
            raise NotImplementedError(
                "MACE-ICTC currently requires ictd_fix_edge_lmax >= lmax; "
                f"got edge_lmax={self.ictd_fix_edge_lmax}, lmax={self.lmax}."
            )
        self.num_interaction = int(num_interaction)
        self.max_radius = float(max_embed_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.ictd_fix_route = str(ictd_fix_route)
        self.ictd_fix_contraction_combine = str(ictd_fix_contraction_combine)
        self.ictd_fix_requested_product_backend = requested_product_backend
        self.ictd_fix_product_backend = (
            "ictd-bridge-u"
            # native ICTC U generation is the reliability limit on the contraction INPUT degree,
            # which is edge_lmax (the message), not just the hidden lmax; fall back past degree 3.
            if requested_product_backend == "ictd-pure-u" and max(self.lmax, self.ictd_fix_edge_lmax) > 3
            else requested_product_backend
        )
        self.ictd_fix_use_reduced_cg = bool(ictd_fix_use_reduced_cg)
        self.ictd_fix_first_layer_self_connection = bool(ictd_fix_first_layer_self_connection)
        self.ictd_fix_conv_tp_scale_init = str(ictd_fix_conv_tp_scale_init)
        self.ictd_fix_freeze_conv_tp_weight = bool(ictd_fix_freeze_conv_tp_weight)
        self.ictd_fix_interaction_init = str(ictd_fix_interaction_init)
        self.ictd_fix_product_backend_fallback = self.ictd_fix_product_backend != self.ictd_fix_requested_product_backend
        if self.ictd_fix_edge_lmax != self.lmax and self.ictd_fix_product_backend not in {"native-mace", "ictd-bridge-u", "cueq", "ictd-pure-u"}:
            raise NotImplementedError(
                "ictd_fix_edge_lmax != lmax is currently supported only for the exact MACE product "
                "backends 'native-mace', 'ictd-bridge-u', 'cueq', and the native 'ictd-pure-u' "
                "(which itself falls back to 'ictd-bridge-u' when max(lmax, edge_lmax) > 3)."
            )
        self.ictd_fix_interaction_scale = str(ictd_fix_interaction_scale)
        self.ictd_fix_fusion_scale_init = float(ictd_fix_fusion_scale_init)
        self.ictd_fix_fusion_heads = int(ictd_fix_fusion_heads)
        self.ictd_fix_fusion_head_weight_mode = str(ictd_fix_fusion_head_weight_mode)
        self.ictd_fix_fusion_input_scale_init = float(ictd_fix_fusion_input_scale_init)
        self.ictd_fix_fusion_input_scale_trainable = bool(ictd_fix_fusion_input_scale_trainable)
        self.ictd_fix_fusion_depth_attention = bool(ictd_fix_fusion_depth_attention)
        self.ictd_fix_gmix_gate_init = float(ictd_fix_gmix_gate_init)
        self.ictd_fix_gmix_gate_trainable = bool(ictd_fix_gmix_gate_trainable)
        self.ictd_fix_gmix_block_rmsnorm = bool(ictd_fix_gmix_block_rmsnorm)
        self.ictd_fix_gmix_block_rmsnorm_gamma_init = float(ictd_fix_gmix_block_rmsnorm_gamma_init)
        self.ictd_fix_readout_head_scale_init = float(ictd_fix_readout_head_scale_init)
        self.ictd_fix_readout_head_scale_trainable = bool(ictd_fix_readout_head_scale_trainable)
        self.ictd_fix_fusion_readout_mixed_channels = bool(ictd_fix_fusion_readout_mixed_channels)
        self.ictd_fix_fusion_pre_product_norm = bool(ictd_fix_fusion_pre_product_norm)
        self.ictd_fix_interaction_rms_norm = bool(ictd_fix_interaction_rms_norm)
        # radial_sqrt_num_basis=False -> byte-literal MACE radial (default for new models);
        # from_checkpoint forces True for back-compat with FSCETP checkpoints trained with the scale.
        self.radial_sqrt_num_basis = bool(radial_sqrt_num_basis)
        self.ictd_fix_interaction_attn_heads = int(ictd_fix_interaction_attn_heads)
        self.ictd_fix_interaction_attn_mode = str(
            ictd_fix_interaction_attn_mode
        )
        self.ictd_fix_interaction_attn_scope = str(
            ictd_fix_interaction_attn_scope
        )
        if self.ictd_fix_interaction_attn_mode not in {
            "legacy-softmax",
            "density-preserving",
        }:
            raise ValueError(
                "ictd_fix_interaction_attn_mode must be 'legacy-softmax' or "
                f"'density-preserving', got {self.ictd_fix_interaction_attn_mode!r}"
            )
        if self.ictd_fix_interaction_attn_scope not in {"all", "final"}:
            raise ValueError(
                "ictd_fix_interaction_attn_scope must be 'all' or 'final', "
                f"got {self.ictd_fix_interaction_attn_scope!r}"
            )
        self.ictd_fix_phase_mode = str(ictd_fix_phase_mode)
        self.ictd_fix_phase_hidden_channels = int(ictd_fix_phase_hidden_channels)
        self.ictd_fix_phase_residual_scale_init = float(ictd_fix_phase_residual_scale_init)
        self.ictd_fix_phase_amplitude = str(ictd_fix_phase_amplitude)
        self.ictd_fix_phase_coefficient = str(ictd_fix_phase_coefficient)
        self.ictd_fix_phase_context = str(ictd_fix_phase_context)
        self.ictd_fix_phase_placement = str(ictd_fix_phase_placement)
        self.ictd_fix_phase_density_rank = int(ictd_fix_phase_density_rank)
        self.ictd_fix_phase_density_species_mode = str(
            ictd_fix_phase_density_species_mode
        )
        self.ictd_fix_phase_density_species_embedding_dim = int(
            ictd_fix_phase_density_species_embedding_dim
        )
        self.ictd_fix_phase_density_species_rank = int(
            ictd_fix_phase_density_species_rank
        )
        self.ictd_fix_phase_density_pairs = str(ictd_fix_phase_density_pairs)
        self.ictd_fix_phase_coherence_init = float(ictd_fix_phase_coherence_init)
        self.ictd_fix_phase_normalization = str(ictd_fix_phase_normalization)
        self.ictd_fix_phase_scope = str(ictd_fix_phase_scope)
        self.ictd_fix_phase_heads = int(ictd_fix_phase_heads)
        if self.ictd_fix_phase_mode not in {
            "none",
            "final-scalar-residual",
            "final-full-l-residual",
        }:
            raise ValueError(
                "ictd_fix_phase_mode must be 'none', 'final-scalar-residual', or "
                "'final-full-l-residual', "
                f"got {self.ictd_fix_phase_mode!r}"
            )
        if self.ictd_fix_phase_hidden_channels <= 0:
            raise ValueError(
                "ictd_fix_phase_hidden_channels must be positive, "
                f"got {self.ictd_fix_phase_hidden_channels}"
            )
        if (
            self.ictd_fix_phase_heads <= 0
            or self.channels % self.ictd_fix_phase_heads != 0
        ):
            raise ValueError(
                "ictd_fix_phase_heads must be positive and divide channels, got "
                f"phase_heads={self.ictd_fix_phase_heads}, channels={self.channels}"
            )
        if self.ictd_fix_phase_amplitude not in {"unit", "softplus"}:
            raise ValueError(
                "ictd_fix_phase_amplitude must be 'unit' or 'softplus', "
                f"got {self.ictd_fix_phase_amplitude!r}"
            )
        if self.ictd_fix_phase_coefficient not in {
            "polar",
            "positive",
            "signed",
            "cartesian",
        }:
            raise ValueError(
                "ictd_fix_phase_coefficient must be 'polar', 'positive', 'signed', "
                f"or 'cartesian', got {self.ictd_fix_phase_coefficient!r}"
            )
        if self.ictd_fix_phase_context not in {
            "content",
            "radial",
            "irrep-norm",
            "content-irrep-norm",
        }:
            raise ValueError(
                "ictd_fix_phase_context must be 'content', 'radial', "
                "'irrep-norm', or 'content-irrep-norm', "
                f"got {self.ictd_fix_phase_context!r}"
            )
        if self.ictd_fix_phase_density_pairs not in {
            "full",
            "charge2",
            "diagonal",
            "full-gated",
            "full-adaptive",
            "full-adaptive-env",
            "full-balanced",
            "full-nonlinear",
            "full-nonlinear-readout",
        }:
            raise ValueError(
                "ictd_fix_phase_density_pairs must be 'full', 'charge2', 'diagonal', "
                "'full-gated', 'full-adaptive', 'full-adaptive-env', or "
                "'full-balanced', 'full-nonlinear', or "
                "'full-nonlinear-readout', "
                f"got {self.ictd_fix_phase_density_pairs!r}"
            )
        if not 0.0 < self.ictd_fix_phase_coherence_init < 1.0:
            raise ValueError(
                "ictd_fix_phase_coherence_init must lie strictly between zero and "
                f"one, got {self.ictd_fix_phase_coherence_init}"
            )
        if self.ictd_fix_phase_normalization not in {
            "avg-neighbors",
            "local-effective",
        }:
            raise ValueError(
                "ictd_fix_phase_normalization must be 'avg-neighbors' or "
                f"'local-effective', got {self.ictd_fix_phase_normalization!r}"
            )
        if self.ictd_fix_phase_density_rank <= 0:
            raise ValueError(
                "ictd_fix_phase_density_rank must be positive, "
                f"got {self.ictd_fix_phase_density_rank}"
            )
        if self.ictd_fix_phase_density_species_mode not in {
            "onehot-full",
            "embedded-lowrank",
        }:
            raise ValueError(
                "ictd_fix_phase_density_species_mode must be 'onehot-full' or "
                f"'embedded-lowrank', got "
                f"{self.ictd_fix_phase_density_species_mode!r}"
            )
        if self.ictd_fix_phase_density_species_embedding_dim <= 0:
            raise ValueError(
                "ictd_fix_phase_density_species_embedding_dim must be positive"
            )
        if self.ictd_fix_phase_density_species_rank <= 0:
            raise ValueError(
                "ictd_fix_phase_density_species_rank must be positive"
            )
        if (
            self.ictd_fix_phase_density_species_mode == "embedded-lowrank"
            and self.ictd_fix_phase_mode != "final-full-l-residual"
        ):
            raise ValueError(
                "embedded-lowrank phase-density species writeback requires "
                "phase_mode='final-full-l-residual'"
            )
        if self.ictd_fix_phase_scope not in {"final", "persistent"}:
            raise ValueError(
                "ictd_fix_phase_scope must be 'final' or 'persistent', "
                f"got {self.ictd_fix_phase_scope!r}"
            )
        if self.ictd_fix_phase_mode == "none" and self.ictd_fix_phase_scope != "final":
            raise ValueError("ictd_fix_phase_scope='persistent' requires an enabled phase mode")
        if self.ictd_fix_phase_scope == "persistent" and self.num_interaction < 2:
            raise ValueError(
                "ictd_fix_phase_scope='persistent' requires num_interaction >= 2"
            )
        if (
            self.ictd_fix_phase_density_pairs
            in {
                "full-gated",
                "full-adaptive",
                "full-adaptive-env",
                "full-balanced",
            }
            and self.ictd_fix_phase_scope != "final"
        ):
            raise ValueError(
                "edge-resolved phase density currently requires phase_scope='final'"
            )
        if (
            self.ictd_fix_phase_density_pairs
            in {
                "diagonal",
                "full-gated",
                "full-adaptive",
                "full-adaptive-env",
                "full-balanced",
            }
            and self.ictd_fix_phase_mode != "final-full-l-residual"
        ):
            raise ValueError(
                "edge-resolved phase density currently requires "
                "phase_mode='final-full-l-residual'"
            )
        if self.ictd_fix_phase_density_pairs == "charge2" and (
            self.ictd_fix_phase_scope != "final"
            or self.ictd_fix_phase_mode != "final-full-l-residual"
        ):
            raise ValueError(
                "charge2 quadratic density currently requires "
                "phase_mode='final-full-l-residual' and phase_scope='final'"
            )
        if self.ictd_fix_phase_placement not in {
            "post-product",
            "pre-product-l0",
            "pre-product-full-l",
            "pre-and-post",
        }:
            raise ValueError(
                "ictd_fix_phase_placement must be 'post-product', 'pre-product-l0', "
                "'pre-product-full-l', or 'pre-and-post', "
                f"got {self.ictd_fix_phase_placement!r}"
            )
        if (
            self.ictd_fix_phase_mode == "final-full-l-residual"
            and self.ictd_fix_phase_placement != "pre-product-full-l"
        ):
            raise ValueError(
                "final-full-l-residual requires ictd_fix_phase_placement="
                "'pre-product-full-l' so non-scalar irreps enter the symmetric contraction"
            )
        if (
            self.ictd_fix_phase_mode != "final-full-l-residual"
            and self.ictd_fix_phase_placement == "pre-product-full-l"
        ):
            raise ValueError(
                "pre-product-full-l requires ictd_fix_phase_mode='final-full-l-residual'"
            )
        if (
            self.ictd_fix_phase_scope == "persistent"
            and self.ictd_fix_phase_mode == "final-scalar-residual"
            and self.ictd_fix_phase_placement != "pre-product-l0"
        ):
            raise ValueError(
                "persistent scalar phase requires ictd_fix_phase_placement='pre-product-l0' "
                "so every layer can feed the neutral density into its product"
            )
        if (
            self.ictd_fix_phase_mode != "none"
            and self.ictd_fix_interaction_attn_heads > 0
            and self.ictd_fix_interaction_attn_mode != "density-preserving"
        ):
            raise ValueError(
                "phase mode can only be combined with density-preserving interaction "
                "attention; legacy-softmax changes the baseline density scale"
            )
        self.ictd_fix_gmix_energy_readout = bool(ictd_fix_gmix_energy_readout)
        self.ictd_fix_gmix_readout_output_init_std = float(ictd_fix_gmix_readout_output_init_std)
        self.ictd_fix_layer_readout_output_init_std = float(ictd_fix_layer_readout_output_init_std)
        self.ictd_fix_readout_hidden_channels = int(ictd_fix_readout_hidden_channels)
        self.ictd_fix_nonlinear_layer_readouts = bool(
            ictd_fix_nonlinear_layer_readouts
        )
        self.ictd_fix_final_layer_readout_only = bool(
            ictd_fix_final_layer_readout_only
        )
        self.ictd_fix_element_energy_correction = bool(
            ictd_fix_element_energy_correction
        )
        self.ictd_fix_scalar_ffn = bool(ictd_fix_scalar_ffn)
        if self.ictd_fix_readout_hidden_channels <= 0:
            raise ValueError(
                "ictd_fix_readout_hidden_channels must be positive, "
                f"got {self.ictd_fix_readout_hidden_channels}"
            )
        self.ictd_fix_gmix_readout_scale_init = (
            float(self.ictd_fix_readout_head_scale_init)
            if ictd_fix_gmix_readout_scale_init is None
            else float(ictd_fix_gmix_readout_scale_init)
        )
        # Fusion gmix output lmax: the gmix (multiple_contraction_mix) symmetric
        # contraction can emit a HIGHER output lmax than the backbone input lmax,
        # giving product5 extra higher-l angular invariants from the (already
        # message-passed) backbone features, at near-zero backbone cost. Default
        # = lmax => byte-identical to before.
        self.ictd_fix_gmix_output_lmax = (
            self.lmax if ictd_fix_gmix_output_lmax is None else int(ictd_fix_gmix_output_lmax)
        )
        if self.ictd_fix_gmix_output_lmax < self.lmax:
            raise ValueError(
                f"ictd_fix_gmix_output_lmax ({self.ictd_fix_gmix_output_lmax}) must be >= lmax ({self.lmax})"
            )
        self.polynomial_cutoff_p = (
            None
            if polynomial_cutoff_p is None or int(polynomial_cutoff_p) <= 0
            else int(polynomial_cutoff_p)
        )
        if self.ictd_fix_fusion_heads < 1:
            raise ValueError(f"ictd_fix_fusion_heads must be >= 1, got {self.ictd_fix_fusion_heads}")
        self.max_atomvalue = int(max_atomvalue)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.edge_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        if atomic_numbers is None:
            atomic_numbers = tuple(range(self.max_atomvalue))
        else:
            atomic_numbers = tuple(sorted({int(z) for z in atomic_numbers}))
            if len(atomic_numbers) == 0:
                raise ValueError("atomic_numbers must not be empty")
        self.atomic_numbers = atomic_numbers
        self.num_elements = len(self.atomic_numbers)
        if self.ictd_fix_element_energy_correction:
            self.element_energy_correction = nn.Parameter(
                torch.zeros(self.num_elements)
            )
        else:
            self.register_parameter("element_energy_correction", None)
        map_size = max(self.max_atomvalue, max(self.atomic_numbers) + 1)
        atomic_number_to_index = torch.full((map_size,), -1, dtype=torch.long)
        for idx, z in enumerate(self.atomic_numbers):
            if z < 0:
                raise ValueError(f"atomic_numbers must be non-negative, got {z}")
            atomic_number_to_index[z] = idx
        self.register_buffer("atomic_number_to_index", atomic_number_to_index, persistent=False)

        self.node_embedding = nn.Linear(self.num_elements, self.channels, bias=False)
        self.phase_density_species_embedding = (
            nn.Embedding(
                self.num_elements,
                self.ictd_fix_phase_density_species_embedding_dim,
            )
            if self.ictd_fix_phase_density_species_mode == "embedded-lowrank"
            else None
        )
        if self.phase_density_species_embedding is not None:
            nn.init.normal_(
                self.phase_density_species_embedding.weight,
                mean=0.0,
                std=1.0
                / math.sqrt(
                    float(self.ictd_fix_phase_density_species_embedding_dim)
                ),
            )
        product_target_lmax = [
            self.lmax if layer_idx < self.num_interaction - 1 else 0
            for layer_idx in range(self.num_interaction)
        ]
        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        self.phase_adapters = nn.ModuleDict()
        self.phase_direct_readouts = nn.ModuleDict()
        self.phase_direct_readout_scales = nn.ParameterDict()
        self.charged_updates = nn.ModuleDict()
        self.ictd_fix_effective_product_backends: list[str] = []
        for layer_idx, target_lmax in enumerate(product_target_lmax):
            attention_enabled = (
                self.ictd_fix_interaction_attn_heads > 0
                and (
                    self.ictd_fix_interaction_attn_scope == "all"
                    or layer_idx == self.num_interaction - 1
                )
            )
            phase_enabled = (
                self.ictd_fix_phase_mode != "none"
                and (
                    self.ictd_fix_phase_scope == "persistent"
                    or layer_idx == self.num_interaction - 1
                )
            )
            effective_product_backend = self.ictd_fix_product_backend
            self.ictd_fix_effective_product_backends.append(effective_product_backend)
            input_lmax = 0 if layer_idx == 0 else self.lmax
            first_layer_sc = self.ictd_fix_first_layer_self_connection and layer_idx == 0
            sc_lmax = 0 if first_layer_sc else target_lmax
            message_scale_init = None
            sc_scale_init = None
            if self.ictd_fix_interaction_scale == "mace-rms":
                # Initialized from the ICTC/native-MACE basisbridge diagnostic on
                # aspirin lmax=3/ch64. The scales are learnable, so this is a
                # stabilization prior rather than a fixed calibration.
                message_presets = {
                    0: [0.625, 0.561, 0.540, 0.403],
                    1: [0.489, 0.745, 0.741, 0.620],
                }
                preset = list(message_presets.get(layer_idx, [0.5] * (self.ictd_fix_edge_lmax + 1)))
                if len(preset) < self.ictd_fix_edge_lmax + 1:
                    preset = preset + [0.5] * (self.ictd_fix_edge_lmax + 1 - len(preset))
                message_scale_init = preset[: self.ictd_fix_edge_lmax + 1]
                if sc_lmax == 0 and layer_idx > 0:
                    sc_scale_init = [0.342]
                elif sc_lmax > 0 and layer_idx > 0:
                    sc_scale_init = [0.342] + [0.5] * sc_lmax
            self.interactions.append(
                ICTDResidualInteractionBlock(
                    channels=self.channels,
                    lmax=self.ictd_fix_edge_lmax,
                    input_lmax=input_lmax,
                    target_lmax=self.ictd_fix_edge_lmax,
                    sc_lmax=sc_lmax,
                    number_of_basis=self.number_of_basis,
                    num_elements=self.num_elements,
                    function_type=self.function_type,
                    ictd_save_tp_mode=ictd_save_tp_mode,
                    ictd_tp_path_policy=ictd_tp_path_policy,
                    ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                    ictd_tp_backend=ictd_tp_backend,
                    equivariant_post_linear=equivariant_post_linear,
                    use_self_connection=(layer_idx > 0) or first_layer_sc,
                    avg_num_neighbors=self.avg_num_neighbors,
                    message_scale_init=message_scale_init,
                    sc_scale_init=sc_scale_init,
                    conv_tp_scale_init=self.ictd_fix_conv_tp_scale_init,
                    freeze_conv_tp_weight=self.ictd_fix_freeze_conv_tp_weight,
                    interaction_init=self.ictd_fix_interaction_init,
                    use_rms_norm=self.ictd_fix_interaction_rms_norm,
                    interaction_attn_heads=(
                        self.ictd_fix_interaction_attn_heads
                        if attention_enabled
                        else 0
                    ),
                    interaction_attn_mode=self.ictd_fix_interaction_attn_mode,
                    phase_enabled=phase_enabled,
                    phase_hidden_channels=self.ictd_fix_phase_hidden_channels,
                    phase_amplitude=self.ictd_fix_phase_amplitude,
                    phase_coefficient=self.ictd_fix_phase_coefficient,
                    phase_context=self.ictd_fix_phase_context,
                    phase_normalization=self.ictd_fix_phase_normalization,
                    phase_angular_channels=(
                        self.ictd_fix_phase_density_pairs
                        in {"full-nonlinear", "full-nonlinear-readout"}
                    ),
                    phase_heads=self.ictd_fix_phase_heads,
                )
            )
            if effective_product_backend == "native-mace":
                self.products.append(
                    NativeMACEProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                        use_reduced_cg=self.ictd_fix_use_reduced_cg,
                    )
                )
            elif effective_product_backend == "ictd-bridge-u":
                self.products.append(
                    ICTDBridgeUProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                        use_reduced_cg=self.ictd_fix_use_reduced_cg,
                    )
                )
            elif effective_product_backend == "cueq":
                self.products.append(
                    CueqMACEProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                        use_reduced_cg=self.ictd_fix_use_reduced_cg,
                    )
                )
            elif effective_product_backend == "ictd-pure-u":
                self.products.append(
                    ICTDPureUProductBasisBlockSO3(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        # the interaction emits a full-SO(3) message up to edge_lmax; the contraction
                        # consumes it and truncates to target_lmax (== self.lmax). Mirror the exact
                        # MACE backends (native-mace/bridge-u/cueq) and pass edge_lmax as the input
                        # degree, not self.lmax, so edge_lmax != lmax (max_ell) is handled natively.
                        lmax=self.ictd_fix_edge_lmax,
                        target_lmax=target_lmax,
                        correlation=save_contraction_order,
                    )
                )
            elif target_lmax == self.lmax:
                self.products.append(
                    ICTDProductBasisBlock(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        ictd_tp_path_policy=ictd_tp_path_policy,
                        ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                        internal_compute_dtype=internal_compute_dtype,
                        ictd_tp_backend=ictd_tp_backend,
                        contraction_combine=self.ictd_fix_contraction_combine,
                    )
                )
            else:
                self.products.append(
                    ICTDScalarProductBasisBlock(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        ictd_tp_path_policy=ictd_tp_path_policy,
                        ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                        internal_compute_dtype=internal_compute_dtype,
                        ictd_tp_backend=ictd_tp_backend,
                        contraction_combine=self.ictd_fix_contraction_combine,
                    )
                )
            if phase_enabled:
                if (
                    self.ictd_fix_phase_scope == "persistent"
                    and self.ictd_fix_phase_density_pairs != "diagonal"
                    and layer_idx > 0
                ):
                    self.charged_updates[str(layer_idx)] = PersistentChargedUpdate(
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                    )
                if self.ictd_fix_phase_mode == "final-full-l-residual":
                    self.phase_adapters[str(layer_idx)] = PhaseHermitianFullLResidual(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                        density_rank=self.ictd_fix_phase_density_rank,
                        residual_scale_init=self.ictd_fix_phase_residual_scale_init,
                        coherence_gate=(
                            self.ictd_fix_phase_density_pairs == "full-gated"
                        ),
                        adaptive_coherence=(
                            self.ictd_fix_phase_density_pairs == "full-adaptive"
                        ),
                        environment_adaptive_coherence=(
                            self.ictd_fix_phase_density_pairs
                            == "full-adaptive-env"
                        ),
                        adaptive_coherence_init=self.ictd_fix_phase_coherence_init,
                        quadratic_form=(
                            "charge2"
                            if self.ictd_fix_phase_density_pairs == "charge2"
                            else "hermitian"
                        ),
                        output_gate=(
                            self.ictd_fix_phase_density_pairs
                            in {"full-nonlinear", "full-nonlinear-readout"}
                        ),
                        species_mode=self.ictd_fix_phase_density_species_mode,
                        species_embedding_dim=(
                            self.ictd_fix_phase_density_species_embedding_dim
                        ),
                        species_rank=self.ictd_fix_phase_density_species_rank,
                    )
                    if (
                        self.ictd_fix_phase_density_pairs
                        == "full-nonlinear-readout"
                    ):
                        self.phase_direct_readouts[str(layer_idx)] = (
                            MACEStyleScalarReadoutSO3(
                                self.channels,
                                hidden_channels=self.ictd_fix_readout_hidden_channels,
                            )
                        )
                        self.phase_direct_readout_scales[str(layer_idx)] = (
                            nn.Parameter(
                                torch.tensor(
                                    0.1,
                                    dtype=torch.get_default_dtype(),
                                )
                            )
                        )
                else:
                    self.phase_adapters[str(layer_idx)] = PhaseHermitianScalarResidual(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.ictd_fix_edge_lmax,
                        residual_scale_init=self.ictd_fix_phase_residual_scale_init,
                        internal_compute_dtype=internal_compute_dtype,
                    )
        if self.ictd_fix_final_layer_readout_only:
            self.layer_energy_readouts = nn.ModuleList()
        elif self.ictd_fix_nonlinear_layer_readouts:
            self.layer_energy_readouts = nn.ModuleList(
                [
                    MACEStyleScalarReadoutSO3(
                        self.channels,
                        hidden_channels=self.ictd_fix_readout_hidden_channels,
                        output_init_std=self.ictd_fix_layer_readout_output_init_std,
                    )
                    for _ in range(self.num_interaction - 1)
                ]
            )
        else:
            self.layer_energy_readouts = nn.ModuleList(
                [
                    EquivariantScalarReadoutSO3(
                        self.channels,
                        self.lmax,
                        output_init_std=self.ictd_fix_layer_readout_output_init_std,
                    )
                    for _ in range(self.num_interaction - 1)
                ]
            )
        self.last_layer_energy_readout = MACEStyleScalarReadoutSO3(
            self.channels,
            hidden_channels=self.ictd_fix_readout_hidden_channels,
            output_init_std=self.ictd_fix_layer_readout_output_init_std,
        )
        self.scalar_ffns = (
            nn.ModuleList(
                [
                    InvariantScalarResidualFFN(self.channels)
                    for _ in range(self.num_interaction)
                ]
            )
            if self.ictd_fix_scalar_ffn
            else None
        )
        if self.ictd_fix_readout_head_scale_trainable:
            self.readout_head_scales = nn.Parameter(
                torch.full((2,), self.ictd_fix_readout_head_scale_init, dtype=torch.get_default_dtype())
            )
        else:
            self.readout_head_scales = None

        # Fusion route removed in this baseline-only build -> these submodules are always absent.
        self.save_multiple_mix_channels = None
        self.multiple_contraction_mix = None
        self.multiple_contract_fuse = None
        self.ictd_fix_fusion_mix_backend = None
        self.fusion_readouts = None
        self.fusion_readout = None
        self.fusion_head_logits = None
        self.fusion_head_weights = None
        self.fusion_energy_scale = None
        self.fusion_input_scales = None
        self.fusion_depth_attention = None
        self.g_mix_gate = None
        self.gmix_block_rmsnorm_gamma = None
        self.gmix_energy_readout = None
        self.gmix_readout_head_scale = None

        # --- long-range interaction module (None when mode=="none"; no-op when off,
        # so the flagship's numerics + checkpoints are unchanged with long_range off).
        # Fed the final per-atom SCALAR descriptor (scalar_last for fusion /
        # layer_states[-1] for baseline, both invariant -> equivariance-safe);
        # energy_scale inits to 0 -> zero contribution at init.
        self.long_range_mode = str(long_range_mode)
        self.long_range_reciprocal_backend = str(long_range_reciprocal_backend)
        self.long_range_assignment = str(long_range_assignment)
        self.long_range_backend = str(long_range_backend)
        self.long_range_energy_partition = str(long_range_energy_partition)
        self.long_range_green_mode = str(long_range_green_mode)
        self.long_range_boundary = str(long_range_boundary)
        self.long_range_neutralize = bool(long_range_neutralize)
        self.long_range_mesh_fft_full_ewald = bool(long_range_mesh_fft_full_ewald)
        self.long_range_mesh_size = int(long_range_mesh_size)
        self.long_range_slab_padding_factor = int(long_range_slab_padding_factor)
        self.long_range_module = build_long_range_module(
            mode=self.long_range_mode,
            feature_dim=self.channels,
            hidden_dim=long_range_hidden_dim,
            boundary=long_range_boundary,
            neutralize=long_range_neutralize,
            filter_hidden_dim=long_range_filter_hidden_dim,
            kmax=long_range_kmax,
            mesh_size=long_range_mesh_size,
            slab_padding_factor=long_range_slab_padding_factor,
            include_k0=long_range_include_k0,
            source_channels=long_range_source_channels,
            backend=long_range_backend,
            reciprocal_backend=long_range_reciprocal_backend,
            energy_partition=long_range_energy_partition,
            green_mode=long_range_green_mode,
            assignment=long_range_assignment,
            mesh_fft_full_ewald=long_range_mesh_fft_full_ewald,
            mesh_fft_reciprocal_only=long_range_mesh_fft_reciprocal_only,
            max_multipole_l=long_range_max_multipole_l,
            multipole_feature_channels=self.channels,
            theta=long_range_theta,
            leaf_size=long_range_leaf_size,
            multipole_order=long_range_multipole_order,
            far_source_dim=long_range_far_source_dim,
            far_num_shells=long_range_far_num_shells,
            far_shell_growth=long_range_far_shell_growth,
            far_tail=long_range_far_tail,
            far_tail_bins=long_range_far_tail_bins,
            far_stats=long_range_far_stats,
            far_max_radius_multiplier=long_range_far_max_radius_multiplier,
            far_source_norm=long_range_far_source_norm,
            far_gate_init=long_range_far_gate_init,
            cutoff_radius=self.max_radius,
        )
        self.long_range_exports_reciprocal_source = (
            bool(getattr(self.long_range_module, "exports_reciprocal_source", False))
            if self.long_range_module is not None
            else False
        )

        # Multipole long-range: tap the deepest full-SO(3) node features for equivariant
        # Cartesian monopole/dipole/quadrupole sources (a degree-l carrier IS a rank-l
        # multipole) and feed the mesh-FFT reciprocal kernel. OFF (max_multipole_l==0) ->
        # readout is None and the scalar latent-source path is used unchanged (byte-identical).
        self.long_range_max_multipole_l = int(long_range_max_multipole_l)
        self.long_range_multipole_gate_init = float(long_range_multipole_gate_init)
        self.multipole_readout = None
        if self.long_range_module is not None and self.long_range_max_multipole_l > 0:
            if str(long_range_reciprocal_backend) != "mesh_fft":
                raise ValueError(
                    "long_range_max_multipole_l>0 requires long_range_reciprocal_backend='mesh_fft' "
                    "(only the mesh-FFT kernel exposes multipole_energy)"
                )
            if not bool(long_range_mesh_fft_full_ewald):
                raise ValueError(
                    "long_range_max_multipole_l>0 requires long_range_mesh_fft_full_ewald=True: the "
                    "reciprocal multipole sum needs Ewald Gaussian screening for accuracy (without it "
                    "the in-cell translation error is tens of %); long_range_assignment='pcs' recommended"
                )
            if self.long_range_max_multipole_l > self.lmax:
                raise ValueError(
                    f"long_range_max_multipole_l={self.long_range_max_multipole_l} exceeds model lmax={self.lmax}"
                )
            from chorus.models.multipole_readout import MultipoleReadout

            self.multipole_readout = MultipoleReadout(
                channels=self.channels,
                lmax=self.lmax,
                max_multipole_l=self.long_range_max_multipole_l,
                source_channels=int(long_range_source_channels),
                source_scale_init=self.long_range_multipole_gate_init,
            )
            if getattr(self.long_range_module, "energy_scale", None) is not None:
                with torch.no_grad():
                    self.long_range_module.energy_scale.fill_(1.0)
            # at export the multipole route emits a packed [q|mu|Q] reciprocal_source for the
            # C++ solver (instead of computing the reciprocal energy in-model).
            self.long_range_exports_reciprocal_source = True
            # Deploy metadata read by export_libtorch_core -> .json -> the C++ engine/solver.
            # The packed per-atom source is [q | dipole_xyz | quad_3x3] per channel (channel-major);
            # the C++ reciprocal solver rebuilds q/mu/Q from source_channels + max_multipole_l, and
            # multipole_reciprocal_energy mirrors the in-model multipole_energy (screened |S(k)|^2 PME).
            self.long_range_runtime_backend = "mesh_fft"
            self.long_range_runtime_source_kind = "latent_multipole"
            self.long_range_runtime_source_channels = int(long_range_source_channels)
            self.long_range_runtime_source_layout = "packed_q_dipole_quad"
            self.long_range_runtime_source_boundary = "periodic"
            # Deploy config the .json writer reads, taken from the built mesh kernel (the source of
            # truth) so the C++ reciprocal solver reproduces the in-model multipole_energy exactly:
            # screened |S(k)|^2 PME with green_mode/mesh_size/boundary/full_ewald all matched.
            _mp_kernel = self.long_range_module.kernel
            self.long_range_mesh_fft_full_ewald = bool(getattr(_mp_kernel, "full_ewald", bool(long_range_mesh_fft_full_ewald)))
            self.long_range_mesh_size = int(getattr(_mp_kernel, "mesh_size", 16))
            self.long_range_green_mode = str(getattr(_mp_kernel, "green_mode", "poisson"))
            self.long_range_boundary = str(getattr(_mp_kernel, "boundary", "periodic"))
            # multipole_energy always distributes e_graph/n_local uniformly (it ignores the kernel's
            # energy_partition); report that honestly so the C++ per-atom decomposition can match.
            self.long_range_energy_partition = "uniform"
            self.long_range_neutralize = bool(getattr(self.long_range_module, "neutralize", True))

        # Long-range dispersion term. OFF by default -> None -> no contribution
        # (byte-identical). The legacy boolean `long_range_dispersion=True` maps to
        # `pairwise-c6`; future MBD modes should plug into this same interface rather
        # than adding another hand-written branch in forward.
        self.long_range_dispersion_mode = normalize_dispersion_mode(
            long_range_dispersion=bool(long_range_dispersion),
            long_range_dispersion_mode=long_range_dispersion_mode,
        )
        self.long_range_dispersion = self.long_range_dispersion_mode != "none"
        self.dispersion_cutoff = float(dispersion_cutoff)
        if dispersion_max_num_neighbors is not None and int(dispersion_max_num_neighbors) < 0:
            raise ValueError("dispersion_max_num_neighbors must be >= 0 or None")
        self.dispersion_max_num_neighbors = (
            None if dispersion_max_num_neighbors is None or int(dispersion_max_num_neighbors) == 0
            else int(dispersion_max_num_neighbors)
        )
        self.dispersion_neighbor_method = str(dispersion_neighbor_method)
        self.dispersion_bruteforce_threshold = int(dispersion_bruteforce_threshold)
        self.dispersion_allow_large_bruteforce_fallback = bool(dispersion_allow_large_bruteforce_fallback)
        self.dispersion_slq_num_probes = int(dispersion_slq_num_probes)
        self.dispersion_slq_lanczos_steps = int(dispersion_slq_lanczos_steps)
        self.mbd_operator_backend = str(mbd_operator_backend)
        self.mbd_pme_mesh_size = int(mbd_pme_mesh_size)
        self.mbd_pme_assignment = str(mbd_pme_assignment)
        self.mbd_pme_k_norm_floor = float(mbd_pme_k_norm_floor)
        self.mbd_pme_assignment_window_floor = float(mbd_pme_assignment_window_floor)
        self.mbd_pme_ewald_alpha_prefactor = float(mbd_pme_ewald_alpha_prefactor)
        self.mbd_anisotropic_polarizability = bool(mbd_anisotropic_polarizability)
        self.mbd_learnable_energy_scale = bool(mbd_learnable_energy_scale)
        self.mbd_alpha_floor = float(mbd_alpha_floor)
        self.dispersion_min_cutoff = float(dispersion_min_cutoff)
        self.dispersion_switch_width = float(dispersion_switch_width)
        self.dispersion_pbc = str(long_range_boundary) == "periodic"
        self.dispersion = build_long_range_dispersion(
            mode=self.long_range_dispersion_mode,
            feature_dim=self.channels,
            cutoff=self.dispersion_cutoff,
            pbc=self.dispersion_pbc,
            neighbor_method=self.dispersion_neighbor_method,
            bruteforce_threshold=self.dispersion_bruteforce_threshold,
            allow_large_bruteforce_fallback=self.dispersion_allow_large_bruteforce_fallback,
            slq_num_probes=self.dispersion_slq_num_probes,
            slq_lanczos_steps=self.dispersion_slq_lanczos_steps,
            max_num_neighbors=self.dispersion_max_num_neighbors,
            mbd_operator_backend=self.mbd_operator_backend,
            mbd_pme_mesh_size=self.mbd_pme_mesh_size,
            mbd_pme_assignment=self.mbd_pme_assignment,
            mbd_pme_k_norm_floor=self.mbd_pme_k_norm_floor,
            mbd_pme_assignment_window_floor=self.mbd_pme_assignment_window_floor,
            mbd_pme_ewald_alpha_prefactor=self.mbd_pme_ewald_alpha_prefactor,
            mbd_anisotropic_polarizability=self.mbd_anisotropic_polarizability,
            mbd_learnable_energy_scale=self.mbd_learnable_energy_scale,
            mbd_alpha_floor=self.mbd_alpha_floor,
            dispersion_min_cutoff=self.dispersion_min_cutoff,
            dispersion_switch_width=self.dispersion_switch_width,
        )

        # MBD-source packing metadata (read by the exporter -> .json -> C++ engine/pair-style). When the
        # mbd-slq head exports a source, the deploy forward emits [electrostatic | omega, alpha]; the MBD
        # (omega,alpha) begins at offset = the electrostatic source width (0 if no electrostatic source).
        self.long_range_mbd_source_enabled = bool(
            self.dispersion is not None and self.dispersion.exports_mbd_source()
        )
        # MBD source width: isotropic [omega, alpha] = 2; anisotropic [omega, alpha_iso, 6*B] = 8.
        self.long_range_mbd_source_channels = 8 if self.mbd_anisotropic_polarizability else 2
        _l = self.long_range_max_multipole_l
        self.long_range_mbd_source_offset = (
            int(getattr(self, "long_range_runtime_source_channels", 0))
            * (1 + (3 if _l >= 1 else 0) + (9 if _l >= 2 else 0))
            if self.long_range_exports_reciprocal_source
            else 0
        )
        if self.long_range_mbd_source_enabled:
            # emit the reciprocal_source even for an MBD-only model (no electrostatics) so the C++ MBD
            # solver gets (omega, alpha); the reciprocal solver stays gated on the electrostatic channels.
            self.long_range_exports_reciprocal_source = True

        # Optional fixed scale/shift on the network (short-range) per-atom interaction energy.
        # This mirrors MACE ScaleShiftMACE: E_inter_atom = scale * readout + shift. OFF by
        # default -> None buffers are excluded from state_dict, so old checkpoints load unchanged
        # with strict=True. E0 is added afterward (outside the model) and is NOT scaled/shifted.
        self.energy_output_scale_enabled = bool(energy_output_scale_enabled)
        if self.energy_output_scale_enabled:
            # Store at full (float64) precision; cast to the compute dtype at use in forward.
            # (Storing in the default float32 would round the scale and perturb energies ~1e-8.)
            self.register_buffer(
                "energy_output_scale",
                torch.tensor(float(energy_output_scale), dtype=torch.float64),
            )
        else:
            self.register_buffer("energy_output_scale", None)
        self.energy_output_shift_enabled = bool(energy_output_shift_enabled)
        if self.energy_output_shift_enabled:
            self.register_buffer(
                "energy_output_shift",
                torch.tensor(float(energy_output_shift), dtype=torch.float64),
            )
        else:
            self.register_buffer("energy_output_shift", None)
        # Optional converter-only additive term for MACE's first interaction skip connection.
        # It is None for normal training/checkpoints. `convert_mace_to_ictd` installs a
        # tensor of shape (num_elements, channels) so the converted model can reproduce
        # mace-torch without relying on Python forward hooks, which are brittle under export.
        self.register_buffer("mace_first_layer_sc0", None)

    def _readout_head_scale(self, index: int, ref: torch.Tensor) -> torch.Tensor:
        if self.readout_head_scales is None:
            # new_zeros(()) is a device memset (no host->device copy) so this stays
            # CUDA-graph capturable; +scalar is a kernel arg. Equals the scalar.
            return ref.new_zeros(()) + float(self.ictd_fix_readout_head_scale_init)
        return self.readout_head_scales[index].to(dtype=ref.dtype, device=ref.device)

    def _fusion_input_scale(self, index: int, ref: torch.Tensor) -> torch.Tensor:
        if self.fusion_input_scales is None:
            return ref.new_zeros(()) + float(self.ictd_fix_fusion_input_scale_init)
        return self.fusion_input_scales[index].to(dtype=ref.dtype, device=ref.device)

    def _g_mix_gate(self, ref: torch.Tensor) -> torch.Tensor:
        if self.g_mix_gate is None:
            return ref.new_zeros(()) + float(self.ictd_fix_gmix_gate_init)
        return self.g_mix_gate.to(dtype=ref.dtype, device=ref.device)

    def _maybe_gmix_block_rmsnorm(self, g_mix: torch.Tensor) -> torch.Tensor:
        if self.gmix_block_rmsnorm_gamma is None:
            return g_mix
        channels = self.save_multiple_mix_channels if self.ictd_fix_fusion_readout_mixed_channels else self.channels
        return _so3_block_rmsnorm(
            g_mix,
            int(channels),
            self.ictd_fix_gmix_output_lmax,
            self.gmix_block_rmsnorm_gamma,
        )

    def install_mace_first_layer_sc0(self, sc0_by_element: torch.Tensor | None) -> None:
        """Install the first-layer element skip term used by converted mace-torch models.

        MACE's first residual block adds a pure l=0, element-conditioned self connection.
        This baseline ICTC block has no parameter slot for that exact additive constant, so
        conversion stores it as a non-trainable buffer and `forward` adds it after product[0].
        """
        if sc0_by_element is None:
            self.mace_first_layer_sc0 = None
            return
        sc0 = sc0_by_element.detach().clone()
        if sc0.shape != (self.num_elements, self.channels):
            raise ValueError(
                f"mace_first_layer_sc0 must have shape ({self.num_elements}, {self.channels}), "
                f"got {tuple(sc0.shape)}"
            )
        self.mace_first_layer_sc0 = sc0

    def _apply_e3nn_basis_fold(self) -> None:
        """Fold the FIXED angular operators (interaction Clebsch-Gordan tensors + the
        symmetric-contraction U tensors) into the e3nn/MACE spherical basis so the model
        computes its l>=1 features natively in the e3nn convention (``angular_basis="e3nn"``).

        Combined with the harmonics fold in ``forward``, this is a single global orthogonal
        change of the angular basis: every intermediate equivariant feature becomes its e3nn
        counterpart, while the energy / forces / virial (SO(3) invariants) are unchanged.
        Learnable weights are NOT touched (they index channel / path axes that the fold
        preserves), so an ``e3nn`` model is the SAME function as its ``ictd`` twin in a rotated
        basis -> bit-identical output. Idempotent; runs once (lazily on first forward)."""
        if getattr(self, "_e3nn_folded", False):
            return
        from chorus.mace_basis import orthogonal_Q_blocks
        ref = next(self.parameters())
        q_blocks_cpu = orthogonal_Q_blocks(
            max(self.lmax, self.ictd_fix_edge_lmax),
            dtype=torch.float64,
            device="cpu",
        )
        self._e3nn_q_blocks = [q.to(dtype=ref.dtype, device=ref.device) for q in q_blocks_cpu]
        folded_u = False
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "fold_cg_to_e3nn"):
                module.fold_cg_to_e3nn(q_blocks_cpu)
            if hasattr(module, "enable_e3nn_basis"):
                module.enable_e3nn_basis(self._e3nn_q_blocks)
                folded_u = True
        if not folded_u:
            raise NotImplementedError(
                "angular_basis='e3nn' currently requires the symmetric-contraction backend to "
                "expose enable_e3nn_basis (currently product_backend='ictd-pure-u' or 'cueq'). The "
                f"selected backend {getattr(self, 'ictd_fix_product_backend', '?')!r} has no e3nn fold.")
        self._e3nn_folded = True

    def activate_e3nn_basis_from_folded_state_dict(self) -> None:
        """Restore runtime e3nn-basis state after loading already-folded buffers.

        Training with ``angular_basis='e3nn'`` folds fixed interaction tensors on the
        first forward. Those folded tensors are saved in ``state_dict``. Reload must
        not fold them again, but forward still needs the Q blocks for edge harmonics
        and product backends such as cuEq still need their e3nn-basis runtime flag.
        """
        if self.angular_basis != "e3nn":
            raise ValueError("activate_e3nn_basis_from_folded_state_dict requires angular_basis='e3nn'")
        if getattr(self, "_e3nn_folded", False):
            return
        from chorus.mace_basis import orthogonal_Q_blocks
        ref = next(self.parameters())
        q_blocks_cpu = orthogonal_Q_blocks(
            max(self.lmax, self.ictd_fix_edge_lmax),
            dtype=torch.float64,
            device="cpu",
        )
        self._e3nn_q_blocks = [q.to(dtype=ref.dtype, device=ref.device) for q in q_blocks_cpu]
        enabled = False
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "enable_e3nn_basis"):
                module.enable_e3nn_basis(self._e3nn_q_blocks)
                enabled = True
        if not enabled:
            raise NotImplementedError(
                "Loaded angular_basis='e3nn' checkpoint requires a product backend with enable_e3nn_basis."
            )
        self._e3nn_folded = True

    def to_mace_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Re-express an equivariant feature tensor in the *original-MACE / e3nn* basis.

        ``x`` carries the ``(lmax+1)**2`` angular components in its last axis (ICTC basis); the
        result is ``x @ Q`` with the fixed block-diagonal orthogonal ``Q`` (see
        :mod:`chorus.mace_basis`). Energy, forces and the virial are SO(3) invariants / physical
        tensors and are basis-independent, so this only matters for equivariant (l>=1) features.
        """
        from chorus.mace_basis import to_mace_basis as _to_mace
        return _to_mace(x, self.lmax)

    def to_ictd_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`to_mace_basis` (original-MACE/e3nn -> ICTC)."""
        from chorus.mace_basis import to_ictd_basis as _to_ictd
        return _to_ictd(x, self.lmax)

    def forward(
        self,
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        *,
        precomputed_edge_vec=None,
        dispersion_edge_src=None,
        dispersion_edge_dst=None,
        dispersion_edge_shifts=None,
        precomputed_dispersion_edge_vec=None,
        return_combined_features: bool = False,
        sync_after_scatter: callable | None = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
        if return_physical_tensors:
            raise ValueError("pure-cartesian-ictd-fix does not currently support return_physical_tensors=True")

        dtype = next(self.parameters()).dtype
        if self.angular_basis == "e3nn" and not self._e3nn_folded:
            self._apply_e3nn_basis_fold()
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        if not getattr(self, "preserve_edge_order", False):
            sort_idx = torch.argsort(edge_dst)
            edge_src = edge_src[sort_idx]
            edge_dst = edge_dst[sort_idx]
            edge_shifts = edge_shifts[sort_idx]
        else:
            sort_idx = None
        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec if sort_idx is None else precomputed_edge_vec[sort_idx]
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs

        edge_length = edge_vec.norm(dim=1)
        n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        edge_mask = (
            None
            if getattr(self, "assume_edges_within_radius", False)
            else (edge_length <= self.max_radius).to(dtype=pos.dtype).unsqueeze(-1)
        )
        Y_list = direction_harmonics_all(n.to(dtype=dtype), self.ictd_fix_edge_lmax)
        if self.angular_basis == "e3nn":
            # fold the angular embedding into the e3nn/MACE spherical basis (Y_ictd @ Q_l = Y_e3nn)
            Y_list = [Y_list[l] @ self._e3nn_q_blocks[l].to(dtype=Y_list[l].dtype, device=Y_list[l].device)
                      for l in range(self.ictd_fix_edge_lmax + 1)]
        edge_attrs = {l: Y_list[l].to(dtype=dtype).unsqueeze(-2) for l in range(self.ictd_fix_edge_lmax + 1)}
        edge_feats = mace_radial_embedding(
            edge_length,
            r_max=self.max_radius,
            number_of_basis=self.number_of_basis,
            function_type=self.function_type,
            polynomial_cutoff_p=self.polynomial_cutoff_p,
            sqrt_num_basis_norm=self.radial_sqrt_num_basis,
        ).to(dtype=dtype)
        # Per-edge cutoff envelope for optional neighbor attention and the smooth
        # effective-coordination normalization of the charged stream.  It is the
        # same MACE polynomial cutoff already baked into edge_feats.
        edge_env = (
            mace_polynomial_cutoff(
                edge_length,
                self.max_radius,
                self.polynomial_cutoff_p if self.polynomial_cutoff_p is not None else 6,
            ).to(dtype=dtype)
            if (
                self.ictd_fix_interaction_attn_heads > 0
                or (
                    self.ictd_fix_phase_mode != "none"
                    and self.ictd_fix_phase_normalization == "local-effective"
                )
                or self.ictd_fix_phase_density_pairs == "full-balanced"
            )
            else None
        )

        A_long = A.long()
        # `skip_input_validation` removes the two host syncs below (`.item()` /
        # `torch.any` + `.tolist()`) so this forward can be captured by a CUDA
        # graph. It only disables guards; the numerics (compact_idx, one_hot) are
        # unchanged. The capture wrapper validates inputs once before enabling it.
        if not getattr(self, "skip_input_validation", False):
            if int(A_long.max().item()) >= self.atomic_number_to_index.numel():
                raise ValueError(
                    f"Encountered atomic number {int(A_long.max().item())}, but compact mapping supports only up to "
                    f"{self.atomic_number_to_index.numel() - 1}. atomic_numbers={self.atomic_numbers}"
                )
        compact_idx = self.atomic_number_to_index[A_long]
        if not getattr(self, "skip_input_validation", False):
            if torch.any(compact_idx < 0):
                bad = torch.unique(A_long[compact_idx < 0]).tolist()
                raise ValueError(
                    f"Encountered atomic numbers without compact mapping: {bad}. "
                    f"Configured atomic_numbers={self.atomic_numbers}"
                )
        type_idx_only_eval = (
            (not self.training)
            and all(type(product).__name__.startswith("Cueq") for product in self.products)
            and all(
                interaction.self_connection is not None
                or getattr(interaction, "_fused_selector_message_enabled", False)
                for interaction in self.interactions
            )
        )
        if type_idx_only_eval:
            node_attrs = None
            embedding_weight = self.node_embedding.weight.t().to(dtype=dtype, device=pos.device)
            h = embedding_weight.index_select(0, compact_idx)
        else:
            node_attrs = F.one_hot(compact_idx, num_classes=self.num_elements).to(dtype=dtype)
            h = self.node_embedding(node_attrs)
        phase_density_species_embedding = (
            self.phase_density_species_embedding(compact_idx)
            if self.phase_density_species_embedding is not None
            else None
        )

        layer_states: List[torch.Tensor] = []
        last_preproduct_state: torch.Tensor | None = None
        charged_doublet: torch.Tensor | None = None
        total_energy = None
        for layer_idx, (interaction, product) in enumerate(zip(self.interactions, self.products)):
            interaction_out = interaction(
                node_attrs=node_attrs,
                node_feats=h,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                edge_mask=edge_mask,
                edge_env=edge_env,
                node_type_idx=compact_idx,
                sync_after_scatter=sync_after_scatter,
                return_phase_doublet=interaction.phase_enabled,
                return_phase_edges=(
                    interaction.phase_enabled
                    and self.ictd_fix_phase_density_pairs
                    in {
                        "diagonal",
                        "full-gated",
                        "full-adaptive",
                        "full-adaptive-env",
                        "full-balanced",
                    }
                ),
            )
            phase_doublet = None
            phase_edge_orbital = None
            phase_edge_norm_sq = None
            if interaction.phase_enabled:
                if self.ictd_fix_phase_density_pairs in {
                        "diagonal",
                        "full-gated",
                        "full-adaptive",
                        "full-adaptive-env",
                        "full-balanced",
                }:
                    (
                        message,
                        sc,
                        phase_doublet,
                        phase_edge_orbital,
                        phase_edge_norm_sq,
                    ) = interaction_out
                else:
                    message, sc, phase_doublet = interaction_out
            else:
                message, sc = interaction_out
            phase_key = str(layer_idx)
            phase_delta = None
            phase_direct_energy = None
            if phase_key in self.phase_adapters:
                if phase_doublet is None:
                    raise RuntimeError(
                        f"phase adapter exists for layer {layer_idx}, but the interaction returned no doublet"
                    )
                if (
                    self.ictd_fix_phase_scope == "persistent"
                    and self.ictd_fix_phase_density_pairs != "diagonal"
                ):
                    if charged_doublet is None:
                        if layer_idx != 0:
                            raise RuntimeError(
                                "persistent charged stream was not initialized by the first layer"
                            )
                        charged_doublet = phase_doublet
                    else:
                        if phase_key not in self.charged_updates:
                            raise RuntimeError(
                                f"persistent charged update is missing for layer {layer_idx}"
                            )
                        charged_doublet = self.charged_updates[
                            phase_key
                        ].forward_doublet(
                            charged_doublet,
                            phase_doublet,
                        )
                    phase_doublet = charged_doublet
                if self.ictd_fix_phase_density_pairs == "diagonal":
                    if phase_edge_orbital is None or phase_edge_norm_sq is None:
                        raise RuntimeError(
                            "diagonal phase density is missing its factorized edge "
                            f"state at layer {layer_idx}"
                        )
                    phase_delta = self.phase_adapters[
                        phase_key
                    ].forward_diagonal_edges_factorized(
                        phase_edge_orbital,
                        phase_edge_norm_sq,
                        edge_dst=edge_index[1],
                        num_nodes=h.shape[0],
                        node_attrs=node_attrs,
                        node_type_idx=compact_idx,
                        species_embedding=phase_density_species_embedding,
                    )
                elif self.ictd_fix_phase_density_pairs in {
                    "full-gated",
                    "full-adaptive",
                    "full-adaptive-env",
                }:
                    if phase_edge_orbital is None or phase_edge_norm_sq is None:
                        raise RuntimeError(
                            "coherence-gated phase density is missing its factorized "
                            "edge state "
                            f"at layer {layer_idx}"
                        )
                    phase_delta = self.phase_adapters[
                        phase_key
                    ].forward_coherence_gated_factorized(
                        phase_doublet,
                        phase_edge_orbital,
                        phase_edge_norm_sq,
                        gate_features=(
                            h[:, : self.channels]
                            if self.ictd_fix_phase_density_pairs
                            == "full-adaptive-env"
                            else None
                        ),
                        edge_dst=edge_index[1],
                        num_nodes=h.shape[0],
                        node_attrs=node_attrs,
                        node_type_idx=compact_idx,
                        species_embedding=phase_density_species_embedding,
                    )
                elif self.ictd_fix_phase_density_pairs == "full-balanced":
                    if phase_edge_orbital is None or phase_edge_norm_sq is None:
                        raise RuntimeError(
                            "pair-count-balanced phase density is missing its "
                            f"factorized edge state at layer {layer_idx}"
                        )
                    if edge_env is None:
                        raise RuntimeError(
                            "pair-count-balanced phase density requires the smooth "
                            "cutoff envelope"
                        )
                    if self.avg_num_neighbors is None:
                        raise RuntimeError(
                            "pair-count-balanced phase density requires a fixed "
                            "avg_num_neighbors reference"
                        )
                    effective_coordination = (
                        interaction._phase_effective_coordination(
                            edge_env=edge_env,
                            edge_mask=edge_mask,
                            edge_dst=edge_index[1],
                            num_nodes=h.shape[0],
                            dtype=phase_doublet.dtype,
                            sync_after_scatter=sync_after_scatter,
                        )
                    )
                    phase_delta = self.phase_adapters[
                        phase_key
                    ].forward_pair_count_balanced_factorized(
                        phase_doublet,
                        phase_edge_orbital,
                        phase_edge_norm_sq,
                        effective_coordination,
                        reference_coordination=self.avg_num_neighbors,
                        edge_dst=edge_index[1],
                        num_nodes=h.shape[0],
                        node_attrs=node_attrs,
                        node_type_idx=compact_idx,
                        species_embedding=phase_density_species_embedding,
                    )
                elif self.ictd_fix_phase_density_pairs in {
                    "full-nonlinear",
                    "full-nonlinear-readout",
                }:
                    phase_delta = self.phase_adapters[phase_key].forward_doublet(
                        phase_doublet,
                        node_attrs=node_attrs,
                        node_type_idx=compact_idx,
                        gate_features=h[:, : self.channels],
                        species_embedding=phase_density_species_embedding,
                    )
                else:
                    phase_adapter = self.phase_adapters[phase_key]
                    if isinstance(phase_adapter, PhaseHermitianFullLResidual):
                        phase_delta = phase_adapter.forward_doublet(
                            phase_doublet,
                            node_attrs=node_attrs,
                            node_type_idx=compact_idx,
                            species_embedding=phase_density_species_embedding,
                        )
                    else:
                        phase_delta = phase_adapter.forward_doublet(
                            phase_doublet,
                            node_attrs=node_attrs,
                            node_type_idx=compact_idx,
                        )
                if phase_key in self.phase_direct_readouts:
                    phase_direct_energy = self.phase_direct_readouts[phase_key](
                        phase_delta[:, : self.channels]
                    )
                    direct_scale = self.phase_direct_readout_scales[phase_key].to(
                        dtype=phase_direct_energy.dtype,
                        device=phase_direct_energy.device,
                    )
                    phase_direct_energy = direct_scale * phase_direct_energy
                if self.ictd_fix_phase_placement in {"pre-product-l0", "pre-and-post"}:
                    # The Hermitian contraction is a neutral l=0 feature. Inject it into
                    # the scalar block before the ordinary MACE symmetric contraction so
                    # correlation terms can mix the real message with the phase density.
                    if message.shape[-1] < self.channels:
                        raise RuntimeError(
                            "phase pre-product injection requires an l=0 message block with "
                            f"at least {self.channels} channels, got shape {tuple(message.shape)}"
                        )
                    message = torch.cat(
                        (
                            message[..., : self.channels] + phase_delta,
                            message[..., self.channels :],
                        ),
                        dim=-1,
                    )
                elif self.ictd_fix_phase_placement == "pre-product-full-l":
                    if phase_delta.shape != message.shape:
                        raise RuntimeError(
                            "phase full-L pre-product injection requires a complete SO(3) "
                            f"message-shaped residual, got delta={tuple(phase_delta.shape)} "
                            f"message={tuple(message.shape)}"
                        )
                    message = message + phase_delta
            if type(product).__name__.startswith("Cueq"):
                h = product(node_feats=message, sc=sc, node_attrs=node_attrs, node_type_idx=compact_idx)
            else:
                h = product(node_feats=message, sc=sc, node_attrs=node_attrs)
            if (
                phase_delta is not None
                and self.ictd_fix_phase_placement in {"post-product", "pre-and-post"}
            ):
                h = h + phase_delta
            if self.scalar_ffns is not None:
                h = self.scalar_ffns[layer_idx](h)
            if layer_idx == 0 and self.mace_first_layer_sc0 is not None:
                add = self.mace_first_layer_sc0.to(dtype=h.dtype, device=h.device)[compact_idx]
                h = torch.cat((h[..., : self.channels] + add, h[..., self.channels :]), dim=-1)
            layer_states.append(h)
            if layer_idx == self.num_interaction - 1:
                last_preproduct_state = message  # last-layer interaction output (pre-product, full SO3, 2-hop)
            if layer_idx < self.num_interaction - 1:
                if self.ictd_fix_final_layer_readout_only:
                    e_layer = h.new_zeros((h.shape[0], 1))
                else:
                    readout_input = (
                        h[..., : self.channels]
                        if self.ictd_fix_nonlinear_layer_readouts
                        else h
                    )
                    e_layer = self.layer_energy_readouts[layer_idx](readout_input)
                    e_layer = self._readout_head_scale(0, e_layer) * e_layer
            else:
                e_layer = self.last_layer_energy_readout(h)
                e_layer = self._readout_head_scale(1, e_layer) * e_layer
            if phase_direct_energy is not None:
                e_layer = e_layer + phase_direct_energy
            total_energy = e_layer if total_energy is None else (total_energy + e_layer)

        out = total_energy.sum(dim=-1, keepdim=True)

        # Optional MACE ScaleShiftMACE-style scale/shift on the short-range interaction energy.
        # None when disabled -> byte-identical. Applied BEFORE the long-range add so it affects
        # only the network interaction energy; E0 is added outside this module.
        if self.energy_output_scale is not None:
            out = out * self.energy_output_scale.to(dtype=out.dtype, device=out.device)
        if self.energy_output_shift is not None:
            out = out + self.energy_output_shift.to(dtype=out.dtype, device=out.device)
        if self.element_energy_correction is not None:
            out = out + self.element_energy_correction[
                compact_idx
            ].to(dtype=out.dtype).unsqueeze(-1)

        # --- long-range additive term (skipped entirely when module is None) ---
        reciprocal_source = None
        if self.long_range_module is not None and self.multipole_readout is not None:
            # Multipole route: equivariant Cartesian monopole/dipole/quadrupole sources fed to the
            # mesh-FFT reciprocal kernel. Source = the last layer's INTERACTION output (pre-product
            # message): the interaction always emits full-SO(3) (edge_lmax); only the PRODUCT truncates
            # the last layer to l=0. So it carries l=1/2 at every depth, single-layer included, and is
            # the deepest such feature. multipole_readout front-slices via _split_irreps, so a wider
            # edge_lmax message is safe.
            mp_feat = last_preproduct_state
            if mp_feat is None:
                raise RuntimeError(
                    "multipole long-range needs a full-SO(3) node-feature layer; none found"
                )
            monopole, dipole, quadrupole = self.multipole_readout(mp_feat)
            if return_reciprocal_source and self.long_range_exports_reciprocal_source:
                # deployment: emit the packed [q|mu|Q] per-atom source for the C++ reciprocal
                # solver (mff_reciprocal_solver) to do the long-range sum; defer the energy here.
                from chorus.models.multipole_readout import pack_multipole_source

                reciprocal_source = pack_multipole_source(monopole, dipole, quadrupole)
            else:
                # training/validation: compute the reciprocal multipole energy in-model.
                long_range_energy = self.long_range_module.forward_multipole(
                    pos, batch, cell, monopole, dipole, quadrupole
                )
                if long_range_energy is not None:
                    out = out + long_range_energy
        elif self.long_range_module is not None:
            # final per-atom INVARIANT descriptor [N, channels] (in scope for both routes):
            # baseline last layer_state is already scalar; fusion last_preproduct is full-SO3 -> take l=0.
            last_state = layer_states[-1]
            # baseline: the last layer is scalar (l=0) -> the long-range source is the invariant
            # per-atom descriptor directly (no l>=1 multipole tap; that path needed the fusion route).
            if last_state.shape[-1] == self.channels:
                lr_feat = last_state
            else:
                lr_feat = _split_irreps(last_state, self.channels, self.lmax)[0].reshape(last_state.shape[0], self.channels)
            defer = False
            if return_reciprocal_source and self.long_range_exports_reciprocal_source:
                # Deploy: emit ONLY the latent charge source (source_head); defer the reciprocal
                # energy + neutralization to the C++ solver. emit_source avoids the kernel's per-graph
                # torch.nonzero so make_fx/AOTI can trace it (forward(return_source=True) would not).
                reciprocal_source = self.long_range_module.emit_source(lr_feat)
                long_range_energy = None
                defer = True
            else:
                long_range_energy = self.long_range_module(
                    lr_feat, pos, batch, cell, edge_src=edge_src, edge_dst=edge_dst
                )
            if long_range_energy is not None and not defer:
                out = out + long_range_energy

        # --- pairwise dispersion additive term (invariant) ---
        if self.dispersion is not None:
            last_state = layer_states[-1]
            disp_l2 = None
            if last_state.shape[-1] == self.channels:
                disp_feat = last_state
            else:
                disp_feat = _split_irreps(last_state, self.channels, self.lmax)[0].reshape(
                    last_state.shape[0], self.channels
                )
            # ANISOTROPIC MBD: feed the equivariant l=2 node block to the dispersion head (ICTC l=2 ->
            # 3x3 anisotropic polarizability). Source = the last layer's INTERACTION output (pre-product
            # message): the interaction always emits full-SO(3) (edge_lmax); only the PRODUCT truncates
            # the last layer to l=0. So the l=2 block is present at every depth, single-layer included.
            # _split_irreps front-slices l=2, so a wider edge_lmax message stays correct. (Without an
            # l=2 source disp_l2 stays None and polarizability_factor falls back to the ISOTROPIC b0*I,
            # silently leaving the anisotropic l2_mix/l2_gate untrained.)
            if self.mbd_anisotropic_polarizability and self.lmax >= 2 and last_preproduct_state is not None:
                disp_l2 = _split_irreps(last_preproduct_state, self.channels, self.lmax)[2]  # [N, channels, 5]
            if return_reciprocal_source and self.dispersion.exports_mbd_source():
                # Deploy: emit the head's (omega, alpha) as the MBD source and DEFER the coupled-dipole
                # energy to the C++ MBD solver (no double count). PACK after any electrostatic source so
                # a COMBINED model carries both: [elec | omega, alpha]. The C++ reciprocal solver ignores
                # the trailing channels; the MBD solver reads source[:, mbd_offset:mbd_offset+2].
                mbd_source = self.dispersion.emit_source(disp_feat, disp_l2)
                reciprocal_source = (
                    mbd_source if reciprocal_source is None
                    else torch.cat([reciprocal_source, mbd_source], dim=1)
                )
            else:
                disp_edge_src = edge_src
                disp_edge_dst = edge_dst
                disp_edge_length = edge_length
                disp_edge_vec = edge_vec
                disp_cutoff = self.dispersion_cutoff
                if dispersion_edge_src is not None or dispersion_edge_dst is not None:
                    if dispersion_edge_src is None or dispersion_edge_dst is None:
                        raise ValueError("dispersion_edge_src and dispersion_edge_dst must be provided together")
                    disp_edge_src = dispersion_edge_src
                    disp_edge_dst = dispersion_edge_dst
                    disp_cutoff = 0.0
                    if precomputed_dispersion_edge_vec is not None:
                        disp_edge_vec = precomputed_dispersion_edge_vec
                    else:
                        if dispersion_edge_shifts is None:
                            raise ValueError(
                                "dispersion_edge_shifts or precomputed_dispersion_edge_vec is required "
                                "when explicit dispersion edges are provided"
                            )
                        disp_shift = dispersion_edge_shifts.to(dtype=dtype)
                        disp_cells = cell[batch[disp_edge_dst]]
                        disp_shift_vec = torch.einsum("ni,nij->nj", disp_shift, disp_cells)
                        disp_edge_vec = pos[disp_edge_dst] - pos[disp_edge_src] + disp_shift_vec
                    disp_edge_length = disp_edge_vec.norm(dim=1)
                out = out + self.dispersion(
                    disp_feat,
                    pos,
                    batch,
                    cell,
                    edge_src=disp_edge_src,
                    edge_dst=disp_edge_dst,
                    edge_lengths=disp_edge_length,
                    edge_vec=disp_edge_vec,
                    cutoff=disp_cutoff,
                    pbc=self.dispersion_pbc,
                    l2_feats=disp_l2,
                )

        if return_combined_features:
            combined_features = torch.cat(layer_states, dim=-1)
            if return_reciprocal_source:
                rs = reciprocal_source if reciprocal_source is not None else out.new_empty((out.size(0), 0))
                return out, combined_features, rs
            return out, combined_features
        if return_reciprocal_source:
            rs = reciprocal_source if reciprocal_source is not None else out.new_empty((out.size(0), 0))
            return out, rs
        return out
