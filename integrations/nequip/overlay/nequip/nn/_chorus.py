"""CHORUS channelwise Hermitian-density residuals."""

from typing import Dict, Optional

import math
import torch
from e3nn import o3
from torch_runstats.scatter import scatter


CHORUS_CHARGED_REAL_KEY = "chorus_charged_real"
CHORUS_CHARGED_IMAG_KEY = "chorus_charged_imag"


def _natural_parity(l_value: int) -> int:
    return 1 if int(l_value) % 2 == 0 else -1


def _validate_natural_layout(irreps: o3.Irreps) -> Dict[int, int]:
    """Return one multiplicity per l for a natural-parity layout."""
    multiplicities: Dict[int, int] = {}
    for mul, ir in irreps:
        if ir.p != _natural_parity(ir.l):
            raise ValueError(
                "CHORUS channelwise density requires natural-parity irreps; "
                f"got {mul}x{ir}"
            )
        if int(ir.l) in multiplicities:
            raise ValueError(
                "CHORUS channelwise density requires one block per angular "
                f"degree; repeated l={ir.l} in {irreps}"
            )
        multiplicities[int(ir.l)] = int(mul)
    if not multiplicities or 0 not in multiplicities:
        raise ValueError("CHORUS channelwise density requires an l=0 block")
    expected = set(range(max(multiplicities) + 1))
    if set(multiplicities) != expected:
        raise ValueError(
            "CHORUS channelwise density requires contiguous angular degrees "
            f"0..lmax; got {sorted(multiplicities)}"
        )
    return multiplicities


def _split_by_l(tensor: torch.Tensor, irreps: o3.Irreps) -> Dict[int, torch.Tensor]:
    blocks: Dict[int, torch.Tensor] = {}
    for (mul, ir), component_slice in zip(irreps, irreps.slices()):
        blocks[int(ir.l)] = tensor[..., component_slice].reshape(
            *tensor.shape[:-1], int(mul), int(ir.dim)
        )
    return blocks


def _merge_by_l(blocks: Dict[int, torch.Tensor], irreps: o3.Irreps) -> torch.Tensor:
    pieces = []
    for mul, ir in irreps:
        block = blocks[int(ir.l)]
        pieces.append(block.reshape(*block.shape[:-2], int(mul) * int(ir.dim)))
    return torch.cat(pieces, dim=-1)


class HermitianDensityResidual(torch.nn.Module):
    r"""Map a charged atomic environment to a neutral channelwise density.

    A shared equivariant projection reduces each angular block from its message
    multiplicity to ``rank`` latent orbitals.  Fixed Clebsch--Gordan tensors
    then contract only equal latent ranks,

    .. math::

        \rho^{LM}_{r} =
        \sum_{m_1m_2} C^{LM}_{l_1m_1,l_2m_2}
        z^{l_1}_{rm_1} z^{l_2*}_{rm_2}.

    There is deliberately no learned ``r1 x r2`` tensor product.  Learnable
    output weights mix the resulting ``(path, rank)`` axis back to the original
    message multiplicities.  This is the same low-rank/channelwise structure as
    the MACE CHORUS operator.
    """

    def __init__(
        self,
        irreps_message,
        irreps_node,
        edge_invariant_dim: int,
        num_elements: int,
        rank: int = 8,
        hidden_channels: int = 32,
        avg_num_neighbors: Optional[float] = None,
        scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.irreps_message = o3.Irreps(irreps_message).simplify()
        self.irreps_node = o3.Irreps(irreps_node).simplify()
        self.rank = int(rank)
        self.num_elements = int(num_elements)
        self.avg_num_neighbors = avg_num_neighbors
        if self.rank <= 0:
            raise ValueError("CHORUS rank must be positive")
        if self.num_elements <= 0:
            raise ValueError("CHORUS num_elements must be positive")
        if int(edge_invariant_dim) <= 0:
            raise ValueError("edge_invariant_dim must be positive")

        message_multiplicities = _validate_natural_layout(self.irreps_message)
        self.lmax = max(message_multiplicities)
        self.irreps_charged = o3.Irreps(
            [(self.rank, (l_value, _natural_parity(l_value)))
             for l_value in range(self.lmax + 1)]
        )

        # One C_l -> R projection for every angular block.  The same projected
        # orbital is used by the real and imaginary components because phase is
        # an invariant scalar and therefore commutes with this map.
        self.edge_projection = o3.Linear(
            self.irreps_message,
            self.irreps_charged,
            internal_weights=True,
            shared_weights=True,
        )
        self.node_norm = o3.Norm(self.irreps_node, squared=True)
        phase_in = (
            int(edge_invariant_dim)
            + 2 * self.node_norm.irreps_out.num_irreps
        )
        self.phase_network = torch.nn.Sequential(
            torch.nn.Linear(phase_in, int(hidden_channels)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_channels), 2),
        )
        torch.nn.init.zeros_(self.phase_network[-1].weight)
        torch.nn.init.zeros_(self.phase_network[-1].bias)

        # A path is (l1, l2, L, component, cg_buffer_name).  l1 <= l2 avoids
        # storing both Hermitian-conjugate blocks.  For natural parity, the
        # antisymmetric component vanishes when l1 == l2.
        self.paths = []
        input_multiplicities = {l_value: 0 for l_value in range(self.lmax + 1)}
        cg_index = 0
        for l1 in range(self.lmax + 1):
            for l2 in range(l1, self.lmax + 1):
                for out_l in range(
                    abs(l1 - l2), min(l1 + l2, self.lmax) + 1
                ):
                    if (l1 + l2 + out_l) % 2:
                        continue
                    # Preserve the exact fixed geometry at float64 precision.
                    # The forward path casts it to the feature dtype; creating
                    # it at float32 and later calling module.double() cannot
                    # recover the discarded precision.
                    cg = o3.wigner_3j(
                        l1, l2, out_l, dtype=torch.float64
                    )
                    cg = cg * (
                        math.sqrt(float(2 * out_l + 1))
                        / cg.square().sum().sqrt().clamp_min(1.0e-30)
                    )
                    cg_name = f"cg_{cg_index}"
                    self.register_buffer(cg_name, cg.contiguous(), persistent=False)
                    self.paths.append((l1, l2, out_l, "real", cg_name))
                    input_multiplicities[out_l] += self.rank
                    if l1 < l2:
                        self.paths.append((l1, l2, out_l, "imag", cg_name))
                        input_multiplicities[out_l] += self.rank
                    cg_index += 1

        self.input_multiplicities = tuple(
            input_multiplicities[l_value]
            for l_value in range(self.lmax + 1)
        )
        self.output_weights = torch.nn.ParameterDict()
        for out_l, in_mul in enumerate(self.input_multiplicities):
            out_mul = message_multiplicities[out_l]
            weight = torch.nn.Parameter(
                torch.empty(self.num_elements, out_mul, int(in_mul))
            )
            torch.nn.init.normal_(
                weight,
                mean=0.0,
                std=1.0 / math.sqrt(float(max(in_mul, 1))),
            )
            self.output_weights[str(out_l)] = weight
        self.residual_scale = torch.nn.Parameter(
            torch.full(
                (self.lmax + 1,),
                float(scale_init),
                dtype=torch.get_default_dtype(),
            )
        )

    def charged_edge_messages(
        self,
        edge_messages: torch.Tensor,
        node_features: torch.Tensor,
        edge_invariants: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_invariants = self.node_norm(node_features)
        context = torch.cat(
            (
                edge_invariants,
                node_invariants[edge_src],
                node_invariants[edge_dst],
            ),
            dim=-1,
        )
        amplitude_raw, phase_raw = self.phase_network(context).unbind(-1)
        amplitude = torch.nn.functional.softplus(amplitude_raw) / math.log(2.0)
        phase = math.pi * torch.tanh(phase_raw)
        charged = self.edge_projection(edge_messages)
        return (
            charged * (amplitude * torch.cos(phase)).unsqueeze(-1),
            charged * (amplitude * torch.sin(phase)).unsqueeze(-1),
        )

    @staticmethod
    def _couple(
        left: torch.Tensor, right: torch.Tensor, cg: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("...rm,...rn,mnk->...rk", left, right, cg)

    def density_blocks(
        self, charged_real: torch.Tensor, charged_imag: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        real_blocks = _split_by_l(charged_real, self.irreps_charged)
        imag_blocks = _split_by_l(charged_imag, self.irreps_charged)
        outputs = {l_value: [] for l_value in range(self.lmax + 1)}
        for l1, l2, out_l, component, cg_name in self.paths:
            cg = getattr(self, cg_name).to(
                dtype=charged_real.dtype, device=charged_real.device
            )
            if component == "real":
                value = self._couple(real_blocks[l1], real_blocks[l2], cg)
                value = value + self._couple(
                    imag_blocks[l1], imag_blocks[l2], cg
                )
            else:
                value = self._couple(imag_blocks[l1], real_blocks[l2], cg)
                value = value - self._couple(
                    real_blocks[l1], imag_blocks[l2], cg
                )
            outputs[out_l].append(value)
        return {
            out_l: torch.cat(values, dim=-2)
            for out_l, values in outputs.items()
        }

    def contract_charged(
        self,
        charged_real: torch.Tensor,
        charged_imag: torch.Tensor,
        node_attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contract a real U(1) doublet into neutral real irreps."""
        density = self.density_blocks(charged_real, charged_imag)
        output_blocks: Dict[int, torch.Tensor] = {}
        for out_l, block in density.items():
            weight = self.output_weights[str(out_l)]
            if node_attrs is None:
                effective_weight = weight[0].expand(
                    block.shape[0], -1, -1
                )
            else:
                effective_weight = torch.einsum(
                    "ne,eoi->noi",
                    node_attrs.to(dtype=weight.dtype),
                    weight,
                )
            value = torch.einsum("noi,nim->nom", effective_weight, block)
            scale = self.residual_scale[out_l].to(
                dtype=value.dtype, device=value.device
            )
            output_blocks[out_l] = scale * value
        return _merge_by_l(output_blocks, self.irreps_message)

    def forward(
        self,
        edge_messages: torch.Tensor,
        node_features: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_invariants: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        charged_real, charged_imag = self.aggregate_charged(
            edge_messages=edge_messages,
            node_features=node_features,
            edge_invariants=edge_invariants,
            edge_src=edge_src,
            edge_dst=edge_dst,
            num_nodes=num_nodes,
        )
        return self.neutral_from_charged(
            charged_real,
            charged_imag,
            node_attrs=node_attrs,
        )

    def aggregate_charged(
        self,
        edge_messages: torch.Tensor,
        node_features: torch.Tensor,
        edge_invariants: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift edge messages and aggregate one incoming charged state."""
        charged_real, charged_imag = self.charged_edge_messages(
            edge_messages,
            node_features,
            edge_invariants,
            edge_src,
            edge_dst,
        )
        normalizer = 1.0
        if self.avg_num_neighbors is not None:
            normalizer = float(self.avg_num_neighbors) ** 0.5
        charged_real = scatter(
            charged_real / normalizer,
            edge_dst,
            dim=0,
            dim_size=num_nodes,
        )
        charged_imag = scatter(
            charged_imag / normalizer,
            edge_dst,
            dim=0,
            dim_size=num_nodes,
        )
        return charged_real, charged_imag

    def neutral_from_charged(
        self,
        charged_real: torch.Tensor,
        charged_imag: torch.Tensor,
        node_attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hermitian-neutralize an already aggregated charged state."""
        neutral = self.contract_charged(
            charged_real, charged_imag, node_attrs=node_attrs
        )
        normalizer = 1.0
        if self.avg_num_neighbors is not None:
            normalizer = float(self.avg_num_neighbors) ** 0.5
        return neutral / normalizer


class PersistentChargedUpdate(torch.nn.Module):
    r"""Depth update for a global-\(U(1)\) charged NequIP stream.

    Each angular block uses the same channel maps for the real and imaginary
    components.  The update therefore commutes with a common U(1) rotation and
    with O(3).  A bounded learned gate mixes the previous charged environment
    with the incoming charged environment at the current interaction layer.
    """

    def __init__(self, irreps_charged) -> None:
        super().__init__()
        self.irreps_charged = o3.Irreps(irreps_charged).simplify()
        multiplicities = _validate_natural_layout(self.irreps_charged)
        self.lmax = max(multiplicities)
        self.previous_weights = torch.nn.ParameterList()
        self.incoming_weights = torch.nn.ParameterList()
        for l_value in range(self.lmax + 1):
            multiplicity = multiplicities[l_value]
            identity = torch.eye(
                multiplicity, dtype=torch.get_default_dtype()
            )
            self.previous_weights.append(
                torch.nn.Parameter(identity.clone())
            )
            self.incoming_weights.append(
                torch.nn.Parameter(identity.clone())
            )
        self.memory_logits = torch.nn.Parameter(
            torch.zeros(self.lmax + 1, dtype=torch.get_default_dtype())
        )

    @staticmethod
    def _channel_linear(
        block: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("...cm,oc->...om", block, weight)

    def forward(
        self,
        previous_real: torch.Tensor,
        previous_imag: torch.Tensor,
        incoming_real: torch.Tensor,
        incoming_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_real_blocks = _split_by_l(
            previous_real, self.irreps_charged
        )
        previous_imag_blocks = _split_by_l(
            previous_imag, self.irreps_charged
        )
        incoming_real_blocks = _split_by_l(
            incoming_real, self.irreps_charged
        )
        incoming_imag_blocks = _split_by_l(
            incoming_imag, self.irreps_charged
        )
        gates = self.memory_logits.to(
            dtype=previous_real.dtype, device=previous_real.device
        ).sigmoid()
        real_blocks: Dict[int, torch.Tensor] = {}
        imag_blocks: Dict[int, torch.Tensor] = {}
        for l_value in range(self.lmax + 1):
            gate = gates[l_value]
            previous_weight = self.previous_weights[l_value].to(
                dtype=previous_real.dtype, device=previous_real.device
            )
            incoming_weight = self.incoming_weights[l_value].to(
                dtype=previous_real.dtype, device=previous_real.device
            )
            real_blocks[l_value] = gate * self._channel_linear(
                previous_real_blocks[l_value], previous_weight
            ) + (1.0 - gate) * self._channel_linear(
                incoming_real_blocks[l_value], incoming_weight
            )
            imag_blocks[l_value] = gate * self._channel_linear(
                previous_imag_blocks[l_value], previous_weight
            ) + (1.0 - gate) * self._channel_linear(
                incoming_imag_blocks[l_value], incoming_weight
            )
        return (
            _merge_by_l(real_blocks, self.irreps_charged),
            _merge_by_l(imag_blocks, self.irreps_charged),
        )
