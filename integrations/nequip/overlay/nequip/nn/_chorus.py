"""CHORUS channelwise Hermitian-density residuals."""

from typing import Dict, Optional

import math
import torch
from e3nn import o3
from torch_runstats.scatter import scatter
from . import _native_dataflow


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
    # One split concatenates adjoints instead of zero-filling a full input
    # for each irrep slice before gradient accumulation.
    sections=torch.split(tensor,[int(mul)*int(ir.dim) for mul,ir in irreps],dim=-1)
    for (mul, ir), section in zip(irreps, sections):
        blocks[int(ir.l)] = section.reshape(
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
        density_pairs: str = "full",
        openequivariance_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.irreps_message = o3.Irreps(irreps_message).simplify()
        self.irreps_node = o3.Irreps(irreps_node).simplify()
        self.rank = int(rank)
        self.num_elements = int(num_elements)
        self.avg_num_neighbors = avg_num_neighbors
        self.density_pairs = str(density_pairs).lower()
        if self.rank <= 0:
            raise ValueError("CHORUS rank must be positive")
        if self.num_elements <= 0:
            raise ValueError("CHORUS num_elements must be positive")
        if int(edge_invariant_dim) <= 0:
            raise ValueError("edge_invariant_dim must be positive")
        if self.density_pairs not in {"full", "diagonal", "aligned"}:
            raise ValueError(
                "density_pairs must be 'full', 'diagonal', or 'aligned', "
                f"got {self.density_pairs!r}"
            )

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
        phase_output = self.phase_network[-1]
        if not isinstance(phase_output, torch.nn.Linear):
            raise RuntimeError("CHORUS phase network must end in Linear")
        # Start the positive amplitude at one, but do not initialise every
        # edge at exactly the same phase.  At Delta theta = 0 the Hermitian
        # kernel has zero first derivative with respect to phase, so a
        # zero-initialised phase row is an absorbing aligned solution: Full
        # becomes bitwise equivalent to the aligned control and the phase head
        # can never learn.  A small MACE-style random phase row breaks only
        # that optimisation symmetry while keeping the initial lift close to
        # the unit-amplitude, phase-aligned state.
        with torch.no_grad():
            phase_output.weight[0].zero_()
            phase_output.bias[0].zero_()
            torch.nn.init.normal_(
                phase_output.weight[1],
                mean=0.0,
                std=0.1 / math.sqrt(float(hidden_channels)),
            )
            phase_output.bias[1].zero_()

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
        self.openequivariance_density = None
        self.factorized_phase = bool(openequivariance_enabled)
        self.factorized_density = bool(openequivariance_enabled)
        self.native_dataflow = bool(openequivariance_enabled) and _native_dataflow.enabled()
        if openequivariance_enabled:
            from ._openequivariance import OpenEquivarianceHermitianTensorProduct
            self.openequivariance_density = OpenEquivarianceHermitianTensorProduct(
                self.irreps_charged,self.paths)

    def charged_edge_messages(
        self,
        edge_messages: torch.Tensor,
        node_features: torch.Tensor,
        edge_invariants: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_invariants = self.node_norm(node_features)
        use_factorized_phase = self.factorized_phase
        if use_factorized_phase and not torch.jit.is_scripting():
            # Phase diagnostics and same-checkpoint interventions attach hooks
            # to this Sequential (or its first Linear). Keep their original
            # inputs/outputs and intervention semantics when instrumented.
            for module in (self.phase_network,self.phase_network[0]):
                if (module._forward_hooks or module._forward_pre_hooks or
                        module._backward_hooks or module._backward_pre_hooks):
                    use_factorized_phase = False
        if use_factorized_phase:
            # Distribute the existing first affine map over its three input
            # blocks. Node projections are evaluated once per node, not once
            # per edge. All original Parameter objects remain live.
            first=self.phase_network[0]
            width=edge_invariants.shape[-1]
            node_width=node_invariants.shape[-1]
            node_weight=torch.cat((first.weight[:,width:width+node_width],
                                   first.weight[:,width+node_width:]),dim=0)
            projected=torch.nn.functional.linear(node_invariants,node_weight)
            hidden=torch.nn.functional.linear(edge_invariants,first.weight[:,:width],first.bias)
            h=first.weight.shape[0]
            hidden=hidden+projected[:,:h][edge_src]+projected[:,h:][edge_dst]
            raw=self.phase_network[2](self.phase_network[1](hidden))
        else:
            context = torch.cat((edge_invariants,node_invariants[edge_src],
                                 node_invariants[edge_dst]),dim=-1)
            raw=self.phase_network(context)
        amplitude_raw, phase_raw = raw.unbind(-1)
        amplitude = torch.nn.functional.softplus(amplitude_raw) / math.log(2.0)
        phase = math.pi * torch.tanh(phase_raw)
        if self.density_pairs == "aligned":
            # Retain the complete Hermitian j,k closure and the identical
            # parameter/state schema, while removing relative phase during
            # training.  Multiplication by zero keeps the inactive head in the
            # graph and makes the causal contrast auditable from checkpoints.
            phase = phase * 0.0
        charged = (_native_dataflow.linear(self.edge_projection,edge_messages)
                   if self.native_dataflow and not torch.jit.is_scripting() else self.edge_projection(edge_messages))
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
        if self.openequivariance_density is not None:
            return self.openequivariance_density(charged_real,charged_imag)
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
        return self.contract_density(density, node_attrs=node_attrs)

    def contract_density(
        self,
        density: Dict[int, torch.Tensor],
        node_attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map precomputed neutral density blocks to the message layout."""
        output_blocks: Dict[int, torch.Tensor] = {}
        for out_l, block in density.items():
            weight = self.output_weights[str(out_l)]
            if self.factorized_density:
                # Shared bank GEMM before species mixing avoids a separate
                # [N,Cout,Cin] weight tensor. It also preserves arbitrary soft
                # node attributes and all their gradients, not only one-hot.
                n,_,m=block.shape
                # An explicit 2-D GEMM is important: Linear on the strided
                # [N,m,Cin] view may dispatch N small broadcasted BMMs.
                angular=block.transpose(1,2).reshape(n*m,block.shape[1])
                if node_attrs is None:
                    value=torch.nn.functional.linear(angular,weight[0]).reshape(n,m,weight.shape[1]).transpose(1,2)
                else:
                    mapped=torch.nn.functional.linear(angular,weight.flatten(0,1))
                    mapped=mapped.reshape(n,m,weight.shape[0],weight.shape[1])
                    value=torch.einsum('ne,nmeo->nom',node_attrs.to(dtype=weight.dtype),mapped)
            elif node_attrs is None:
                effective_weight = weight[0].expand(
                    block.shape[0], -1, -1
                )
            else:
                effective_weight = torch.einsum(
                    "ne,eoi->noi",
                    node_attrs.to(dtype=weight.dtype),
                    weight,
                )
            if not self.factorized_density:
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
        if self.density_pairs == "diagonal":
            return self.diagonal_from_edges(
                edge_messages=edge_messages,
                node_features=node_features,
                node_attrs=node_attrs,
                edge_invariants=edge_invariants,
                edge_src=edge_src,
                edge_dst=edge_dst,
                num_nodes=num_nodes,
            )
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

    def diagonal_from_edges(
        self,
        edge_messages: torch.Tensor,
        node_features: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_invariants: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Build only the same-edge ``j=k`` Hermitian self-density.

        Neutral edge densities are formed before neighbour reduction.  This
        removes every ``j!=k`` term while retaining the same projections,
        phase/amplitude network, CG paths, output maps, and parameter count as
        the full operator.
        """
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
        density_edges = self.density_blocks(
            charged_real / normalizer,
            charged_imag / normalizer,
        )
        density_nodes = {
            out_l: scatter(
                block,
                edge_dst,
                dim=0,
                dim_size=int(num_nodes),
            )
            for out_l, block in density_edges.items()
        }
        neutral = self.contract_density(density_nodes, node_attrs=node_attrs)
        return neutral / normalizer

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
            charged_real if self.native_dataflow else charged_real / normalizer,
            edge_dst,
            dim=0,
            dim_size=num_nodes,
        )
        charged_imag = scatter(
            charged_imag if self.native_dataflow else charged_imag / normalizer,
            edge_dst,
            dim=0,
            dim_size=num_nodes,
        )
        if self.native_dataflow:
            charged_real=charged_real/normalizer
            charged_imag=charged_imag/normalizer
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
