"""Optional OpenEquivariance tensor products for the NequIP interaction.

CHORUS consumes per-edge equivariant messages before neighbor reduction.  The
standalone OpenEquivariance tensor product therefore provides the appropriate
acceleration boundary: it replaces the e3nn tensor-product kernel while
preserving the edge-message tensor needed by both Final and Persistent CHORUS.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from e3nn import o3


class OpenEquivarianceTensorProduct(torch.nn.Module):
    """Parameter-free OpenEquivariance replacement for an e3nn TensorProduct."""

    def __init__(
        self,
        feature_irreps_in,
        irreps_edge_attr,
        irreps_mid,
        instructions,
    ) -> None:
        super().__init__()
        try:
            from openequivariance import (
                TPProblem,
                TensorProduct,
                torch_to_oeq_dtype,
            )
        except ImportError as exc:
            raise ImportError(
                "openequivariance_enabled=True requires the "
                "'openequivariance' package"
            ) from exc

        model_dtype = torch.get_default_dtype()
        problem = TPProblem(
            feature_irreps_in,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            irrep_dtype=torch_to_oeq_dtype(model_dtype),
            weight_dtype=torch_to_oeq_dtype(model_dtype),
            shared_weights=False,
            internal_weights=False,
        )
        self.tensor_product = TensorProduct(
            problem,
            torch_op=True,
            use_opaque=False,
        )
        self.model_dtype = model_dtype

    def forward(
        self,
        node_features: torch.Tensor,
        edge_attributes: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Explicit casts make the strict-float32 contract robust to ambient AMP.
        return self.tensor_product(
            node_features.to(self.model_dtype),
            edge_attributes.to(self.model_dtype),
            edge_weights.to(self.model_dtype),
        )


class OpenEquivarianceHermitianTensorProduct(torch.nn.Module):
    """OEq backend for the channelwise CHORUS Hermitian contraction.

    The charged real and imaginary irreps are packed into the two halves of a
    single tensor-product input.  Every allowed ``(l1, l2, L)`` channelwise
    contraction is then emitted by one OpenEquivariance ``uvu`` problem with
    fixed identity weights.  This exactly selects equal latent ranks while
    using the connection modes supported by the current OEq CUDA kernel.  The
    positive real terms accumulate directly in the OEq output; antisymmetric
    imaginary terms are emitted separately and subtracted after the kernel.

    This module is parameter free.  Its output has exactly the same
    ``(path, rank, m)`` ordering as the eager CHORUS implementation.
    """

    def __init__(
        self,
        irreps_charged,
        paths: Iterable[tuple[int, int, int, str, str]],
    ) -> None:
        super().__init__()
        try:
            from openequivariance import (
                TPProblem,
                TensorProduct,
                torch_to_oeq_dtype,
            )
        except ImportError as exc:
            raise ImportError(
                "openequivariance_enabled=True requires the "
                "'openequivariance' package"
            ) from exc

        charged = o3.Irreps(irreps_charged)
        if len(charged) == 0:
            raise ValueError("CHORUS charged irreps cannot be empty")
        multiplicities = {int(ir.l): int(mul) for mul, ir in charged}
        if len(multiplicities) != len(charged):
            raise ValueError(
                "OEq CHORUS requires one charged block per angular degree"
            )
        if len(set(multiplicities.values())) != 1:
            raise ValueError(
                "OEq CHORUS requires a common channelwise rank for all l"
            )
        self.rank = next(iter(multiplicities.values()))
        self.lmax = max(multiplicities)
        self.input_irreps = charged + charged

        output_irreps = []
        instructions = []
        constant_weights = []
        # Records are (out_l, output_index).  The fixed Hermitian signs are
        # supplied to OEq as non-trainable channelwise weights.
        output_records = []
        num_charged_blocks = len(charged)

        def real_index(l_value: int) -> int:
            return int(l_value)

        def imag_index(l_value: int) -> int:
            return num_charged_blocks + int(l_value)

        identity_weight = torch.eye(self.rank).reshape(-1).tolist()
        negative_identity_weight = (
            -torch.eye(self.rank)
        ).reshape(-1).tolist()

        for l1, l2, out_l, component, _ in paths:
            if component == "imag":
                continue
            out_irrep = (int(out_l), 1 if int(out_l) % 2 == 0 else -1)

            real_out = len(output_irreps)
            output_irreps.append((self.rank, out_irrep))
            instructions.extend(
                (
                    (
                        real_index(l1),
                        real_index(l2),
                        real_out,
                        "uvu",
                        True,
                    ),
                    (
                        imag_index(l1),
                        imag_index(l2),
                        real_out,
                        "uvu",
                        True,
                    ),
                )
            )
            constant_weights.extend(identity_weight)
            constant_weights.extend(identity_weight)
            output_records.append((int(out_l), real_out))

            if int(l1) < int(l2):
                imag_out = len(output_irreps)
                output_irreps.append((self.rank, out_irrep))
                instructions.append(
                    (
                        imag_index(l1),
                        real_index(l2),
                        imag_out,
                        "uvu",
                        True,
                    )
                )
                instructions.append(
                    (
                        real_index(l1),
                        imag_index(l2),
                        imag_out,
                        "uvu",
                        True,
                    )
                )
                constant_weights.extend(identity_weight)
                constant_weights.extend(negative_identity_weight)
                output_records.append((int(out_l), imag_out))

        self.output_irreps = o3.Irreps(output_irreps)
        self.output_records = tuple(output_records)
        self.output_slices = tuple(self.output_irreps.slices())
        model_dtype = torch.get_default_dtype()
        problem = TPProblem(
            self.input_irreps,
            self.input_irreps,
            self.output_irreps,
            instructions,
            irrep_dtype=torch_to_oeq_dtype(model_dtype),
            weight_dtype=torch_to_oeq_dtype(model_dtype),
            shared_weights=True,
            internal_weights=False,
            # component/none reproduces sqrt(2L+1) times the normalized
            # real-basis Wigner-3j tensor used by eager CHORUS.
            irrep_normalization="component",
            path_normalization="none",
        )
        if problem.weight_numel != len(constant_weights):
            raise RuntimeError(
                "OEq CHORUS constant-weight layout is inconsistent"
            )
        self.tensor_product = TensorProduct(
            problem,
            torch_op=True,
            use_opaque=False,
        )
        self.register_buffer(
            "constant_weights",
            torch.tensor(constant_weights, dtype=model_dtype),
            persistent=False,
        )

    def forward(
        self,
        charged_real: torch.Tensor,
        charged_imag: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        packed = torch.cat((charged_real, charged_imag), dim=-1)
        coupled = self.tensor_product(
            packed,
            packed,
            self.constant_weights.to(dtype=packed.dtype, device=packed.device),
        )
        outputs = {l_value: [] for l_value in range(self.lmax + 1)}
        for out_l, output_index in self.output_records:
            output_slice = self.output_slices[output_index]
            value = coupled[..., output_slice].reshape(
                *coupled.shape[:-1], self.rank, 2 * out_l + 1
            )
            outputs[out_l].append(value)
        return {
            out_l: torch.cat(values, dim=-2)
            for out_l, values in outputs.items()
        }
