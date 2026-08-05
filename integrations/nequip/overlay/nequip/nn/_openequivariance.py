"""Optional OpenEquivariance tensor products for the NequIP interaction.

CHORUS consumes per-edge equivariant messages before neighbor reduction.  The
standalone OpenEquivariance tensor product therefore provides the appropriate
acceleration boundary: it replaces the e3nn tensor-product kernel while
preserving the edge-message tensor needed by both Final and Persistent CHORUS.
"""

from __future__ import annotations

import torch


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
