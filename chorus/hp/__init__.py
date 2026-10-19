"""Optional ICTC-O3-HP execution for CHORUS, without changing its model.

Importing this package does not import the optional HP library or compile kernels.
"""

from .runtime import check_model, configure_model
from .capture import CapturedInference, capture_inference
from .inference import energy_forces_virial
from .training import backward_parameters


def prepare_for_export(model, **kwargs):
    """Lazily import the optional prepared AOTInductor implementation."""
    from .export import prepare_for_export as prepare
    return prepare(model, **kwargs)

__all__ = ["check_model", "configure_model", "CapturedInference", "capture_inference",
           "energy_forces_virial", "backward_parameters", "prepare_for_export"]
