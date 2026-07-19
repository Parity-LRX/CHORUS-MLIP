"""
CHORUS-MLIP reference implementation.

CHORUS adds phase-coherent Hermitian neighbor aggregation to the MACE-ICTC
framework. The ``mace_ictc`` namespace is retained for checkpoint, compiled
extension, and deployment compatibility.
"""

__version__ = "0.1.0"

import torch
if not hasattr(torch, "compiler"):
    class _CompilerShim:
        @staticmethod
        def disable(fn=None, recursive=True):  # type: ignore[no-untyped-def]
            if fn is None:
                def decorator(inner):  # type: ignore[no-untyped-def]
                    return inner

                return decorator
            return fn

    torch.compiler = _CompilerShim()  # type: ignore[attr-defined]
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

# Import the compatible ICTC model and shared helpers. Keep this guarded so
# version metadata is still importable when heavy dependencies are absent.
try:
    from mace_ictc.models import (
        PureCartesianICTDFix,
        MainNet,
        MainNet2,
        RMSELoss,
        RobustScalarWeightedSum,
    )

    __all__ = [
        "PureCartesianICTDFix",
        "MainNet",
        "MainNet2",
        "RMSELoss",
        "RobustScalarWeightedSum",
    ]
except ImportError:
    # If dependencies are not installed, just define version
    __all__ = []
