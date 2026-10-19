"""Optional backend contract; never imported by the reference model path."""

from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError, version as _version


HP_DISTRIBUTION = "ictc-o3-hp"
HP_VERSION = "0.2.12"
HP_API_VERSION = 1
HP_COMMIT = "9fdea47eef620013e9ea591f7d90534bb27671c8"
HP_REQUIREMENT = f"ictc-o3-hp[hp] @ git+https://github.com/Parity-LRX/ICTC-O3-HP.git@{HP_COMMIT}"
_INSTALL = "reinstall this CHORUS source revision with its [hp] extra"


def backend_reasons():
    """Return a diagnostic instead of importing kernels or changing a model.

    Check distribution metadata *and* the imported namespace: an old MACE-ICTC
    installation can overwrite ``mace_ictc`` while leaving HP metadata behind.
    Source checkouts should be installed editable so their metadata also agrees.
    """
    try:
        installed = _version(HP_DISTRIBUTION)
    except PackageNotFoundError:
        return [f"optional {HP_REQUIREMENT} is not installed; {_INSTALL}"]
    if installed != HP_VERSION:
        return [f"CHORUS requires ICTC-O3-HP {HP_VERSION} / API {HP_API_VERSION}, found {installed}; {_INSTALL}"]
    try:
        backend = _import_module("mace_ictc")
    except ImportError as error:
        return [f"cannot import the installed {HP_REQUIREMENT}: {error}"]
    if (getattr(backend, "__version__", None) != HP_VERSION
            or getattr(backend, "__hp_api_version__", None) != HP_API_VERSION):
        return [
            "mace_ictc does not match the installed ICTC-O3-HP release/API; "
            "an old MACE-ICTC checkout may shadow it. Use an isolated environment "
            f"and {_INSTALL}"
        ]
    return []
