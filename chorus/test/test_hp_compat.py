"""Backend packaging contract; run on an allocated remote test host."""

from types import SimpleNamespace

import pytest
import torch

from chorus.hp import configure_model
from chorus.hp import _compat


def _metadata(monkeypatch, version="0.2.12", module_version="0.2.12", api=1):
    monkeypatch.setattr(_compat, "_version", lambda name: version)
    monkeypatch.setattr(_compat, "_import_module", lambda name: SimpleNamespace(
        __version__=module_version, __hp_api_version__=api))


def test_matching_optional_backend(monkeypatch):
    _metadata(monkeypatch)
    assert _compat.backend_reasons() == []


def test_missing_backend_does_not_import_namespace(monkeypatch):
    def missing(name):
        raise _compat.PackageNotFoundError(name)

    def forbidden(name):
        raise AssertionError("missing backend must not import mace_ictc")

    monkeypatch.setattr(_compat, "_version", missing)
    monkeypatch.setattr(_compat, "_import_module", forbidden)
    assert "not installed" in _compat.backend_reasons()[0]
    assert "[hp] extra" in _compat.backend_reasons()[0]


@pytest.mark.parametrize("version", ["0.2.11", "0.2.13", "0.3.0", "0.2.12.dev1"])
def test_unqualified_release_is_rejected(monkeypatch, version):
    _metadata(monkeypatch, version=version)
    assert f"found {version}" in _compat.backend_reasons()[0]


@pytest.mark.parametrize("version,api", [("0.2.11", 1), ("0.2.12", None), ("0.2.12", 2)])
def test_namespace_collision_is_rejected(monkeypatch, version, api):
    _metadata(monkeypatch, module_version=version, api=api)
    assert "shadow" in _compat.backend_reasons()[0]


def test_backend_import_failure_is_reported(monkeypatch):
    _metadata(monkeypatch)

    def broken(name):
        raise ImportError("missing optional binary")

    monkeypatch.setattr(_compat, "_import_module", broken)
    assert "missing optional binary" in _compat.backend_reasons()[0]


def test_reference_policy_never_checks_or_loads_hp(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reference policy must not check or import HP")

    monkeypatch.setattr(_compat, "backend_reasons", forbidden)
    monkeypatch.setattr(_compat, "_import_module", forbidden)
    model = torch.nn.Linear(2, 1)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert configure_model(model, policy="reference")["selected"] == "reference"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
