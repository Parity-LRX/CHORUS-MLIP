from __future__ import annotations

import torch

from mace_ictc.cli import export_aoti_core as export_aoti


class _NoFoldProduct(torch.nn.Module):
    def forward(self, x):
        return x


class _FoldProduct(torch.nn.Module):
    def enable_e3nn_basis(self, _q_blocks=None) -> None:
        self.enabled = True

    def forward(self, x):
        return x


class _ToyModel(torch.nn.Module):
    def __init__(self, product: torch.nn.Module, *, backend: str = "ictd-bridge-u"):
        super().__init__()
        self.products = torch.nn.ModuleList([product])
        self.angular_basis = "ictd"
        self._e3nn_folded = False
        self.ictd_fix_product_backend = backend


class _DummyContraction(torch.nn.Module):
    def __init__(self, *, u_value: float, weight_value: float):
        super().__init__()
        self.weights_max = torch.nn.Parameter(torch.full((1, 2, 3), weight_value))
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.full((1, 4, 3), weight_value + 1.0))])
        self.register_buffer("U_matrix_1", torch.full((2, 2), u_value))


class _DummySymmetricContractions(torch.nn.Module):
    def __init__(self, *, u_value: float, weight_value: float):
        super().__init__()
        self.contractions = torch.nn.ModuleList([
            _DummyContraction(u_value=u_value, weight_value=weight_value),
        ])


def test_torch_export_retries_non_strict_after_strict_failure(monkeypatch) -> None:
    calls = []

    def fake_export(_gm, _inputs, *, dynamic_shapes=None, strict=True):
        calls.append(bool(strict))
        assert dynamic_shapes is None
        if strict:
            raise RuntimeError("synthetic strict export failure")
        return "exported"

    monkeypatch.setattr(torch.export, "export", fake_export)

    exported, strict = export_aoti._torch_export_with_strict_fallback(
        object(),
        (torch.ones(1),),
        dynamic_shapes=None,
        prefer_strict=True,
    )

    assert exported == "exported"
    assert strict is False
    assert calls == [True, False]


def test_cueq_replacement_copies_learned_weights_without_copying_u_buffers() -> None:
    src = _DummySymmetricContractions(u_value=11.0, weight_value=3.0)
    dst = _DummySymmetricContractions(u_value=-7.0, weight_value=0.0)

    export_aoti._copy_contraction_learnable_weights_only(src, dst)

    assert torch.equal(dst.contractions[0].weights_max, src.contractions[0].weights_max)
    assert torch.equal(dst.contractions[0].weights[0], src.contractions[0].weights[0])
    assert torch.equal(dst.contractions[0].U_matrix_1, torch.full((2, 2), -7.0))


def test_bridge_u_without_e3nn_fold_falls_back_to_ictd_basis(capsys) -> None:
    model = _ToyModel(_NoFoldProduct(), backend="ictd-bridge-u")

    basis = export_aoti._configure_angular_basis_for_export(model, "e3nn")

    assert basis == "ictd"
    assert model.angular_basis == "ictd"
    captured = capsys.readouterr()
    assert "bridge-U has no e3nn-fold path" in captured.out


def test_fold_capable_backend_keeps_requested_e3nn_basis() -> None:
    model = _ToyModel(_FoldProduct(), backend="cueq")

    basis = export_aoti._configure_angular_basis_for_export(model, "e3nn")

    assert basis == "e3nn"
    assert model.angular_basis == "e3nn"
