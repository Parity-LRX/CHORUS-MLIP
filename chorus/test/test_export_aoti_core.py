from __future__ import annotations

import torch

from chorus.cli import export_aoti_core as export_aoti
from chorus.models.pure_cartesian_ictd_fix import PhaseHermitianFullLResidual


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


class _PhasePruneModel(torch.nn.Module):
    def __init__(self, species_mode: str):
        super().__init__()
        self.atomic_numbers = (1, 6, 7, 8)
        self.num_elements = 4
        self.node_embedding = torch.nn.Linear(4, 3, bias=False)
        self.phase_density_species_embedding = (
            torch.nn.Embedding(4, 5)
            if species_mode == "embedded-lowrank"
            else None
        )
        self.phase_adapter = PhaseHermitianFullLResidual(
            num_elements=4,
            channels=3,
            lmax=1,
            density_rank=2,
            species_mode=species_mode,
            species_embedding_dim=5,
            species_rank=2,
        )
        self.register_buffer(
            "atomic_number_to_index",
            torch.tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3]),
        )


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


def test_element_pruning_handles_both_phase_species_modes() -> None:
    for species_mode in ("onehot-full", "embedded-lowrank"):
        model = _PhasePruneModel(species_mode)
        old_phase_embedding = model.phase_density_species_embedding
        old_phase_rows = (
            old_phase_embedding.weight.detach().clone()
            if old_phase_embedding is not None
            else None
        )
        old_output_rows = (
            {
                key: value.detach().clone()
                for key, value in model.phase_adapter.output_weights.items()
            }
            if model.phase_adapter.output_weights is not None
            else None
        )

        selected = export_aoti._prune_model_elements(model, [8, 1])

        assert selected == [8, 1]
        assert model.atomic_numbers == (8, 1)
        assert model.num_elements == 2
        assert model.phase_adapter.num_elements == 2
        if species_mode == "embedded-lowrank":
            assert model.phase_density_species_embedding is not None
            torch.testing.assert_close(
                model.phase_density_species_embedding.weight,
                old_phase_rows[[3, 0]],
            )
        else:
            assert model.phase_adapter.output_weights is not None
            for key, value in model.phase_adapter.output_weights.items():
                torch.testing.assert_close(value, old_output_rows[key][[3, 0]])
