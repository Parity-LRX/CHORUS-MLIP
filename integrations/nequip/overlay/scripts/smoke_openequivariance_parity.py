"""Numerical and differential parity checks for the OpenEquivariance backend."""

from __future__ import annotations

import copy

import torch
from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.model import model_from_config


BASE_CONFIG = {
    "model_builders": ["EnergyModel", "ForceOutput"],
    "num_types": 2,
    "chemical_symbol_to_type": {"H": 0, "C": 1},
    "chemical_embedding_irreps_out": "8x0e",
    "irreps_edge_sh": "0e + 1o + 2e",
    "feature_irreps_hidden": "8x0e + 8x1o + 8x2e",
    "conv_to_output_hidden_irreps_out": "4x0e",
    "r_max": 4.0,
    "num_layers": 2,
    "num_basis": 4,
    "PolynomialCutoff_p": 6,
    "nonlinearity_type": "gate",
    "avg_num_neighbors": 4.0,
    "use_sc": True,
    "invariant_layers": 1,
    "invariant_neurons": 16,
    "interaction_backend": "e3nn",
    "chorus_rank": 4,
    "chorus_hidden_channels": 16,
    "chorus_scale_init": 0.05,
    "model_dtype": "float32",
}


def graph(positions: torch.Tensor) -> AtomicDataDict.Type:
    n_atoms = positions.shape[0]
    edge_src = []
    edge_dst = []
    for dst in range(n_atoms):
        for src in range(n_atoms):
            if src != dst:
                edge_src.append(src)
                edge_dst.append(dst)
    edge_index = torch.tensor(
        [edge_dst, edge_src], dtype=torch.long, device=positions.device
    )
    return {
        AtomicDataDict.POSITIONS_KEY: positions,
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros(
            edge_index.shape[1], 3, device=positions.device
        ),
        AtomicDataDict.CELL_KEY: torch.zeros(3, 3, device=positions.device),
        AtomicDataDict.PBC_KEY: torch.zeros(
            3, dtype=torch.bool, device=positions.device
        ),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(
            [0, 1, 0, 1, 0, 1], dtype=torch.long, device=positions.device
        ),
        AtomicDataDict.BATCH_KEY: torch.zeros(
            n_atoms, dtype=torch.long, device=positions.device
        ),
        AtomicDataDict.BATCH_PTR_KEY: torch.tensor(
            [0, n_atoms], dtype=torch.long, device=positions.device
        ),
    }


def outputs(model, positions):
    result = model(graph(positions))
    return (
        result[AtomicDataDict.TOTAL_ENERGY_KEY],
        result[AtomicDataDict.FORCE_KEY],
    )


def build_pair(chorus_enabled: bool, scope: str):
    torch.manual_seed(20260803)
    reference_config = dict(BASE_CONFIG)
    reference_config.update(
        chorus_enabled=chorus_enabled,
        chorus_scope=scope,
        openequivariance_enabled=False,
    )
    reference = model_from_config(reference_config, initialize=True).cuda()
    accelerated_config = dict(reference_config)
    accelerated_config["openequivariance_enabled"] = True
    accelerated = model_from_config(accelerated_config, initialize=True).cuda()
    accelerated.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)
    return reference, accelerated


def compare_mode(chorus_enabled: bool, scope: str) -> dict:
    reference, accelerated = build_pair(chorus_enabled, scope)
    torch.manual_seed(20260804)
    base_positions = torch.randn(6, 3, device="cuda")

    ref_positions = base_positions.clone().requires_grad_(True)
    acc_positions = base_positions.clone().requires_grad_(True)
    ref_energy, ref_forces = outputs(reference, ref_positions)
    acc_energy, acc_forces = outputs(accelerated, acc_positions)
    torch.testing.assert_close(acc_energy, ref_energy, atol=3.0e-5, rtol=3.0e-5)
    torch.testing.assert_close(acc_forces, ref_forces, atol=8.0e-5, rtol=8.0e-5)

    ref_loss = ref_energy.square().mean() + ref_forces.square().mean()
    acc_loss = acc_energy.square().mean() + acc_forces.square().mean()
    ref_loss.backward()
    acc_loss.backward()
    ref_grads = dict(reference.named_parameters())
    acc_grads = dict(accelerated.named_parameters())
    compared_gradients = 0
    for name, ref_parameter in ref_grads.items():
        acc_parameter = acc_grads[name]
        if ref_parameter.grad is None and acc_parameter.grad is None:
            continue
        assert ref_parameter.grad is not None and acc_parameter.grad is not None
        torch.testing.assert_close(
            acc_parameter.grad,
            ref_parameter.grad,
            atol=3.0e-4,
            rtol=3.0e-4,
        )
        compared_gradients += 1
    assert compared_gradients > 0

    rotation = o3.rand_matrix(dtype=torch.float32, device="cuda")
    rotated_positions = (base_positions @ rotation.T).requires_grad_(True)
    rotated_energy, rotated_forces = outputs(accelerated, rotated_positions)
    torch.testing.assert_close(
        rotated_energy, acc_energy.detach(), atol=5.0e-5, rtol=5.0e-5
    )
    torch.testing.assert_close(
        rotated_forces,
        acc_forces.detach() @ rotation.T,
        atol=1.5e-4,
        rtol=1.5e-4,
    )
    return {
        "chorus": chorus_enabled,
        "scope": scope,
        "energy_max_abs": float((acc_energy - ref_energy).abs().max()),
        "force_max_abs": float((acc_forces - ref_forces).abs().max()),
        "compared_gradients": compared_gradients,
    }


def main() -> None:
    results = [
        compare_mode(False, "final"),
        compare_mode(True, "final"),
        compare_mode(True, "persistent"),
    ]
    print({"status": "passed", "results": results})


if __name__ == "__main__":
    main()
