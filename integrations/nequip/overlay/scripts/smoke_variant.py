"""GPU forward/backward smoke for one NequIP/ICTC × CHORUS variant."""

import argparse

import torch

from nequip.data import AtomicDataDict
from nequip.model import model_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("e3nn", "ictc"), required=True)
    parser.add_argument("--chorus", choices=("on", "off"), required=True)
    parser.add_argument(
        "--chorus-scope",
        choices=("final", "persistent"),
        default="final",
    )
    parser.add_argument("--openequivariance", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(20260728)
    device = torch.device("cuda")
    config = {
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
        "interaction_backend": args.backend,
        "openequivariance_enabled": bool(args.openequivariance),
        "chorus_enabled": args.chorus == "on",
        "chorus_scope": args.chorus_scope,
        "chorus_rank": 4,
        "chorus_hidden_channels": 16,
        "chorus_scale_init": 0.05,
        "model_dtype": "float32",
    }
    model = model_from_config(config=config, initialize=True).to(device)
    positions = torch.randn(6, 3, device=device, requires_grad=True)
    edge_src = []
    edge_dst = []
    for dst in range(6):
        for src in range(6):
            if src != dst:
                edge_src.append(src)
                edge_dst.append(dst)
    edge_index = torch.tensor(
        [edge_dst, edge_src], dtype=torch.long, device=device
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: positions,
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.zeros(
            edge_index.shape[1], 3, device=device
        ),
        AtomicDataDict.CELL_KEY: torch.zeros(3, 3, device=device),
        AtomicDataDict.PBC_KEY: torch.zeros(3, dtype=torch.bool, device=device),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(
            [0, 1, 0, 1, 0, 1], dtype=torch.long, device=device
        ),
        AtomicDataDict.BATCH_KEY: torch.zeros(6, dtype=torch.long, device=device),
        AtomicDataDict.BATCH_PTR_KEY: torch.tensor(
            [0, 6], dtype=torch.long, device=device
        ),
    }
    output = model(data)
    energy = output[AtomicDataDict.TOTAL_ENERGY_KEY]
    forces = output[AtomicDataDict.FORCE_KEY]
    loss = energy.square().mean() + forces.square().mean()
    loss.backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()
    grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    print(
        {
            "backend": args.backend,
            "chorus": args.chorus,
            "chorus_scope": args.chorus_scope,
            "openequivariance": bool(args.openequivariance),
            "parameters": sum(p.numel() for p in model.parameters()),
            "energy": float(energy.detach().sum()),
            "force_rms": float(forces.detach().square().mean().sqrt()),
            "max_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        }
    )


if __name__ == "__main__":
    main()
