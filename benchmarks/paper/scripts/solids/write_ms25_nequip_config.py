#!/usr/bin/env python3
"""Write a two-layer C84 NequIP/CHORUS configuration for one MS25 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ase.data import chemical_symbols


METRICS = [
    ["forces", "mae"],
    ["forces", "rmse"],
    ["total_energy", "mae", {"PerAtom": True}],
    ["total_energy", "rmse", {"PerAtom": True}],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--size-transfer-data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("off", "final", "persistent"), required=True)
    parser.add_argument("--channels", type=int, default=84)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--chorus-density-pairs",
        choices=("full", "diagonal", "aligned"),
        default="full",
    )
    parser.add_argument(
        "--disable-openequivariance",
        action="store_true",
        help="Use the e3nn reference tensor-product implementation.",
    )
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    symbols = [chemical_symbols[z] for z in metadata["atomic_numbers"]]
    # These are periodic solids.  Let NequIP build the graph from ASE so that
    # cell shifts are retained.  The MACE graph HDF5 format used elsewhere in
    # this repository stores edge indices but not the corresponding periodic
    # image shifts, so it is not a valid NequIP input for these systems.
    extxyz_dir = args.data_dir.parent / "extxyz"
    ase_cache = args.data_dir.parent / "nequip_ase_cache"

    def ase_dataset(split: str, *, prefix: str = "dataset") -> dict[str, object]:
        return {
            prefix: "ASEDataset",
            f"{prefix}_root": str(ase_cache),
            f"{prefix}_file_name": str(extxyz_dir / f"{split}.extxyz"),
            f"{prefix}_ase_args": {"format": "extxyz"},
        }
    run_root = args.output_dir / "run"
    chorus_enabled = args.scope != "off"
    train = {
        "root": str(args.output_dir),
        "run_name": "run",
        "seed": args.seed,
        "dataset_seed": args.seed,
        "append": True,
        "default_dtype": "float32",
        "model_dtype": "float32",
        "allow_tf32": False,
        "model_builders": ["EnergyModel", "PerSpeciesRescale", "ForceOutput", "RescaleEnergyEtc"],
        "chemical_embedding_irreps_out": f"{args.channels}x0e",
        "irreps_edge_sh": "0e + 1o + 2e",
        "feature_irreps_hidden": (
            f"{args.channels}x0e + {args.channels}x1o + {args.channels}x2e"
        ),
        "conv_to_output_hidden_irreps_out": f"{args.channels // 2}x0e",
        "interaction_backend": "e3nn",
        "openequivariance_enabled": not args.disable_openequivariance,
        "chorus_enabled": chorus_enabled,
        "chorus_scope": "final" if args.scope == "off" else args.scope,
        "chorus_rank": args.rank,
        "chorus_hidden_channels": 32,
        "chorus_scale_init": 0.05,
        "chorus_density_pairs": args.chorus_density_pairs,
        "r_max": float(metadata["cutoff_angstrom"]),
        "num_layers": 2,
        "num_basis": 8,
        "BesselBasis_trainable": True,
        "PolynomialCutoff_p": 6,
        "nonlinearity_type": "gate",
        "nonlinearity_scalars": {"e": "silu", "o": "tanh"},
        "nonlinearity_gates": {"e": "silu", "o": "tanh"},
        "invariant_layers": 2,
        "invariant_neurons": 2 * args.channels,
        "avg_num_neighbors": float(metadata["average_train_neighbours"]),
        "use_sc": True,
        **ase_dataset("train"),
        **ase_dataset("val", prefix="validation_dataset"),
        "chemical_symbols": symbols,
        "n_train": 1000,
        "n_val": 80,
        "batch_size": args.batch_size,
        "validation_batch_size": args.batch_size,
        "dataloader_num_workers": 0,
        "max_epochs": args.epochs,
        "train_val_split": "sequential",
        "shuffle": True,
        "loss_coeffs": {
            "forces": 100.0,
            "total_energy": [1.0, "PerAtomMSELoss"],
        },
        "metrics_components": METRICS,
        "metrics_key": "validation_f_mae",
        "optimizer_name": "Adam",
        "optimizer_amsgrad": True,
        "learning_rate": 0.003,
        "lr_scheduler_name": "ReduceLROnPlateau",
        "lr_scheduler_patience": 8,
        "lr_scheduler_factor": 0.5,
        "early_stopping_patiences": {"validation_loss": args.epochs + 1},
        "early_stopping_lower_bounds": {"LR": 1.0e-6},
        "early_stopping_upper_bounds": {"validation_loss": 100000.0},
        "use_ema": False,
        "report_init_validation": True,
        "log_batch_freq": 200,
        "log_epoch_freq": 1,
        "save_checkpoint_freq": 1,
        "save_ema_checkpoint_freq": -1,
        "wandb": False,
        "verbose": "info",
        "per_species_rescale_shifts": "dataset_per_atom_total_energy_mean",
        "per_species_rescale_scales": None,
    }
    test = {
        "r_max": float(metadata["cutoff_angstrom"]),
        **ase_dataset("test"),
        "chemical_symbols": symbols,
        "metrics_components": METRICS,
        "default_dtype": "float32",
        "model_dtype": "float32",
        "allow_tf32": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.yaml").write_text(yaml.safe_dump(train, sort_keys=False))
    (args.output_dir / "test_test.yaml").write_text(yaml.safe_dump(test, sort_keys=False))
    if args.scope in ("off", "final", "persistent"):
        for split in ("train", "val", "test"):
            calibration = dict(test)
            calibration.update(ase_dataset(split))
            (args.output_dir / f"calibration_{split}.yaml").write_text(
                yaml.safe_dump(calibration, sort_keys=False)
            )
    if args.size_transfer_data_dir is not None:
        size_transfer = dict(test)
        transfer_extxyz = args.size_transfer_data_dir.parent / "extxyz" / "test.extxyz"
        transfer_cache = args.size_transfer_data_dir.parent / "nequip_ase_cache"
        size_transfer.update(
            {
                "dataset": "ASEDataset",
                "dataset_root": str(transfer_cache),
                "dataset_file_name": str(transfer_extxyz),
                "dataset_ase_args": {"format": "extxyz"},
            }
        )
        (args.output_dir / "test_mgo4x4.yaml").write_text(
            yaml.safe_dump(size_transfer, sort_keys=False)
        )
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "symbols": symbols,
                "scope": args.scope,
                "chorus_density_pairs": args.chorus_density_pairs,
                "openequivariance": not args.disable_openequivariance,
            }
        )
    )


if __name__ == "__main__":
    main()
