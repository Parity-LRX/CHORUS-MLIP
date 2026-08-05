"""Write one auditable NequIP-v0.6 rMD17 training/test configuration."""

import argparse
from pathlib import Path

import yaml


ELEMENTS = {
    "revised_aspirin": ["H", "C", "O"],
    "revised_ethanol": ["H", "C", "O"],
    "revised_benzene": ["H", "C"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", choices=tuple(ELEMENTS), required=True)
    parser.add_argument("--backend", choices=("e3nn", "ictc"), required=True)
    parser.add_argument("--chorus", choices=("on", "off"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--chorus-rank", type=int, default=16)
    args = parser.parse_args()

    data = args.data_root / args.molecule
    channels = int(args.channels)
    if channels <= 0:
        raise ValueError("--channels must be positive")
    if int(args.chorus_rank) <= 0:
        raise ValueError("--chorus-rank must be positive")
    config = {
        "root": str(args.run_root),
        "run_name": "run",
        "seed": 20260616,
        "dataset_seed": 20260616,
        "append": True,
        "default_dtype": "float32",
        "model_dtype": "float32",
        "allow_tf32": False,
        "model_builders": [
            "EnergyModel",
            "PerSpeciesRescale",
            "ForceOutput",
            "RescaleEnergyEtc",
        ],
        "chemical_embedding_irreps_out": f"{channels}x0e",
        "irreps_edge_sh": "0e + 1o + 2e",
        "feature_irreps_hidden": (
            f"{channels}x0e + {channels}x1o + {channels}x2e"
        ),
        "conv_to_output_hidden_irreps_out": f"{max(1, channels // 2)}x0e",
        "interaction_backend": args.backend,
        "chorus_enabled": args.chorus == "on",
        "chorus_scope": "final",
        "chorus_rank": int(args.chorus_rank),
        "chorus_hidden_channels": 32,
        "chorus_scale_init": 0.05,
        "r_max": 4.5,
        "num_layers": 3,
        "num_basis": 8,
        "BesselBasis_trainable": True,
        "PolynomialCutoff_p": 6,
        "nonlinearity_type": "gate",
        "nonlinearity_scalars": {"e": "silu", "o": "tanh"},
        "nonlinearity_gates": {"e": "silu", "o": "tanh"},
        "invariant_layers": 2,
        "invariant_neurons": 2 * channels,
        "avg_num_neighbors": "auto",
        "use_sc": True,
        "dataset": "MACEGraphHDF5Dataset",
        "dataset_root": str(data),
        "dataset_file_name": str(data / "processed_train.h5"),
        "validation_dataset": "MACEGraphHDF5Dataset",
        "validation_dataset_root": str(data),
        "validation_dataset_file_name": str(data / "processed_val.h5"),
        "chemical_symbols": ELEMENTS[args.molecule],
        "n_train": 1000,
        "n_val": 1000,
        "batch_size": 4,
        "validation_batch_size": 25,
        # Four trainers share one Slurm allocation. Forkserver workers from all
        # four processes can race on SemLock cleanup on this cluster, while
        # these 9--21 atom frames are cheap to collate in the trainer process.
        "dataloader_num_workers": 0,
        "max_epochs": int(args.epochs),
        "train_val_split": "sequential",
        "shuffle": True,
        "loss_coeffs": {
            "forces": 100.0,
            "total_energy": [1.0, "PerAtomMSELoss"],
        },
        "metrics_components": [
            ["forces", "mae"],
            ["forces", "rmse"],
            ["total_energy", "mae", {"PerAtom": True}],
            ["total_energy", "rmse", {"PerAtom": True}],
        ],
        "metrics_key": "validation_f_mae",
        "optimizer_name": "Adam",
        "optimizer_amsgrad": True,
        "learning_rate": 0.003,
        "lr_scheduler_name": "ReduceLROnPlateau",
        "lr_scheduler_patience": 30,
        "lr_scheduler_factor": 0.5,
        "early_stopping_patiences": {"validation_loss": 1000},
        "early_stopping_lower_bounds": {"LR": 1.0e-6},
        "early_stopping_upper_bounds": {"validation_loss": 1.0e6},
        "use_ema": False,
        "report_init_validation": True,
        "log_batch_freq": 100,
        "log_epoch_freq": 1,
        "save_checkpoint_freq": 5,
        "save_ema_checkpoint_freq": -1,
        "wandb": False,
        "verbose": "info",
        "per_species_rescale_shifts": "dataset_per_atom_total_energy_mean",
        "per_species_rescale_scales": "dataset_forces_rms",
    }
    test_config = {
        "r_max": 4.5,
        "dataset": "MACEGraphHDF5Dataset",
        "dataset_root": str(data),
        "dataset_file_name": str(data / "processed_test.h5"),
        "chemical_symbols": ELEMENTS[args.molecule],
        "metrics_components": config["metrics_components"],
        "default_dtype": "float32",
        "model_dtype": "float32",
        "allow_tf32": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    args.test_output.write_text(yaml.safe_dump(test_config, sort_keys=False))


if __name__ == "__main__":
    main()
