"""Write auditable NequIP configs for the extended CHORUS benchmark."""

import argparse
import math
from pathlib import Path

import yaml


SPECS = {
    "xxmd_mal": {
        "relative": "xxmd/processed_dft_temporal_r5/mal",
        "elements": ["H", "C", "O"],
        "n_train": 14000,
        "n_val": 6963,
        "batch": 16,
        "steps": 45000,
        "tests": {"test": "processed_test.h5"},
    },
    "xxmd_sti": {
        "relative": "xxmd/processed_dft_temporal_r5/sti",
        "elements": ["H", "C"],
        "n_train": 12800,
        "n_val": 6364,
        "batch": 16,
        "steps": 45000,
        "tests": {"test": "processed_test.h5"},
    },
    "xxmd_dia": {
        "relative": "xxmd/processed_dft_temporal_r5/dia",
        "elements": ["H", "C", "S"],
        "n_train": 12400,
        "n_val": 6169,
        "batch": 16,
        "steps": 45000,
        "tests": {"test": "processed_test.h5"},
    },
    "transition1x": {
        "relative": "transition1x/chorus_reaction_id_50k_seed20260616",
        "elements": ["H", "C", "N", "O"],
        "n_train": 50000,
        "n_val": 10000,
        "batch": 16,
        "steps": 100000,
        "tests": {"test": "processed_test.h5"},
        "e0": [
            -13.62222753701504,
            -1029.4130839658328,
            -1484.8710358098756,
            -2041.8396277138045,
        ],
    },
    "md22_buckyball": {
        "relative": "md22/chorus_lowdata600_20260720/processed",
        "elements": ["H", "C"],
        "n_train": 600,
        "n_val": 600,
        "batch": 4,
        "steps": 45000,
        "tests": {"test": "processed_test.h5"},
    },
    "3bpa": {
        "relative": "3bpa/standard_450_50_seed20260616_r5",
        "elements": ["H", "C", "N", "O"],
        "n_train": 450,
        "n_val": 50,
        "batch": 4,
        "steps": 45000,
        "tests": {
            "300K": "processed_test_300K.h5",
            "600K": "processed_test_600K.h5",
            "1200K": "processed_test_1200K.h5",
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--backend", choices=("e3nn", "ictc"), required=True)
    parser.add_argument("--chorus", choices=("on", "off"), required=True)
    parser.add_argument(
        "--chorus-scope",
        choices=("final", "all", "persistent"),
        default="final",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--chorus-rank", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--openequivariance", action="store_true")
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    spec = SPECS[args.dataset]
    data = args.data_root / spec["relative"]
    batches_per_epoch = math.ceil(spec["n_train"] / spec["batch"])
    epochs = (
        int(args.epochs_override)
        if args.epochs_override is not None
        else math.ceil(spec["steps"] / batches_per_epoch)
    )
    common_metrics = [
        ["forces", "mae"],
        ["forces", "rmse"],
        ["total_energy", "mae", {"PerAtom": True}],
        ["total_energy", "rmse", {"PerAtom": True}],
    ]
    channels = int(args.channels)
    if channels <= 0:
        raise ValueError("--channels must be positive")
    if int(args.chorus_rank) <= 0:
        raise ValueError("--chorus-rank must be positive")
    if int(args.num_layers) <= 0:
        raise ValueError("--num-layers must be positive")
    if int(args.lmax) < 0:
        raise ValueError("--lmax must be non-negative")
    angular_irreps = " + ".join(
        f"{ell}{'e' if ell % 2 == 0 else 'o'}"
        for ell in range(int(args.lmax) + 1)
    )
    hidden_irreps = " + ".join(
        f"{channels}x{ell}{'e' if ell % 2 == 0 else 'o'}"
        for ell in range(int(args.lmax) + 1)
    )
    config = {
        "root": str(args.run_root),
        "run_name": "run",
        "seed": int(args.seed),
        "dataset_seed": int(args.seed),
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
        "irreps_edge_sh": angular_irreps,
        "feature_irreps_hidden": hidden_irreps,
        "conv_to_output_hidden_irreps_out": f"{max(1, channels // 2)}x0e",
        "interaction_backend": args.backend,
        "openequivariance_enabled": bool(args.openequivariance),
        "chorus_enabled": args.chorus == "on",
        "chorus_scope": args.chorus_scope,
        "chorus_rank": int(args.chorus_rank),
        "chorus_hidden_channels": 32,
        "chorus_scale_init": 0.05,
        "r_max": 5.0,
        "num_layers": int(args.num_layers),
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
        "chemical_symbols": spec["elements"],
        "n_train": spec["n_train"],
        "n_val": spec["n_val"],
        "batch_size": spec["batch"],
        "validation_batch_size": 25,
        "dataloader_num_workers": 0,
        "max_epochs": epochs,
        "train_val_split": "sequential",
        "shuffle": True,
        "loss_coeffs": {
            "forces": 100.0,
            "total_energy": [1.0, "PerAtomMSELoss"],
        },
        "metrics_components": common_metrics,
        "metrics_key": "validation_f_mae",
        "optimizer_name": "Adam",
        "optimizer_amsgrad": True,
        "learning_rate": 0.003,
        "lr_scheduler_name": "ReduceLROnPlateau",
        "lr_scheduler_patience": max(5, epochs // 6),
        "lr_scheduler_factor": 0.5,
        "early_stopping_patiences": {"validation_loss": epochs + 1},
        "early_stopping_lower_bounds": {"LR": 1.0e-6},
        "early_stopping_upper_bounds": {"validation_loss": 1.0e8},
        "use_ema": False,
        "report_init_validation": True,
        "log_batch_freq": 200,
        "log_epoch_freq": 1,
        "save_checkpoint_freq": 1,
        "save_ema_checkpoint_freq": -1,
        "wandb": False,
        "verbose": "info",
        "per_species_rescale_shifts": spec.get(
            "e0", "dataset_per_atom_total_energy_mean"
        ),
        "per_species_rescale_scales": None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    for label, filename in spec["tests"].items():
        test = {
            "r_max": 5.0,
            "dataset": "MACEGraphHDF5Dataset",
            "dataset_root": str(data),
            "dataset_file_name": str(data / filename),
            "chemical_symbols": spec["elements"],
            "metrics_components": common_metrics,
            "default_dtype": "float32",
            "model_dtype": "float32",
            "allow_tf32": False,
        }
        (args.output_dir / f"test_{label}.yaml").write_text(
            yaml.safe_dump(test, sort_keys=False)
        )
    metadata = {
        "dataset": args.dataset,
        "target_steps": spec["steps"],
        "batches_per_epoch": batches_per_epoch,
        "epochs": epochs,
        "actual_step_budget": epochs * batches_per_epoch,
        "checkpoint_selection": "validation Force MAE",
        "seed": int(args.seed),
        "chorus_scope": args.chorus_scope,
        "num_layers": int(args.num_layers),
        "lmax": int(args.lmax),
        "openequivariance": bool(args.openequivariance),
    }
    (args.output_dir / "protocol.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False)
    )


if __name__ == "__main__":
    main()
