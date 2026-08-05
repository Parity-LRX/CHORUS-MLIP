"""Fit and evaluate a train-only elemental energy residual for Transition1x."""

import argparse
import json
from pathlib import Path

import torch

from nequip.data import AtomicData, Collater, dataset_from_config
from nequip.scripts.train import default_config
from nequip.train import Trainer
from nequip.utils import Config
from nequip.utils._global_options import _set_global_options


def predictions(model, dataset, device, num_types, batch_size=50):
    collater = Collater.for_dataset(dataset, exclude_keys=[])
    all_counts, all_ref, all_pred, all_atoms = [], [], [], []
    for start in range(0, len(dataset), batch_size):
        batch = collater.collate(
            [dataset[index] for index in range(start, min(len(dataset), start + batch_size))]
        ).to(device)
        out = model(AtomicData.to_AtomicDataDict(batch))
        graph_index = batch.batch.reshape(-1).long()
        atom_types = batch.atom_types.reshape(-1).long()
        counts = torch.zeros(
            batch.num_graphs, num_types, dtype=torch.float64, device=device
        )
        counts.index_put_(
            (graph_index, atom_types),
            torch.ones_like(graph_index, dtype=torch.float64),
            accumulate=True,
        )
        all_counts.append(counts.cpu())
        all_ref.append(batch.total_energy.detach().reshape(-1).double().cpu())
        all_pred.append(out["total_energy"].detach().reshape(-1).double().cpu())
        all_atoms.append(
            torch.bincount(graph_index, minlength=batch.num_graphs).double().cpu()
        )
    return tuple(map(torch.cat, (all_counts, all_ref, all_pred, all_atoms)))


def energy_metrics(counts, ref, pred, atoms, correction):
    error = (pred + counts @ correction - ref) / atoms
    return {
        "energy_mae_mev_per_atom": float(error.abs().mean() * 1000.0),
        "energy_rmse_mev_per_atom": float(torch.sqrt((error**2).mean()) * 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--test-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = Config.from_file(str(args.train_config), defaults=default_config)
    _set_global_options(config)
    model, _ = Trainer.load_model_from_training_session(
        args.train_dir, model_name="best_model.pth", device=args.device
    )
    model.eval()
    device = torch.device(args.device)
    num_types = len(config["chemical_symbols"])

    train = dataset_from_config(config, prefix="dataset")
    validation = dataset_from_config(config, prefix="validation_dataset")
    test_config = Config.from_file(str(args.test_config), defaults=default_config)
    test = dataset_from_config(test_config, prefix="dataset")

    train_data = predictions(model, train, device, num_types)
    correction = torch.linalg.lstsq(
        train_data[0], train_data[1] - train_data[2]
    ).solution
    val_data = predictions(model, validation, device, num_types)
    test_data = predictions(model, test, device, num_types)
    result = {
        "selection": "best_model.pth selected by validation Force MAE",
        "fit_data": "training split only",
        "chemical_symbols": list(config["chemical_symbols"]),
        "element_energy_residual_ev": correction.tolist(),
        "train_raw": energy_metrics(*train_data, torch.zeros_like(correction)),
        "train_corrected": energy_metrics(*train_data, correction),
        "validation_raw": energy_metrics(*val_data, torch.zeros_like(correction)),
        "validation_corrected": energy_metrics(*val_data, correction),
        "test_raw": energy_metrics(*test_data, torch.zeros_like(correction)),
        "test_corrected": energy_metrics(*test_data, correction),
        "forces_unchanged": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
