#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DATASETS = ("mal", "bucky", "3bpa", "zro")
SEEDS = (20260616, 20260617, 20260618, 20260619, 20260620)


def primary_test(dataset: str) -> str:
    return "300K" if dataset == "3bpa" else "test"


def hierarchical_ci(values: dict[str, list[float]], samples: int = 20000) -> list[float]:
    rng = np.random.default_rng(20260825)
    names = tuple(values)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(names, size=len(names), replace=True)
        dataset_means = []
        for name in selected:
            local = np.asarray(values[name], dtype=np.float64)
            dataset_means.append(float(rng.choice(local, size=len(local), replace=True).mean()))
        draws[index] = float(np.mean(dataset_means))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def effect_payload(
    name: str,
    treatment: str,
    control: str,
    values: dict[str, list[float]],
) -> dict:
    dataset_effects = {
        dataset: 100.0 * (1.0 - math.exp(float(np.mean(local))))
        for dataset, local in values.items()
    }
    macro_log = float(np.mean([np.mean(local) for local in values.values()]))
    ci_log = hierarchical_ci(values)
    return {
        "contrast": name,
        "treatment": treatment,
        "control": control,
        "paired_runs": sum(map(len, values.values())),
        "wins": sum(value < 0.0 for local in values.values() for value in local),
        "dataset_relative_error_reduction_percent": dataset_effects,
        "macro_relative_error_reduction_percent": 100.0 * (1.0 - math.exp(macro_log)),
        "macro_95pct_hierarchical_bootstrap_ci_percent": [
            100.0 * (1.0 - math.exp(ci_log[1])),
            100.0 * (1.0 - math.exp(ci_log[0])),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    rows: list[dict] = []
    training_logs = {dataset: [] for dataset in DATASETS}
    aligned_probe_logs = {dataset: [] for dataset in DATASETS}
    permute_probe_logs = {dataset: [] for dataset in DATASETS}
    common_shift_max = 0.0
    native_replay_max_difference = 0.0
    dose_ratios: dict[str, list[float]] = {
        "aligned": [], "dose_0.25": [], "dose_0.50": [], "dose_0.75": [], "native": []
    }

    for dataset in DATASETS:
        test_name = primary_test(dataset)
        for seed in SEEDS:
            full_path = args.primary_root / dataset / f"seed{seed}" / "full" / "metrics.json"
            aligned_path = args.aligned_root / dataset / f"seed{seed}" / "aligned" / "metrics.json"
            probe_path = args.probe_root / dataset / f"seed{seed}" / test_name / "metrics.json"
            for path in (full_path, aligned_path, probe_path):
                if not path.is_file():
                    missing.append(str(path))
            if any(not path.is_file() for path in (full_path, aligned_path, probe_path)):
                continue
            full = json.loads(full_path.read_text())
            aligned = json.loads(aligned_path.read_text())
            probe = json.loads(probe_path.read_text())
            if full["parameter_count"] != aligned["parameter_count"]:
                raise RuntimeError(
                    f"parameter mismatch for {dataset} seed {seed}: "
                    f"full={full['parameter_count']} aligned={aligned['parameter_count']}"
                )
            full_f = float(full["tests"][test_name]["force_mae_mev_per_angstrom"])
            aligned_f = float(aligned["tests"][test_name]["force_mae_mev_per_angstrom"])
            native_probe = float(probe["metrics"]["native"]["force_mae_mev_per_angstrom"])
            aligned_probe = float(probe["metrics"]["aligned"]["force_mae_mev_per_angstrom"])
            permute_probe = float(probe["metrics"]["permute_mean"]["force_mae_mev_per_angstrom"])
            training_logs[dataset].append(math.log(full_f / aligned_f))
            aligned_probe_logs[dataset].append(math.log(native_probe / aligned_probe))
            permute_probe_logs[dataset].append(math.log(native_probe / permute_probe))
            common_shift_max = max(
                common_shift_max,
                1000.0
                * float(
                    probe["metrics"]["common_shift"][
                        "maximum_absolute_force_change_ev_per_angstrom"
                    ]
                ),
            )
            native_replay_max_difference = max(
                native_replay_max_difference, abs(native_probe - full_f)
            )
            for mode in dose_ratios:
                key = "aligned" if mode == "aligned" else mode
                if mode == "native":
                    ratio = 1.0
                else:
                    ratio = float(probe["metrics"][key]["force_mae_mev_per_angstrom"]) / native_probe
                dose_ratios[mode].append(ratio)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "parameter_count": full["parameter_count"],
                    "full_test_force_mae_mev_per_angstrom": full_f,
                    "aligned_trained_test_force_mae_mev_per_angstrom": aligned_f,
                    "native_probe_force_mae_mev_per_angstrom": native_probe,
                    "aligned_probe_force_mae_mev_per_angstrom": aligned_probe,
                    "permuted_probe_force_mae_mev_per_angstrom": permute_probe,
                }
            )

    audit = {"expected_pairs": 20, "completed_pairs": len(rows), "missing": missing}
    (args.output_dir / "completion_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if missing:
        raise SystemExit(f"phase-specific campaign incomplete: {len(missing)} missing files")
    protocol = json.loads((Path(__file__).with_name("protocol.json")).read_text())
    common_shift_tolerance = float(
        protocol["analysis"]["common_shift_tolerance_mev_per_angstrom"]
    )
    replay_tolerance = float(
        protocol["analysis"]["native_replay_force_mae_tolerance_mev_per_angstrom"]
    )
    audit.update(
        {
            "maximum_common_shift_force_change_mev_per_angstrom": common_shift_max,
            "common_shift_tolerance_mev_per_angstrom": common_shift_tolerance,
            "common_shift_pass": common_shift_max <= common_shift_tolerance,
            "maximum_native_replay_force_mae_difference_mev_per_angstrom": (
                native_replay_max_difference
            ),
            "native_replay_tolerance_mev_per_angstrom": replay_tolerance,
            "native_replay_pass": native_replay_max_difference <= replay_tolerance,
        }
    )
    (args.output_dir / "completion_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    if not audit["common_shift_pass"]:
        raise SystemExit(
            "common-shift negative control failed: "
            f"{common_shift_max:.6g} > {common_shift_tolerance:.6g} meV/A"
        )
    if not audit["native_replay_pass"]:
        raise SystemExit(
            "native checkpoint replay disagrees with the frozen Full evaluation: "
            f"{native_replay_max_difference:.6g} > {replay_tolerance:.6g} meV/A"
        )

    with (args.output_dir / "paired_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    effects = [
        effect_payload(
            "learned_phase_training_effect",
            "Full learned phase",
            "Aligned-Full trained control",
            training_logs,
        ),
        effect_payload(
            "fixed_checkpoint_relative_phase_removal",
            "Factual learned phase",
            "Aligned phase intervention",
            aligned_probe_logs,
        ),
        effect_payload(
            "fixed_checkpoint_phase_edge_assignment",
            "Factual learned assignment",
            "Within-centre phase permutation",
            permute_probe_logs,
        ),
    ]
    dose = {
        "lambda": [0.0, 0.25, 0.50, 0.75, 1.0],
        "macro_force_mae_ratio_to_factual": [
            float(np.exp(np.mean(np.log(dose_ratios[mode]))))
            for mode in ("aligned", "dose_0.25", "dose_0.50", "dose_0.75", "native")
        ],
    }
    summary = {
        "scope": (
            "Secondary phase-specific causal analysis. Same-checkpoint interventions "
            "identify controlled functional phase use; Full versus independently trained "
            "Aligned-Full identifies the end-to-end learning-algorithm effect of relative phase."
        ),
        "selection": "validation Force MAE; test never used for checkpoint selection",
        "effects": effects,
        "phase_dose_response": dose,
        "maximum_common_shift_force_change_mev_per_angstrom": common_shift_max,
        "maximum_native_replay_force_mae_difference_mev_per_angstrom": (
            native_replay_max_difference
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Phase-specific CHORUS causal analysis", "",
        "All effects preserve dataset/seed pairing. Positive reduction means the learned factual phase has lower Force MAE.", "",
        "| Estimand | Wins | Macro Force-MAE reduction | 95% hierarchical paired CI |",
        "|---|---:|---:|---:|",
    ]
    for effect in effects:
        lo, hi = effect["macro_95pct_hierarchical_bootstrap_ci_percent"]
        lines.append(
            f"| {effect['contrast']} | {effect['wins']}/{effect['paired_runs']} | "
            f"{effect['macro_relative_error_reduction_percent']:.2f}% | [{lo:.2f}%, {hi:.2f}%] |"
        )
    lines += [
        "", f"Maximum common-shift force change: {common_shift_max:.6g} meV/A.",
        "", "The fixed-checkpoint effects are controlled model-internal interventions, not estimates of a physical causal phase in the molecule. The independently trained contrast tests whether allowing relative phase improves held-out generalisation under the frozen learning protocol.",
    ]
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
