#!/usr/bin/env python3
"""Summarize MD17 training logs from the apple-to-apple matrix."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ICTC_RE = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\s+step\s+(?P<step>\d+)\s+\S+\]\s+"
    r"train loss=(?P<train_loss>[0-9.eE+-]+).*?\|\s+val loss=(?P<val_loss>[0-9.eE+-]+)\s+"
    r"Frmse=(?P<force_rmse>[0-9.eE+-]+)\s+Ermse=(?P<energy_rmse>[0-9.eE+-]+)"
)
MACE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+):.*?loss=(?P<val_loss>[0-9.eE+-]+),\s+"
    r"RMSE_E_per_atom=\s*(?P<energy_mev_atom>[0-9.eE+-]+)\s+meV,\s+"
    r"RMSE_F=\s*(?P<force_mev_a>[0-9.eE+-]+)\s+meV\s*/\s*A"
)
INITIAL_MACE_RE = re.compile(
    r"Initial:.*?loss=(?P<val_loss>[0-9.eE+-]+),\s+"
    r"RMSE_E_per_atom=\s*(?P<energy_mev_atom>[0-9.eE+-]+)\s+meV,\s+"
    r"RMSE_F=\s*(?P<force_mev_a>[0-9.eE+-]+)\s+meV\s*/\s*A"
)


def _job_parts(stem: str) -> tuple[str, str]:
    marker = "_seed"
    before_seed = stem.split(marker, 1)[0]
    known_modes = [
        "ictd_bridge_u_eager",
        "ictd_bridge_u_makefx",
        "ictd_cueq_makefx",
        "mace_e3nn",
        "mace_cueq",
    ]
    for mode in known_modes:
        suffix = "_" + mode
        if before_seed.endswith(suffix):
            return before_seed[: -len(suffix)], mode
    return before_seed, "unknown"


def parse_log(path: Path) -> list[dict]:
    stem = path.name
    if stem.endswith(".log"):
        stem = stem[:-4]
    if "_run-" in stem:
        return []
    dataset, mode = _job_parts(stem)
    text = path.read_text(errors="replace")
    rows: list[dict] = []
    if mode.startswith("ictd_"):
        for m in ICTC_RE.finditer(text):
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "epoch": int(m.group("epoch")),
                    "step": int(m.group("step")),
                    "val_loss": float(m.group("val_loss")),
                    "train_loss": float(m.group("train_loss")),
                    "energy_rmse_eV": float(m.group("energy_rmse")),
                    "force_rmse_eV_A": float(m.group("force_rmse")),
                    "source_log": str(path),
                }
            )
    elif mode.startswith("mace_"):
        init = INITIAL_MACE_RE.search(text)
        if init:
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "epoch": -1,
                    "step": "",
                    "val_loss": float(init.group("val_loss")),
                    "train_loss": "",
                    "energy_rmse_eV": float(init.group("energy_mev_atom")) / 1000.0,
                    "force_rmse_eV_A": float(init.group("force_mev_a")) / 1000.0,
                    "source_log": str(path),
                }
            )
        for m in MACE_RE.finditer(text):
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "epoch": int(m.group("epoch")),
                    "step": "",
                    "val_loss": float(m.group("val_loss")),
                    "train_loss": "",
                    "energy_rmse_eV": float(m.group("energy_mev_atom")) / 1000.0,
                    "force_rmse_eV_A": float(m.group("force_mev_a")) / 1000.0,
                    "source_log": str(path),
                }
            )
    deduped = []
    seen = set()
    for row in rows:
        key = (row["dataset"], row["mode"], row["epoch"], row["step"], row["val_loss"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log_dir")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows: list[dict] = []
    for path in sorted(Path(args.log_dir).glob("*.log")):
        rows.extend(parse_log(path))
    fields = [
        "dataset",
        "mode",
        "epoch",
        "step",
        "val_loss",
        "train_loss",
        "energy_rmse_eV",
        "force_rmse_eV_A",
        "source_log",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
