#!/usr/bin/env python3
"""Add a zero-initialized element-energy correction to a CHORUS checkpoint.

The source checkpoint is left untouched.  The added tensor is coordinate
independent, so fitting it later cannot change predicted forces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state = checkpoint["e3trans_state_dict"]
    if "element_energy_correction" in state:
        raise RuntimeError(
            f"source checkpoint already has element_energy_correction: {args.source}"
        )
    atomic_energy_keys = checkpoint["atomic_energy_keys"]
    atomic_energy_values = checkpoint["atomic_energy_values"]
    state["element_energy_correction"] = torch.zeros(
        int(atomic_energy_keys.numel()), dtype=atomic_energy_values.dtype
    )
    checkpoint["model_hyperparameters"][
        "ictd_fix_element_energy_correction"
    ] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    audit = {
        "source_checkpoint": str(args.source),
        "augmented_checkpoint": str(args.output),
        "source_sha256": sha256(args.source),
        "augmented_sha256": sha256(args.output),
        "only_added_state_tensor": "element_energy_correction",
        "initial_value": "zeros",
        "num_elements": int(atomic_energy_keys.numel()),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
