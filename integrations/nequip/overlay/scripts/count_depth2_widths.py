"""Count two-layer NequIP/CHORUS parameters across candidate widths.

This utility builds models from finalized two- and four-species training
configs, so the reported interval matches the parameter-range convention used
in the manuscript tables.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from nequip.model import model_from_config


def configure(base: dict, width: int, mode: str) -> dict:
    config = copy.deepcopy(base)
    # The two rescaling builders carry no trainable weights but require an
    # initialized dataset when their statistics are symbolic.  Omitting them
    # makes this a dataset-free architecture count without changing the number
    # of trainable parameters.
    config["model_builders"] = ["EnergyModel", "ForceOutput"]
    config["chemical_embedding_irreps_out"] = f"{width}x0e"
    config["feature_irreps_hidden"] = (
        f"{width}x0e + {width}x1o + {width}x2e"
    )
    config["conv_to_output_hidden_irreps_out"] = f"{max(1, width // 2)}x0e"
    config["invariant_neurons"] = 2 * width
    config["num_layers"] = 2
    config["chorus_enabled"] = mode != "off"
    config["chorus_scope"] = "persistent" if mode == "persistent" else "final"
    config["chorus_rank"] = 16
    return config


def count(config: dict) -> int:
    model = model_from_config(config=config, initialize=True)
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-species-config", type=Path, required=True)
    parser.add_argument("--four-species-config", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # These are trusted, finalized NequIP configs produced by our own runs.
    # NequIP records a few runtime values with Python-specific YAML tags, so a
    # SafeLoader cannot reconstruct them even though they are valid inputs to
    # ``model_from_config``.
    bases = {
        "two_species": yaml.unsafe_load(args.two_species_config.read_text()),
        "four_species": yaml.unsafe_load(args.four_species_config.read_text()),
    }
    records = []
    for width in args.widths:
        for mode in ("off", "final", "persistent"):
            counts = {
                label: count(configure(base, width, mode))
                for label, base in bases.items()
            }
            records.append(
                {
                    "width": width,
                    "mode": mode,
                    **counts,
                    "minimum": min(counts.values()),
                    "maximum": max(counts.values()),
                }
            )
    payload = {
        "num_layers": 2,
        "chorus_rank": 16,
        "widths": args.widths,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
