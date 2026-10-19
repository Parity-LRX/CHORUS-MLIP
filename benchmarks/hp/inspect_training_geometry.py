"""Read-only remote batch audit for the large-graph/training benchmarks."""
import itertools
import json
from pathlib import Path

import torch


def main():
    offsets = [v for v in itertools.product(range(-4, 5), repeat=3) if any(v)]
    candidates = []
    for scales in ((1., 1.08, 1.18), (1., 1.1, 1.3), (1., 1.2, 1.4),
                   (1., 1.07, 1.31), (1., 1.17, 1.43)):
        radii = sorted(sum((a*b)**2 for a, b in zip(v, scales)) for v in offsets)
        if radii[49] < radii[50]-1e-8:
            candidates.append({"axis_ratios": scales, "r50_squared": radii[49],
                               "r51_squared": radii[50],
                               "radius": ((radii[49]+radii[50])/2)**0.5})
    checkpoint = Path("/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print(json.dumps({"degree50_candidates": candidates, "checkpoint_keys": list(raw)}, indent=2))
    for key, value in raw.items():
        if isinstance(value, dict) and key not in {"model", "state_dict", "optimizer", "e3trans_state_dict", "optimizer_state_dict"}:
            selected = {k: v for k, v in value.items()
                        if any(word in k.lower() for word in ("freeze", "phase", "channel", "lmax", "correlation", "interaction"))}
            print(key, repr(selected), flush=True)
        elif "freeze" in key or "arch" in key:
            print(key, repr(value), flush=True)


if __name__ == "__main__":
    main()
