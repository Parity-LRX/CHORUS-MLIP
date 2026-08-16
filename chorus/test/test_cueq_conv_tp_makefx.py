"""CUDA smoke for ``--train-makefx-compile`` with ``conv_tp_backend='cueq'``.

Verifies that the cuEq conv-TP backend (CueqConvTensorProductSO3 inside the
interaction block) survives the make_fx training trace: tracing must either
succeed (required here, so a fallback would raise) and must not silently change
numerics — the make_fx epoch loss is compared against an identical eager run.

Run:
    python -m chorus.test.test_cueq_conv_tp_makefx
"""

from __future__ import annotations

import math
import os
import tempfile

import torch
from torch.utils.data import DataLoader

from chorus.cli.train import build_baseline_model, _avg_num_neighbors_from_h5
from chorus.data.collate import collate_fn_h5
from chorus.data.datasets import H5Dataset
from chorus.test.test_training_smoke import _make_h5
from chorus.training.train_loop import ForceTrainer
from chorus.utils.config import ModelConfig


def _build_model(cfg, avg_num_neighbors: float, device, dtype) -> torch.nn.Module:
    return build_baseline_model(
        cfg,
        avg_num_neighbors=avg_num_neighbors,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-pure-u",
        correlation=2,
        use_reduced_cg=False,
        angular_basis="e3nn",
        radial_sqrt_num_basis=False,
        edge_lmax=None,
        attn_heads=0,
        atomic_numbers=[1, 6, 7, 8],
        ictd_save_tp_mode="fully-connected",
        invariant_channels=4,
        conv_tp_backend="cueq",
        device=device,
        dtype=dtype,
    )


def _train_one_epoch(
    model, loader, cfg, avg_num_neighbors: float, device, dtype, *, makefx: bool
) -> tuple[float, ForceTrainer]:
    trainer = ForceTrainer(
        model,
        loader,
        device=device,
        config=cfg,
        dtype=dtype,
        max_radius=5.0,
        learning_rate=1e-3,
        lr_scheduler="none",
        epochs=1,
        train_makefx_compile=makefx,
        require_train_makefx_compile=makefx,
        makefx_max_slots=2,
        extra_hparams={
            "num_interaction": 2,
            "invariant_channels": 4,
            "ictd_fix_product_backend": "ictd-pure-u",
            "ictd_tp_backend": "cueq",
            "ictd_fix_use_reduced_cg": False,
            "ictd_fix_edge_lmax": int(cfg.lmax),
            "save_contraction_order": 2,
            "ictd_save_tp_mode": "fully-connected",
            "avg_num_neighbors": float(avg_num_neighbors),
            "angular_basis": "e3nn",
        },
    )
    out = trainer.train_epoch(0)
    return float(out["total_loss"]), trainer


def run_case() -> dict[str, float | int | str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuEq conv-TP make_fx training smoke")

    dtype = torch.float32
    device = torch.device("cuda")
    tmp = tempfile.mkdtemp(prefix="cueq_conv_tp_makefx_")
    train_h5 = os.path.join(tmp, "processed_train.h5")
    _make_h5(train_h5, sizes=[6, 6], seed=13)
    avg_num_neighbors = _avg_num_neighbors_from_h5(train_h5)

    dataset = H5Dataset(prefix="train", data_dir=tmp)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_h5)

    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = 4
    cfg.irreps_output_conv_channels = 4
    cfg.lmax = 1
    cfg.num_layers = 1
    cfg.max_radius = 5.0
    cfg.max_radius_main = 5.0
    cfg.function_type = "gaussian"
    cfg.internal_compute_dtype = dtype

    torch.manual_seed(29)
    model = _build_model(cfg, avg_num_neighbors, device, dtype)
    loss_makefx, trainer = _train_one_epoch(
        model, loader, cfg, avg_num_neighbors, device, dtype, makefx=True
    )
    cache_size = 0 if trainer._makefx_cache is None else len(trainer._makefx_cache._cache)
    if trainer._makefx_disabled or cache_size < 1:
        raise AssertionError("cuEq conv-TP make_fx training fell back to eager")
    meta = trainer._collect_arch_metadata()["model_hyperparameters"]
    if str(meta.get("ictd_tp_backend")) != "cueq":
        raise AssertionError(f"checkpoint metadata lost ictd_tp_backend: {meta.get('ictd_tp_backend')}")

    torch.manual_seed(29)
    model_ref = _build_model(cfg, avg_num_neighbors, device, dtype)
    loss_eager, _ = _train_one_epoch(
        model_ref, loader, cfg, avg_num_neighbors, device, dtype, makefx=False
    )

    if not (math.isfinite(loss_makefx) and math.isfinite(loss_eager)):
        raise AssertionError(f"non-finite loss: makefx={loss_makefx} eager={loss_eager}")
    diff = abs(loss_makefx - loss_eager)
    if diff > 1e-4:
        raise AssertionError(f"make_fx epoch loss diverged from eager: {diff:.3e}")
    return {
        "loss_makefx": loss_makefx,
        "loss_eager": loss_eager,
        "loss_diff": diff,
        "cache_size": int(cache_size),
        "ictd_tp_backend": str(meta["ictd_tp_backend"]),
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("cueq conv-tp makefx SKIP CUDA is not available")
        return
    result = run_case()
    print(
        "cueq conv-tp makefx PASS "
        f"backend={result['ictd_tp_backend']} "
        f"loss_makefx={result['loss_makefx']:.6g} "
        f"loss_eager={result['loss_eager']:.6g} "
        f"diff={result['loss_diff']:.3e} "
        f"cache={result['cache_size']}"
    )


if __name__ == "__main__":
    main()
