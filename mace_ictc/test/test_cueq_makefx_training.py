"""CUDA smoke for ``--train-makefx-compile`` with ``product_backend='cueq'``.

Run:
    python -m mace_ictc.test.test_cueq_makefx_training
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch.utils.data import DataLoader

from mace_ictc.cli.train import build_baseline_model, _avg_num_neighbors_from_h5
from mace_ictc.data.collate import collate_fn_h5
from mace_ictc.data.datasets import H5Dataset
from mace_ictc.test.test_training_smoke import _make_h5
from mace_ictc.training.train_loop import ForceTrainer
from mace_ictc.utils.config import ModelConfig


def run_case(*, use_reduced_cg: bool) -> dict[str, float | int | bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuEq make_fx training smoke")

    torch.manual_seed(21 + int(use_reduced_cg))
    dtype = torch.float32
    device = torch.device("cuda")
    tmp = tempfile.mkdtemp(prefix="cueq_makefx_")
    train_h5 = os.path.join(tmp, "processed_train.h5")
    _make_h5(train_h5, sizes=[6, 6], seed=8 + int(use_reduced_cg))
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

    model = build_baseline_model(
        cfg,
        avg_num_neighbors=avg_num_neighbors,
        num_interaction=2,
        route="baseline",
        product_backend="cueq",
        correlation=2,
        use_reduced_cg=use_reduced_cg,
        angular_basis="e3nn",
        radial_sqrt_num_basis=False,
        edge_lmax=None,
        attn_heads=0,
        atomic_numbers=[1, 6, 7, 8],
        ictd_save_tp_mode="fully-connected",
        invariant_channels=4,
        device=device,
        dtype=dtype,
    )
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
        train_makefx_compile=True,
        require_train_makefx_compile=True,
        makefx_max_slots=2,
        extra_hparams={
            "num_interaction": 2,
            "invariant_channels": 4,
            "ictd_fix_product_backend": "cueq",
            "ictd_fix_use_reduced_cg": bool(use_reduced_cg),
            "ictd_fix_edge_lmax": int(cfg.lmax),
            "save_contraction_order": 2,
            "ictd_save_tp_mode": "fully-connected",
            "avg_num_neighbors": float(avg_num_neighbors),
            "angular_basis": "e3nn",
        },
    )
    out = trainer.train_epoch(0)
    cache_size = 0 if trainer._makefx_cache is None else len(trainer._makefx_cache._cache)
    if trainer._makefx_disabled or cache_size < 1:
        raise AssertionError("cuEq make_fx training fell back to eager")
    meta = trainer._collect_arch_metadata()["model_hyperparameters"]
    if model.angular_basis != "e3nn" or not getattr(model, "_e3nn_folded", False):
        raise AssertionError("cuEq make_fx training did not activate angular_basis=e3nn")
    if not bool(meta.get("angular_basis_folded_in_state_dict", False)):
        raise AssertionError("checkpoint metadata did not record folded angular_basis=e3nn state")
    ckpt = os.path.join(tmp, "model.pth")
    trainer.save_checkpoint(ckpt, epoch=0)

    from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
    loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
        ckpt,
        element_types=["H", "C", "N", "O"],
        device=device,
    ).wrapper.model
    loaded.eval()
    if loaded.angular_basis != "e3nn" or not getattr(loaded, "_e3nn_folded", False):
        raise AssertionError("from_checkpoint did not restore angular_basis=e3nn folded runtime state")
    if not getattr(loaded.products[0], "_e3nn_basis", False):
        raise AssertionError("from_checkpoint did not restore cuEq product e3nn-basis runtime flag")

    batch = trainer._unpack(next(iter(loader)))
    args = (
        batch["pos"], batch["A"], batch["batch_idx"], batch["edge_src"],
        batch["edge_dst"], batch["edge_shifts"], batch["cell"],
    )
    model.eval()
    with torch.no_grad():
        e_ref = model(*args)
        e_ref = e_ref[0] if isinstance(e_ref, tuple) else e_ref
        e_loaded = loaded(*args)
        e_loaded = e_loaded[0] if isinstance(e_loaded, tuple) else e_loaded
    reload_diff = (e_ref - e_loaded).abs().max().item()
    if reload_diff > 1e-4:
        raise AssertionError(f"from_checkpoint output diverged after e3nn reload: {reload_diff}")
    return {
        "use_reduced_cg": bool(use_reduced_cg),
        "loss": float(out["total_loss"]),
        "cache_size": int(cache_size),
        "angular_basis": str(model.angular_basis),
        "reload_diff": float(reload_diff),
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("cueq makefx training SKIP CUDA is not available")
        return
    for use_reduced_cg in (False, True):
        result = run_case(use_reduced_cg=use_reduced_cg)
        print(
            "cueq makefx training PASS "
            f"reduced={result['use_reduced_cg']} "
            f"basis={result['angular_basis']} "
            f"loss={result['loss']:.6g} "
            f"cache={result['cache_size']} "
            f"reload_diff={result['reload_diff']:.3e}"
        )


if __name__ == "__main__":
    main()
