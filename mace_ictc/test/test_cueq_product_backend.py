"""Numerical parity for the cuEquivariance product backend.

Run:
    python -m mace_ictc.test.test_cueq_product_backend
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from mace_ictc.models.pure_cartesian_ictd_fix import CueqMaceSymmetricContractionSO3


def _require_cueq() -> None:
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest("cuequivariance is not installed") from exc


def _fast_cueq_contraction(
    module: CueqMaceSymmetricContractionSO3,
    x: torch.Tensor,
    node_type_idx: torch.Tensor,
) -> torch.Tensor:
    module.refresh_cueq_weights()
    flat = x.transpose(1, 2).reshape(x.shape[0], -1)
    idx = node_type_idx.to(device=x.device, dtype=torch.int32)
    outs = [contraction(flat, idx) for contraction in module.cueq_contractions]
    return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]


def _check_case(
    *,
    correlation: int,
    lmax: int,
    target_lmax: int,
    device: torch.device,
    use_reduced_cg: bool = False,
) -> dict[str, float]:
    _require_cueq()
    if device.type == "cpu" and torch.cuda.is_available():
        raise unittest.SkipTest(
            "when CUDA is available, CueqMaceSymmetricContractionSO3 builds CUDA-only fast kernels; "
            "CPU forward intentionally falls back to the reference path"
        )
    torch.manual_seed(1000 + 100 * correlation + 10 * lmax + target_lmax)
    torch.set_default_dtype(torch.float32)

    num_elements = 3
    channels = 3
    num_nodes = 7
    try:
        module = CueqMaceSymmetricContractionSO3(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=target_lmax,
            correlation=correlation,
            use_reduced_cg=use_reduced_cg,
        ).to(device).eval()
    except RuntimeError as exc:
        if use_reduced_cg and "with reduced CG requires" in str(exc):
            raise unittest.SkipTest("local reduced-CG cueq projection helper is not available") from exc
        raise

    # Keep the random input moderate; corr=4 amplifies large values quickly.
    x = 0.1 * torch.randn(
        num_nodes,
        channels,
        sum(2 * ell + 1 for ell in range(lmax + 1)),
        dtype=torch.float32,
        device=device,
    )
    node_type_idx = (torch.arange(num_nodes, device=device, dtype=torch.long) % num_elements)
    node_attrs = F.one_hot(node_type_idx, num_classes=num_elements).to(dtype=x.dtype)

    ref = module.symmetric_contractions(x, node_attrs)
    fast = _fast_cueq_contraction(module, x, node_type_idx)
    max_abs = float((ref - fast).abs().max().detach().cpu())
    max_ref = float(ref.abs().max().detach().cpu())
    rel = max_abs / max(max_ref, 1e-12)
    assert max_abs < 2e-5, (
        f"corr={correlation} lmax={lmax} target_lmax={target_lmax} "
        f"use_reduced_cg={use_reduced_cg} cuEq product mismatch: max_abs={max_abs:.3e} rel={rel:.3e}"
    )
    return {"max_abs": max_abs, "rel": rel}


def test_cueq_product_correlations_1_2_4_match_reference_cpu() -> None:
    for correlation, lmax, target_lmax in ((1, 2, 1), (2, 2, 1), (4, 1, 1), (4, 2, 0)):
        _check_case(
            correlation=correlation,
            lmax=lmax,
            target_lmax=target_lmax,
            device=torch.device("cpu"),
        )


def test_cueq_product_correlations_1_2_4_match_reference_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")
    for correlation, lmax, target_lmax in ((1, 2, 1), (2, 2, 1), (4, 1, 1), (4, 2, 0)):
        _check_case(
            correlation=correlation,
            lmax=lmax,
            target_lmax=target_lmax,
            device=torch.device("cuda"),
        )


def test_cueq_product_reduced_cg_correlations_2_4_match_reference() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for correlation, lmax, target_lmax in ((2, 2, 1), (4, 1, 1)):
        _check_case(
            correlation=correlation,
            lmax=lmax,
            target_lmax=target_lmax,
            device=device,
            use_reduced_cg=True,
        )


def test_cueq_product_training_cuda_uses_cueq_with_reference_weight_grads() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")
    _require_cueq()

    for use_reduced_cg in (False, True):
        torch.manual_seed(4321 + int(use_reduced_cg))
        torch.set_default_dtype(torch.float32)
        module = CueqMaceSymmetricContractionSO3(
            num_elements=3,
            channels=3,
            lmax=2,
            target_lmax=1,
            correlation=2,
            use_reduced_cg=use_reduced_cg,
        ).cuda().train()

        x = 0.1 * torch.randn(
            7,
            3,
            sum(2 * ell + 1 for ell in range(3)),
            device="cuda",
            requires_grad=True,
        )
        node_type_idx = torch.arange(7, device="cuda", dtype=torch.long) % 3
        node_attrs = F.one_hot(node_type_idx, num_classes=3).to(dtype=x.dtype)
        ref = module.symmetric_contractions(x, node_attrs).detach()

        def _reference_forward_disabled(*_args, **_kwargs):
            raise AssertionError("training cueq path called reference forward")

        module.symmetric_contractions.forward = _reference_forward_disabled
        out = module(x, node_attrs, node_type_idx=node_type_idx)
        max_abs = float((out - ref).abs().max().detach().cpu())
        assert max_abs < 2e-5, f"training cuEq output mismatch: {max_abs:.3e}"

        out.square().sum().backward()
        ref_grads = [
            param.grad
            for name, param in module.named_parameters()
            if name.startswith("symmetric_contractions.")
        ]
        fast_grads = [
            param.grad
            for name, param in module.named_parameters()
            if name.startswith("cueq_contractions.")
        ]
        assert any(grad is not None and torch.isfinite(grad).all() for grad in ref_grads)
        assert all(grad is None for grad in fast_grads)


def test_cueq_product_eval_refreshes_after_training_weight_update() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")
    _require_cueq()

    torch.manual_seed(5432)
    torch.set_default_dtype(torch.float32)
    module = CueqMaceSymmetricContractionSO3(
        num_elements=3,
        channels=3,
        lmax=2,
        target_lmax=1,
        correlation=2,
        use_reduced_cg=False,
    ).cuda().train()
    x = 0.1 * torch.randn(7, 3, sum(2 * ell + 1 for ell in range(3)), device="cuda")
    node_type_idx = torch.arange(7, device="cuda", dtype=torch.long) % 3
    node_attrs = F.one_hot(node_type_idx, num_classes=3).to(dtype=x.dtype)

    with torch.no_grad():
        for contraction in module.symmetric_contractions.contractions:
            contraction.weights_max.add_(0.123)
    ref = module.symmetric_contractions(x, node_attrs)
    module.eval()
    out = module(x, node_attrs, node_type_idx=node_type_idx)
    max_abs = float((out - ref).abs().max().detach().cpu())
    assert max_abs < 2e-5, f"eval cuEq mirror was not refreshed after training update: {max_abs:.3e}"


def main() -> None:
    device_cases = [torch.device("cuda")] if torch.cuda.is_available() else [torch.device("cpu")]
    for device in device_cases:
        for label, cases, use_reduced_cg in (
            ("full", ((1, 2, 1), (2, 2, 1), (4, 1, 1), (4, 2, 0)), False),
            ("reduced", ((2, 2, 1), (4, 1, 1)), True),
        ):
            for correlation, lmax, target_lmax in cases:
                try:
                    result = _check_case(
                        correlation=correlation,
                        lmax=lmax,
                        target_lmax=target_lmax,
                        device=device,
                        use_reduced_cg=use_reduced_cg,
                    )
                except unittest.SkipTest as exc:
                    print(
                        f"[{device.type}] {label} corr={correlation} lmax={lmax} "
                        f"target_lmax={target_lmax} SKIP {exc}"
                    )
                    continue
                print(
                    f"[{device.type}] {label} corr={correlation} lmax={lmax} target_lmax={target_lmax} "
                    f"max_abs={result['max_abs']:.3e} rel={result['rel']:.3e}"
                )
    if torch.cuda.is_available():
        test_cueq_product_training_cuda_uses_cueq_with_reference_weight_grads()
        print("[cuda] training cuEq path uses reference-weight grads PASS")
        test_cueq_product_eval_refreshes_after_training_weight_update()
        print("[cuda] eval refresh after training weight update PASS")


if __name__ == "__main__":
    main()
