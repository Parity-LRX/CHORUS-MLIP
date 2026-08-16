"""Numerical parity + O(3) covariance for the cuEquivariance conv-TP backend.

``ICTDResidualInteractionBlock(ictd_tp_backend='cueq')`` mirrors the reference
pytorch conv-TP (``EdgeWeightedPathPreservingTensorProduct``) with a fused
cuEquivariance ``ChannelWiseTensorProduct`` kernel plus exact per-l Q basis
bridges (ICTC <-> e3nn). These tests pin:

  * block-level forward/backward parity (edge blocks, message, all parameter
    grads incl. the authoritative ``tp.weight``) on random graphs, FP32/TF32-off;
  * the CPU fallback (cueq backend on CPU inputs runs the reference path);
  * that the CUDA path really calls the cueq kernel (reference TP is poisoned);
  * model-level energy/force parity and O(3) covariance (rotation + reflection).

Note: the chorus interaction block forward has no edge-block callback, so the
per-l edge blocks are probed by replicating the block's own conv-TP call
(``_capture_edge_blocks``); both backends are probed through their own TP path,
keeping the parity comparison honest.

Tolerances: the two backends evaluate the same algebra with different kernel
reduction orders, so FP32 agreement is at the float32-reordering level
(~1e-6 relative per TP, amplified mildly by the radial MLP gates); absolute
tolerances below are set at ~5-10x the observed maxima, documented per check.

Run:
    python -m chorus.test.test_cueq_conv_tp_backend
"""

from __future__ import annotations

import unittest

import torch

from chorus.models.ictd_irreps import direction_harmonics_all
from chorus.models.pure_cartesian_ictd_fix import ICTDResidualInteractionBlock
from chorus.models.pure_cartesian_ictd_layers import _split_irreps


def _require_cueq() -> None:
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest("cuequivariance is not installed") from exc


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available")
    return torch.device("cuda")


def _disable_tf32() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _make_block(
    backend: str,
    *,
    seed: int = 0,
    channels: int = 6,
    lmax: int = 3,
    input_lmax: int | None = None,
    conv_tp_scale_init: str = "none",
    dtype: torch.dtype = torch.float32,
) -> ICTDResidualInteractionBlock:
    torch.manual_seed(seed)
    return ICTDResidualInteractionBlock(
        channels=channels,
        lmax=lmax,
        input_lmax=input_lmax,
        number_of_basis=8,
        num_elements=3,
        conv_tp_scale_init=conv_tp_scale_init,
        internal_compute_dtype=dtype,
        ictd_tp_backend=backend,
    )


def _block_graph(
    *,
    num_nodes: int,
    num_edges: int,
    channels: int,
    input_lmax: int,
    lmax: int,
    number_of_basis: int = 8,
    num_elements: int = 3,
    seed: int = 7,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> dict:
    g = torch.Generator().manual_seed(seed)
    dim_in = channels * (input_lmax + 1) ** 2
    node_feats = 0.5 * torch.randn(num_nodes, dim_in, generator=g, dtype=torch.float32)
    dirs = torch.randn(num_edges, 3, generator=g, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y = direction_harmonics_all(dirs.to(torch.float64), lmax)
    edge_attrs = {l: y[l].to(dtype).unsqueeze(-2) for l in range(lmax + 1)}
    edge_feats = 0.5 * torch.randn(num_edges, number_of_basis, generator=g, dtype=torch.float32)
    edge_src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    loops = edge_src == edge_dst
    edge_dst[loops] = (edge_dst[loops] + 1) % num_nodes
    node_type_idx = torch.randint(0, num_elements, (num_nodes,), generator=g)
    node_attrs = torch.nn.functional.one_hot(node_type_idx, num_elements).to(torch.float32)
    return {
        "node_feats": node_feats.to(device=device, dtype=dtype),
        "node_attrs": node_attrs.to(device=device, dtype=dtype),
        "edge_attrs": {l: t.to(device) for l, t in edge_attrs.items()},
        "edge_feats": edge_feats.to(device=device, dtype=dtype),
        "edge_index": torch.stack([edge_src, edge_dst], dim=0).to(device),
        "node_type_idx": node_type_idx.to(device),
    }


def _capture_edge_blocks(block, graph) -> dict[int, torch.Tensor]:
    """Replicate the block's conv-TP call to expose the per-l edge blocks.

    Each backend is probed through its own TP path (reference projector vs the
    cueq mirror), mirroring exactly what the block forward does internally.
    """
    edge_src = graph["edge_index"][0]
    node_feats_up = block.linear_up(graph["node_feats"])
    gates = block.fc(graph["edge_feats"])
    x1 = _split_irreps(node_feats_up, block.channels, block.input_lmax)
    if getattr(block, "tp_cueq", None) is not None and node_feats_up.is_cuda:
        return block.tp_cueq(x1, edge_src, graph["edge_attrs"], gates, block.tp.weight)
    x1e = {l: x1[l][edge_src] for l in range(block.input_lmax + 1)}
    return block.tp(x1e, graph["edge_attrs"], gates)


def _run_block(block, graph, *, cotangent_seed: int = 11, backward: bool = False):
    message, sc = block(
        node_attrs=graph["node_attrs"],
        node_feats=graph["node_feats"],
        edge_attrs=graph["edge_attrs"],
        edge_feats=graph["edge_feats"],
        edge_index=graph["edge_index"],
        node_type_idx=graph["node_type_idx"],
    )
    blocks = _capture_edge_blocks(block, graph)
    if not backward:
        return message, sc, blocks
    g = torch.Generator().manual_seed(cotangent_seed)
    cot_msg = torch.randn(message.shape, generator=g, dtype=torch.float32).to(message)
    loss = (message * cot_msg).sum()
    if sc is not None:
        cot_sc = torch.randn(sc.shape, generator=g, dtype=torch.float32).to(sc)
        loss = loss + (sc * cot_sc).sum()
    for l, blk in blocks.items():
        cot = torch.randn(blk.shape, generator=g, dtype=torch.float32).to(blk)
        loss = loss + (blk * cot).sum()
    loss.backward()
    return message, sc, blocks


def _block_parity_case(
    *,
    device: torch.device,
    channels: int,
    lmax: int,
    input_lmax: int | None,
    conv_tp_scale_init: str = "none",
) -> dict[str, float]:
    """Same init, same random graph; cueq must reproduce the pytorch block."""
    in_lmax = lmax if input_lmax is None else input_lmax
    ref = _make_block(
        "pytorch", channels=channels, lmax=lmax, input_lmax=input_lmax,
        conv_tp_scale_init=conv_tp_scale_init,
    ).to(device)
    fast = _make_block(
        "cueq", channels=channels, lmax=lmax, input_lmax=input_lmax,
        conv_tp_scale_init=conv_tp_scale_init,
    ).to(device)
    # defensive: identical parameterization (cueq module adds no parameters)
    ref_sd = ref.state_dict()
    fast_sd = fast.state_dict()
    assert set(ref_sd) == set(fast_sd), (
        f"state_dict mismatch: extra={sorted(set(fast_sd) - set(ref_sd))} "
        f"missing={sorted(set(ref_sd) - set(fast_sd))}"
    )
    for key, value in ref_sd.items():
        assert torch.equal(value, fast_sd[key]), f"init mismatch at {key}"

    graph = _block_graph(
        num_nodes=9, num_edges=57, channels=channels, input_lmax=in_lmax,
        lmax=lmax, device=device,
    )
    graph["node_feats"].requires_grad_(True)
    graph["edge_feats"].requires_grad_(True)

    msg_r, _sc_r, blocks_r = _run_block(ref, graph, backward=True)
    grads_r = {
        "node_feats": graph["node_feats"].grad.clone(),
        "edge_feats": graph["edge_feats"].grad.clone(),
    }
    grads_r.update({f"param:{n}": p.grad.clone() for n, p in ref.named_parameters()})
    graph["node_feats"].grad = None
    graph["edge_feats"].grad = None

    msg_f, _sc_f, blocks_f = _run_block(fast, graph, backward=True)
    grads_f = {
        "node_feats": graph["node_feats"].grad.clone(),
        "edge_feats": graph["edge_feats"].grad.clone(),
    }
    grads_f.update({f"param:{n}": p.grad.clone() for n, p in fast.named_parameters()})

    out = {}
    for l in blocks_r:
        out[f"block_l{l}"] = float((blocks_r[l] - blocks_f[l]).abs().max().cpu())
    out["message"] = float((msg_r - msg_f).abs().max().cpu())
    assert set(grads_r) == set(grads_f)
    for key in grads_r:
        gr, gf = grads_r[key], grads_f[key]
        assert gr is not None and gf is not None, f"missing grad: {key}"
        out[f"grad:{key}"] = float((gr - gf).abs().max().cpu())
    return out


def test_cueq_conv_tp_block_parity_cuda() -> None:
    _require_cueq()
    device = _require_cuda()
    _disable_tf32()
    cases = [
        dict(channels=6, lmax=3, input_lmax=None),           # canonical multi-layer shape
        dict(channels=6, lmax=3, input_lmax=0),              # first-layer scalar input
        dict(channels=5, lmax=2, input_lmax=None, conv_tp_scale_init="e3nn"),
    ]
    for case in cases:
        errs = _block_parity_case(device=device, **case)
        worst_out = max(v for k, v in errs.items() if not k.startswith("grad:"))
        worst_grad = max(v for k, v in errs.items() if k.startswith("grad:"))
        # FP32 cross-kernel reordering: outputs are O(1-10), grads are amplified by
        # the gates/radial MLP; observed maxima are ~5e-6 / ~5e-5 respectively.
        assert worst_out < 2e-5, f"{case}: output parity {worst_out:.3e}\n{errs}"
        assert worst_grad < 5e-4, f"{case}: grad parity {worst_grad:.3e}\n{errs}"


def test_cueq_conv_tp_cpu_falls_back_to_reference() -> None:
    _require_cueq()
    _disable_tf32()
    ref = _make_block("pytorch")
    fast = _make_block("cueq")  # CPU inputs -> reference path inside the cueq block
    graph = _block_graph(num_nodes=7, num_edges=23, channels=6, input_lmax=3, lmax=3)
    msg_r, sc_r, blocks_r = _run_block(ref, graph)
    msg_f, sc_f, blocks_f = _run_block(fast, graph)
    # CPU forward is the SAME reference code path, so the outputs are bit-identical.
    for l in blocks_r:
        assert torch.equal(blocks_r[l], blocks_f[l]), f"CPU fallback l={l} not identical"
    assert torch.equal(msg_r, msg_f) and torch.equal(sc_r, sc_f)


def test_cueq_conv_tp_cuda_uses_cueq_kernel_and_reference_weight_grads() -> None:
    _require_cueq()
    device = _require_cuda()
    _disable_tf32()
    block = _make_block("cueq").to(device).train()
    graph = _block_graph(num_nodes=7, num_edges=29, channels=6, input_lmax=3, lmax=3, device=device)

    def _poisoned_reference_forward(*_args, **_kwargs):
        raise AssertionError("cueq backend fell through to the reference TP on CUDA")

    block.tp.forward = _poisoned_reference_forward
    message, _sc = block(
        node_attrs=graph["node_attrs"],
        node_feats=graph["node_feats"],
        edge_attrs=graph["edge_attrs"],
        edge_feats=graph["edge_feats"],
        edge_index=graph["edge_index"],
        node_type_idx=graph["node_type_idx"],
    )
    message.square().sum().backward()
    w_grad = block.tp.weight.grad
    assert w_grad is not None and torch.isfinite(w_grad).all() and w_grad.abs().max() > 0
    cueq_params = [n for n, _p in block.tp_cueq.named_parameters()]
    assert not cueq_params, f"cueq mirror must not own parameters, got {cueq_params}"


def _build_o3_models(dtype, device, *, conv_tp_backend_pair=("pytorch", "cueq")):
    from chorus.synthetic import build_model

    models = []
    for backend in conv_tp_backend_pair:
        torch.manual_seed(3)
        models.append(
            build_model(
                channels=8,
                lmax=2,
                num_interaction=2,
                route="baseline",
                product_backend="ictd-pure-u",
                dtype=dtype,
                device=device,
                correlation=2,
                ictd_tp_backend=backend,
            ).eval()
        )
    return models


def test_cueq_conv_tp_model_energy_force_parity_cuda() -> None:
    _require_cueq()
    device = _require_cuda()
    _disable_tf32()
    from chorus.synthetic import compute_energy_forces, make_fixed_graph

    dtype = torch.float32
    m_ref, m_fast = _build_o3_models(dtype, device)
    graph = make_fixed_graph(num_nodes=24, avg_degree=12, dtype=dtype, device=device, seed=5)
    e_r, f_r, _ = compute_energy_forces(m_ref, graph, create_graph=False)
    e_f, f_f, _ = compute_energy_forces(m_fast, graph, create_graph=False)
    d_e = float((e_r - e_f).abs().max().cpu())
    d_f = float((f_r - f_f).abs().max().cpu())
    f_scale = float(f_r.abs().max().cpu())
    # end-to-end accumulation of the FP32 reordering through 2 layers + force autodiff
    assert d_e < 1e-4, f"model energy parity {d_e:.3e}"
    assert d_f < 1e-4 * max(1.0, f_scale), f"model force parity {d_f:.3e} (scale {f_scale:.3e})"


def test_cueq_conv_tp_o3_covariance_including_reflection_cuda() -> None:
    _require_cueq()
    device = _require_cuda()
    from chorus.synthetic import compute_energy_forces, make_fixed_graph

    dtype = torch.float64
    (model,) = _build_o3_models(dtype, device, conv_tp_backend_pair=("cueq",))
    graph = make_fixed_graph(num_nodes=24, avg_degree=12, dtype=dtype, device=device, seed=9)
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph

    g = torch.Generator().manual_seed(4)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q[:, 0] = -q[:, 0]  # improper O(3) transform (reflection)
    assert torch.linalg.det(q).item() < 0.0
    q = q.to(device)

    e0, f0, _ = compute_energy_forces(model, graph, create_graph=False)
    graph_rot = (pos @ q.T, A, batch, edge_src, edge_dst, edge_shifts, cell @ q.T)
    e1, f1, _ = compute_energy_forces(model, graph_rot, create_graph=False)

    d_e = float((e1 - e0).abs().max().cpu())
    d_f = float((f1 - f0 @ q.T).abs().max().cpu())
    f_scale = float(f0.abs().max().cpu())
    # float64 cueq kernels: covariance holds to ~1e-12; leave headroom for the
    # scatter reduction order over the rotated-but-identical graph.
    assert d_e < 1e-9, f"cueq energy not O(3)-invariant: {d_e:.3e}"
    assert d_f < 1e-8 * max(1.0, f_scale), (
        f"cueq forces not O(3)-covariant: {d_f:.3e} (scale {f_scale:.3e})"
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; running CPU fallback test only")
        test_cueq_conv_tp_cpu_falls_back_to_reference()
        print("[cpu] fallback PASS")
        return
    _disable_tf32()
    device = torch.device("cuda")
    for case in (
        dict(channels=6, lmax=3, input_lmax=None),
        dict(channels=6, lmax=3, input_lmax=0),
        dict(channels=5, lmax=2, input_lmax=None, conv_tp_scale_init="e3nn"),
    ):
        errs = _block_parity_case(device=device, **case)
        worst_out = max(v for k, v in errs.items() if not k.startswith("grad:"))
        worst_grad = max(v for k, v in errs.items() if k.startswith("grad:"))
        print(f"[cuda] parity {case}: max_out={worst_out:.3e} max_grad={worst_grad:.3e}")
        for key in sorted(errs):
            print(f"    {key}: {errs[key]:.3e}")
    test_cueq_conv_tp_block_parity_cuda()
    print("[cuda] block parity PASS")
    test_cueq_conv_tp_cpu_falls_back_to_reference()
    print("[cpu] fallback PASS")
    test_cueq_conv_tp_cuda_uses_cueq_kernel_and_reference_weight_grads()
    print("[cuda] cueq-kernel usage + reference weight grads PASS")
    test_cueq_conv_tp_model_energy_force_parity_cuda()
    print("[cuda] model energy/force parity PASS")
    test_cueq_conv_tp_o3_covariance_including_reflection_cuda()
    print("[cuda] O(3) covariance incl. reflection PASS")


if __name__ == "__main__":
    main()
