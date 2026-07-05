"""Validate the deployment-only preserve_edge_order fast path."""

from __future__ import annotations

import torch

from mace_ictc.synthetic import build_model, compute_energy_forces, make_fixed_graph


def _sort_graph_by_dst(graph):
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    order = torch.argsort(edge_dst)
    return (
        pos,
        A,
        batch,
        edge_src[order],
        edge_dst[order],
        edge_shifts[order],
        cell,
    )


def test_preserve_edge_order_matches_internal_sort_when_caller_presorts():
    dtype = torch.float64
    device = torch.device("cpu")
    torch.manual_seed(17)
    model = build_model(
        channels=8,
        lmax=2,
        num_interaction=2,
        route="baseline",
        product_backend="ictd-bridge-u",
        dtype=dtype,
        device=device,
        correlation=2,
    ).eval()
    graph = make_fixed_graph(
        num_nodes=14,
        avg_degree=8,
        dtype=dtype,
        device=device,
        seed=23,
    )

    model.preserve_edge_order = False
    e_sort, f_sort, e_atom_sort = compute_energy_forces(model, graph, create_graph=False)

    model.preserve_edge_order = True
    sorted_graph = _sort_graph_by_dst(graph)
    e_keep, f_keep, e_atom_keep = compute_energy_forces(model, sorted_graph, create_graph=False)

    assert torch.equal(sorted_graph[4], torch.sort(sorted_graph[4]).values)
    assert (e_sort - e_keep).abs().item() < 1e-12
    assert (e_atom_sort - e_atom_keep).abs().max().item() < 1e-12
    assert (f_sort - f_keep).abs().max().item() < 1e-12
