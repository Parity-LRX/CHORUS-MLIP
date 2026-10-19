"""Verification-only domain decomposition of the original local checkpoint.

Each disjoint set of energy centres includes its complete K-hop dependency
halo for K interactions. All incoming edges to the (K-1)-hop halo are retained
in original order. We sum centre energies and their full coordinate/strain
derivatives, not forces cropped to the centre. This changes neither the local
potential nor its cutoff. It is never used in throughput measurements.
"""
import numpy as np
import torch


def blocked_reference(model, graph, *, virial=True, centres=128):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Blocked reference is only for frozen local-model verification")
    if model.long_range_module is not None or model.dispersion is not None or graph[6].shape[0] != 1:
        raise ValueError("Dependency-halo reference requires one short-range structure")
    n = graph[0].shape[0]
    src = graph[3].cpu().numpy()
    dst = graph[4].cpu().numpy()
    atomic = graph[0].new_empty(n, 1)
    force = torch.zeros_like(graph[0])
    tensor = torch.zeros_like(graph[6], dtype=torch.float64)
    for start in range(0, n, centres):
        stop = min(n, start+centres)
        reached = np.zeros(n, dtype=bool)
        reached[start:stop] = True
        for _ in model.interactions:
            destinations = reached.copy()
            selected_edges = destinations[dst]
            reached[src[selected_edges]] = True
        node_ids = np.flatnonzero(reached)
        edge_ids = np.flatnonzero(selected_edges)
        mapping = np.full(n, -1, dtype=np.int64)
        mapping[node_ids] = np.arange(len(node_ids))
        device = graph[0].device
        nodes = torch.as_tensor(node_ids, device=device)
        edges = torch.as_tensor(edge_ids, device=device)
        s = torch.as_tensor(mapping[src[edge_ids]], device=device)
        d = torch.as_tensor(mapping[dst[edge_ids]], device=device)
        root = torch.as_tensor(mapping[np.arange(start, stop)], device=device)
        pos = graph[0].index_select(0, nodes).detach().requires_grad_(True)
        shifts = graph[5].index_select(0, edges)
        z = graph[1].index_select(0, nodes)
        batch = torch.zeros(len(node_ids), dtype=torch.long, device=device)
        strain = torch.zeros_like(graph[6], dtype=torch.float64, requires_grad=True)
        deformation = torch.eye(3, device=device, dtype=torch.float64)+strain[0]
        vec = pos[d]-pos[s]+shifts@graph[6][0]
        edge_vec = (vec.double()@deformation).float() if virial else vec
        output = model(pos, z, batch, s, d, shifts, graph[6], precomputed_edge_vec=edge_vec)
        output = output[0] if isinstance(output, tuple) else output
        selected = output.reshape(-1, 1).index_select(0, root)
        energy = selected.sum()
        derivatives = torch.autograd.grad(energy, (pos, strain) if virial else (pos,))
        atomic[start:stop] = selected.detach()
        force.index_add_(0, nodes, -derivatives[0])
        if virial:
            tensor -= derivatives[1].detach()
        # Drop the full dependency graph before constructing the next halo.
        del output, selected, derivatives, energy, edge_vec, vec, pos, strain, deformation
    result = (atomic.sum(), force, atomic)
    return result+(tensor,) if virial else result
