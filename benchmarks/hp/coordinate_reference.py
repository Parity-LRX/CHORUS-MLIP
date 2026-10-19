"""Verification-only coordinate-owned dependency halos of the original model.

For each force-target block, first include every energy centre that can depend
on it, then include those centres' complete incoming K-hop halos. Thus cropped
coordinate derivatives include every contribution, unlike cropping forces of
an energy-centre-only halo. No force partials from separate runs are summed.
"""
import numpy as np
import torch


def coordinate_force_reference(model, graph, *, targets=None, centres=128):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Coordinate reference requires a frozen original local model')
    if model.long_range_module is not None or model.dispersion is not None or graph[6].shape[0] != 1:
        raise ValueError('Coordinate reference covers a single short-range structure')
    n = graph[0].shape[0]
    targets = np.arange(n, dtype=np.int64) if targets is None else np.asarray(targets, dtype=np.int64)
    if len(np.unique(targets)) != len(targets) or np.any((targets < 0) | (targets >= n)):
        raise ValueError('Force targets must be unique node indices')
    src, dst = graph[3].cpu().numpy(), graph[4].cpu().numpy()
    force = graph[0].new_empty(len(targets), 3)
    device = graph[0].device
    for start in range(0, len(targets), centres):
        target_ids = targets[start:start+centres]
        energy_centres = np.zeros(n, dtype=bool)
        energy_centres[target_ids] = True
        for _ in model.interactions:
            previous = energy_centres.copy()
            energy_centres[dst[previous[src]]] = True
        reached = energy_centres.copy()
        for _ in model.interactions:
            selected_edges = reached[dst]
            reached[src[selected_edges]] = True
        node_ids, edge_ids = np.flatnonzero(reached), np.flatnonzero(selected_edges)
        mapping = np.full(n, -1, dtype=np.int64)
        mapping[node_ids] = np.arange(len(node_ids))
        nodes, edges = (torch.as_tensor(v, device=device) for v in (node_ids, edge_ids))
        s, d = (torch.as_tensor(mapping[v[edge_ids]], device=device) for v in (src, dst))
        roots = torch.as_tensor(mapping[np.flatnonzero(energy_centres)], device=device)
        query = torch.as_tensor(mapping[target_ids], device=device)
        pos = graph[0].index_select(0, nodes).detach().requires_grad_(True)
        shifts, z = graph[5].index_select(0, edges), graph[1].index_select(0, nodes)
        batch = torch.zeros(len(node_ids), dtype=torch.long, device=device)
        vec = pos[d]-pos[s]+shifts@graph[6][0]
        value = model(pos, z, batch, s, d, shifts, graph[6], precomputed_edge_vec=vec)
        value = value[0] if isinstance(value, tuple) else value
        energy = value.reshape(-1).index_select(0, roots).sum()+pos.sum()*0.0
        derivative = torch.autograd.grad(energy, pos)[0]
        force[start:start+len(target_ids)] = -derivative.index_select(0, query)
        del value, energy, derivative, vec, pos
    return force
