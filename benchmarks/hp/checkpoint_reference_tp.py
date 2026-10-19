"""Memory-bounded original TP/SC oracle, used only outside timed training."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def checkpoint_original_tp(model, chunk_edges, chunk_nodes=0, chunk_channels=0):
    if chunk_edges < 0 or chunk_nodes < 0 or chunk_channels < 0:
        raise ValueError("Reference chunk sizes must be nonnegative")
    if not chunk_edges and not chunk_nodes and not chunk_channels:
        yield
        return

    def wrap(original):
        def forward(a, b, gates=None):
            edges = next(iter(a.values())).shape[0]
            if edges <= chunk_edges:
                return original(a, b, gates)
            parts = []
            for begin in range(0, edges, chunk_edges):
                end = min(begin+chunk_edges, edges)
                aa = {degree: value[begin:end] for degree, value in a.items()}
                bb = {degree: value[begin:end] for degree, value in b.items()}
                gg = None if gates is None else gates[begin:end]
                parts.append(checkpoint(original, aa, bb, gg,
                                        use_reentrant=False, preserve_rng_state=False))
            return {degree: torch.cat([part[degree] for part in parts], 0) for degree in parts[0]}
        return forward

    with ExitStack() as stack:
        if chunk_edges:
            for layer in model.interactions:
                stack.enter_context(patch.object(layer.tp, "forward", wrap(layer.tp.forward)))
        if chunk_nodes or chunk_channels:
            from chorus.models._mace_symmetric_contraction import MaceSymmetricContraction
            from torch.func import functional_call

            def channel_function(contraction, names):
                def evaluate(x, y, *weights):
                    return functional_call(contraction, dict(zip(names, weights)), (x, y))
                return evaluate

            def wrap_nodes(module, original):
                def node_forward(x, y):
                    if not chunk_channels or x.shape[1] <= chunk_channels:
                        return original(x, y)
                    outputs = []
                    for contraction in module.contractions:
                        named = tuple(contraction.named_parameters())
                        if any(value.shape[-1] != x.shape[1] for _, value in named):
                            raise ValueError('Original SC parameter channel axis changed')
                        evaluate = channel_function(contraction, tuple(name for name, _ in named))
                        parts = []
                        for start in range(0, x.shape[1], chunk_channels):
                            end = min(start+chunk_channels, x.shape[1])
                            # Views of the original parameters keep their full
                            # gradients; no weight/CG path is frozen or removed.
                            weights = tuple(value[..., start:end] for _, value in named)
                            parts.append(checkpoint(evaluate, x[:, start:end], y, *weights,
                                use_reentrant=False, preserve_rng_state=False))
                        outputs.append(torch.cat(parts, dim=-1))
                    return torch.cat(outputs, dim=-1)

                def forward(x, y):
                    if not chunk_nodes or x.shape[0] <= chunk_nodes:
                        return node_forward(x, y)
                    # Every original SC contraction is independent across its
                    # node axis. Recompute slices without changing U, weights,
                    # channel/path sums or any derivative target.
                    return torch.cat([checkpoint(node_forward, x[start:start+chunk_nodes],
                        y[start:start+chunk_nodes], use_reentrant=False, preserve_rng_state=False)
                        for start in range(0, x.shape[0], chunk_nodes)], dim=0)
                return forward

            for module in model.modules():
                if isinstance(module, MaceSymmetricContraction):
                    stack.enter_context(patch.object(module, "forward", wrap_nodes(module, module.forward)))
        yield
