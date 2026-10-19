"""Fail-closed CUDA Graph prohibition for numerical qualification.

Use this in the remote numerical process, including DPA/Inductor baselines.
It is not an implementation of a training execution mode.
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch


@contextmanager
def forbid_cuda_graph(*, purpose="training"):
    """Disable compiler graphs and reject capture/replay through public APIs."""
    import torch._inductor.config as config

    def prohibited(*args, **kwargs):
        raise RuntimeError(f"CUDA Graph capture/replay is forbidden in this {purpose} benchmark")

    with ExitStack() as stack:
        stack.enter_context(config.patch({"triton.cudagraphs": False}))
        graph_type = torch.cuda.CUDAGraph
        for name in ("__init__", "capture_begin", "capture_end", "replay"):
            stack.enter_context(patch.object(graph_type, name, prohibited))
        stack.enter_context(patch.object(torch.cuda, "graph", prohibited))
        stack.enter_context(patch.object(torch.cuda, "make_graphed_callables", prohibited))
        yield {"capture": False, "replay": False, "inductor_cudagraphs": False,
               "prohibition": "runtime API guard plus Inductor configuration"}
