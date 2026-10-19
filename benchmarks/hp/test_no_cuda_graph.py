"""Remote-only checks of the formal training benchmark's execution guard."""
import pytest
import torch

from no_cuda_graph import forbid_cuda_graph


def test_training_guard_blocks_capture_and_replay():
    import torch._inductor.config as config
    original = torch.cuda.CUDAGraph.replay
    with forbid_cuda_graph() as policy:
        assert not policy["capture"] and not policy["replay"]
        assert config.triton.cudagraphs is False
        with pytest.raises(RuntimeError, match="forbidden"):
            torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="forbidden"):
            torch.cuda.CUDAGraph.replay(None)
        with pytest.raises(RuntimeError, match="forbidden"):
            torch.cuda.graph(None)
    assert torch.cuda.CUDAGraph.replay is original
