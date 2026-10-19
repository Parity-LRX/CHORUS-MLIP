"""Remote CUDA qualification of atom-owned FP32 force reduction and replay."""
import pytest
import torch
from chorus.hp.edge_geometry import gather_forces

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='remote CUDA required')


@pytest.mark.parametrize('nodes,edges', [(11, 47), (11, 517), (1, 105), (3, 0)])
def test_force_gather_live_inputs(nodes, edges):
    torch.manual_seed(20261017)
    positions = torch.randn(nodes, 3, device='cuda')
    derivative = torch.randn(edges, 3, device='cuda')*0.1
    src, dst = (torch.randint(nodes, (edges,), device='cuda') for _ in range(2))

    def reference():
        return torch.zeros_like(positions).index_add(0, src, derivative).index_add(0, dst, -derivative)

    for _ in range(3):gather_forces(derivative, src, dst, positions)
    torch.testing.assert_close(gather_forces(derivative, src, dst, positions), reference(), atol=3e-6, rtol=5e-5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):gather_forces(derivative, src, dst, positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured = gather_forces(derivative, src, dst, positions)
    derivative.add_(0.017)
    if nodes > 1:
        src.add_(1).remainder_(nodes)
        dst.add_(2).remainder_(nodes)
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=3e-6, rtol=5e-5)
    assert captured.dtype == torch.float32
