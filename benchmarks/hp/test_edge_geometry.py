"""Remote checks of periodic geometry and combined force scatter."""
import pytest
import torch
from chorus.hp.edge_geometry import edge_vectors, scatter_forces

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='remote CUDA required')


@pytest.mark.parametrize('edges', [0, 53, 1024])
def test_triclinic_multistructure_geometry_and_scatter(edges):
    torch.manual_seed(20261017)
    n = 29
    positions = torch.randn(n, 3, device='cuda')
    batch = torch.arange(n, device='cuda') % 3
    cell = torch.randn(3, 3, 3, device='cuda')+torch.eye(3, device='cuda')[None]*5
    src, dst = torch.randint(n, (edges,), device='cuda'), torch.randint(n, (edges,), device='cuda')
    shifts = torch.randint(-2, 3, (edges, 3), device='cuda').float()
    expected = positions[dst]-positions[src]+torch.einsum('ei,eij->ej', shifts, cell[batch[src]])
    actual = edge_vectors(positions, batch, src, dst, shifts, cell)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-6)
    derivative = torch.randn(edges, 3, device='cuda')
    expected_force = torch.zeros_like(positions).index_add(0, src, derivative).index_add(0, dst, -derivative)
    actual_force = scatter_forces(derivative, src, dst, positions)
    torch.testing.assert_close(actual_force, expected_force, atol=2e-5, rtol=1e-5)
