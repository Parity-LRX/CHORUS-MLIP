"""Remote CUDA checks for the exact terminal persistent linear fold."""
import pytest
import torch
from chorus.hp import configure_model
from chorus.hp.density import DensityPlan
from chorus.hp.terminal import TerminalChargedProjection
from chorus.models.pure_cartesian_ictd_fix import PersistentChargedUpdate, PhaseHermitianFullLResidual


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="remote CUDA required")


@pytest.mark.parametrize("form", ["hermitian", "charge2"])
@pytest.mark.parametrize("lmax", [1, 2])
def test_terminal_projection_values_derivatives_and_live_weights(form, lmax):
    torch.manual_seed(741)
    update = PersistentChargedUpdate(channels=13, lmax=lmax).cuda().float().eval().requires_grad_(False)
    adapter = PhaseHermitianFullLResidual(
        num_elements=2, channels=13, lmax=lmax, density_rank=5,
        quadratic_form=form).cuda().float().eval().requires_grad_(False)
    with torch.no_grad():
        for linear in (update.previous_linear, update.incoming_linear):
            for block in linear.adapters.values():
                block.weight.normal_(std=0.2)
        update.memory_logits.uniform_(-1, 1)
    object.__setattr__(adapter, "_hp_density", DensityPlan(adapter))
    op = TerminalChargedProjection(update, adapter)
    shape = (7, 2, 13*(lmax+1)**2)
    previous = (torch.randn(shape, device="cuda")*0.2).requires_grad_()
    incoming = (torch.randn(shape, device="cuda")*0.2).requires_grad_()
    species = torch.arange(7, device="cuda") % 2
    attrs = torch.nn.functional.one_hot(species, 2).float()
    probe = torch.randn(7, 13*(lmax+1)**2, device="cuda")

    def reference():
        return adapter.forward_doublet(update.forward_doublet(previous, incoming),
                                      node_attrs=attrs, node_type_idx=species)

    for mutation in (False, True):
        if mutation:
            with torch.no_grad():
                update.memory_logits.add_(0.19)
                update.incoming_linear.adapters["1"].weight.mul_(0.9)
                adapter.rank_projections["0"].weight.add_(0.017)
        expected = reference()
        actual = op(previous, incoming, node_attrs=attrs, node_type_idx=species)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=6e-5)
        ref_grad = torch.autograd.grad((expected*probe).sum(), (previous, incoming), create_graph=True)
        got_grad = torch.autograd.grad((actual*probe).sum(), (previous, incoming), create_graph=True)
        for got, ref in zip(got_grad, ref_grad):
            torch.testing.assert_close(got, ref, atol=2e-6, rtol=6e-5)
        direction = tuple(torch.randn_like(value) for value in (previous, incoming))
        ref_second = torch.autograd.grad(sum((g*v).sum() for g, v in zip(ref_grad, direction)),
                                         (previous, incoming))
        got_second = torch.autograd.grad(sum((g*v).sum() for g, v in zip(got_grad, direction)),
                                         (previous, incoming))
        for got, ref in zip(got_second, ref_second):
            torch.testing.assert_close(got, ref, atol=3e-6, rtol=8e-5)
    update.memory_logits.requires_grad_(True)
    assert op(previous, incoming, node_attrs=attrs, node_type_idx=species) is None
    update.memory_logits.requires_grad_(False)
    adapter.train()
    assert op(previous, incoming, node_attrs=attrs, node_type_idx=species) is None


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_terminal_projection_whole_model(layers):
    from chorus.test.test_hp_backend import _build
    from chorus.synthetic import compute_energy_forces, make_fixed_graph
    torch.manual_seed(618)
    model = _build("final" if layers == 1 else "persistent", layers=layers).eval().requires_grad_(False)
    graph = make_fixed_graph(num_nodes=13, avg_degree=5, dtype=torch.float32,
                             device=torch.device("cuda"), seed=81)
    reference = compute_energy_forces(model, graph, create_graph=False)
    info = configure_model(model, policy="generated", topology_backend="device")
    assert info["counts"]["terminal_projections"] == int(layers > 1)
    actual = compute_energy_forces(model, graph, create_graph=False)
    for got, ref in zip(actual, reference):
        torch.testing.assert_close(got, ref, atol=2e-5, rtol=5e-4)
    configure_model(model, policy="reference")
    assert all("_hp_terminal_phase" not in vars(module) for module in model.modules())
