"""Run on an allocated CUDA host, never on the local document workstation."""

import copy
import pytest
import torch

from chorus.hp import check_model, configure_model
from chorus.models.pure_cartesian_ictd_fix import PhaseHermitianFullLResidual
from chorus.synthetic import build_model, compute_energy_forces, make_fixed_graph


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification")


def _build(scope="persistent", channels=8, lmax=1, layers=2):
    return build_model(
        channels=channels, lmax=lmax, num_interaction=layers, route="baseline",
        product_backend="ictd-bridge-u", dtype=torch.float32, device=torch.device("cuda"),
        correlation=2, ictd_fix_freeze_conv_tp_weight=True,
        ictd_fix_phase_mode="final-full-l-residual",
        ictd_fix_phase_amplitude="softplus", ictd_fix_phase_placement="pre-product-full-l",
        ictd_fix_phase_density_pairs="full-nonlinear", ictd_fix_phase_density_rank=4,
        ictd_fix_phase_scope=scope,
    )


def _gradient_record(model, graph):
    energy, force, atomic = compute_energy_forces(model, graph, create_graph=True)
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    loss = energy.square()*0.01 + force.square().sum()
    values = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
    return energy.detach(), force.detach(), atomic.detach(), {
        name: None if value is None else value.detach().clone()
        for (name, _), value in zip(parameters, values)}


@cuda
@pytest.mark.parametrize("form", ["hermitian", "charge2"])
@pytest.mark.parametrize("lmax", [0, 1, 2, 3])
def test_density_values_first_and_mixed_derivatives(form, lmax):
    pytest.importorskip("mace_ictc.hp")
    from chorus.hp.density import DensityPlan
    torch.manual_seed(310)
    module = PhaseHermitianFullLResidual(
        num_elements=2, channels=7, lmax=lmax, density_rank=3,
        quadratic_form=form).cuda().float()
    x = torch.randn(5, 2, 7*(lmax+1)**2, device="cuda", requires_grad=True)
    reference = module.quadratic_blocks_doublet(x)
    probe = {l: torch.randn_like(value) for l, value in reference.items()}
    ref_loss = sum((value*probe[l]).sum() for l, value in reference.items())
    ref_grad = torch.autograd.grad(ref_loss, x, create_graph=True)[0]
    tangent = torch.randn_like(x)
    ref_second = torch.autograd.grad((ref_grad*tangent).sum(), x)[0]
    object.__setattr__(module, "_hp_density", DensityPlan(module))
    got = module.quadratic_blocks_doublet(x)
    for l in reference:
        torch.testing.assert_close(got[l], reference[l], atol=1e-5, rtol=2e-5)
    loss = sum((value*probe[l]).sum() for l, value in got.items())
    gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
    second = torch.autograd.grad((gradient*tangent).sum(), x)[0]
    torch.testing.assert_close(gradient, ref_grad, atol=2e-5, rtol=3e-5)
    torch.testing.assert_close(second, ref_second, atol=2e-5, rtol=3e-5)


@cuda
@pytest.mark.parametrize("scope", ["final", "persistent"])
@pytest.mark.parametrize("stream_fusion", [False, True])
@pytest.mark.parametrize("mixed_context", [False, True])
@pytest.mark.parametrize("training_species", [False, True])
def test_whole_force_training_and_checkpoint_identity(scope, stream_fusion, mixed_context, training_species):
    pytest.importorskip("mace_ictc.hp")
    torch.manual_seed(20261017)
    model = _build(scope)
    graph = make_fixed_graph(num_nodes=17, avg_degree=5, dtype=torch.float32,
                             device=torch.device("cuda"), seed=91)
    identity = {name: (id(p), p.requires_grad) for name, p in model.named_parameters()}
    state = copy.deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    reference = _gradient_record(model, graph)
    info = configure_model(model, training=True, policy="generated", stream_fusion=stream_fusion,
                           phase_context_mixed_fusion=mixed_context,
                           training_species_fusion=training_species)
    assert info["selected"] == "generated"
    assert info["counts"]["phase_context_mixed_adjoints"] == (
        sum(layer.phase_enabled for layer in model.interactions) if mixed_context else 0)
    assert identity == {name: (id(p), p.requires_grad) for name, p in model.named_parameters()}
    assert tuple(state) == tuple(model.state_dict())
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    actual = _gradient_record(model, graph)
    for got, expected in zip(actual[:3], reference[:3]):
        torch.testing.assert_close(got, expected, atol=2e-5, rtol=5e-4)
    for name, expected in reference[3].items():
        got = actual[3][name]
        assert (got is None) == (expected is None), name
        if got is not None:
            torch.testing.assert_close(got, expected, atol=3e-5, rtol=1e-3, msg=name)
    configure_model(model, policy="reference")
    assert all(not any(name.startswith("_hp_") for name in vars(module))
               for module in model.modules())
    restored = _gradient_record(model, graph)
    for got, expected in zip(restored[:3], reference[:3]):
        # The reference CUDA scatter itself has order-dependent atomic sums;
        # check removal of the HP handles exactly, and values numerically.
        torch.testing.assert_close(got, expected, atol=1e-7, rtol=2e-6)


def test_cpu_fallback_and_explicit_failure():
    model = torch.nn.Linear(2, 1)
    assert not check_model(model)["supported"]
    assert configure_model(model)["selected"] == "reference"
    with pytest.raises(ValueError, match="not a CHORUS"):
        configure_model(model, policy="generated")


@cuda
@pytest.mark.parametrize("scope", ["final", "persistent"])
@pytest.mark.parametrize("grouped,threads", [(False, 32), (False, 128), (True, 64)])
@pytest.mark.parametrize("radial", [False, True])
def test_high_l_compact_force_training(scope, grouped, threads, radial):
    """High-l compact consumption keeps live/zero TP coefficient gradients."""
    torch.manual_seed(20261017)
    model = _build(scope, channels=8, lmax=3)
    for layer in model.interactions:
        layer.tp.weight.requires_grad_(True)
        with torch.no_grad():
            layer.tp.weight.reshape(-1)[::3] = 0.
    graph = make_fixed_graph(num_nodes=11, avg_degree=4, dtype=torch.float32,
                             device=torch.device("cuda"), seed=91)
    state = copy.deepcopy(model.state_dict())
    identity = {name: (id(p), p.requires_grad) for name, p in model.named_parameters()}
    reference = _gradient_record(model, graph)
    plan = configure_model(model, training=True, policy="generated",
        radial_coefficient_fusion=True, training_mixed_compact=True,
        training_radial_tp_fusion=radial,
        training_mixed_grouped=grouped, training_mixed_threads=threads,
        training_mixed_destination=True, training_destination_first=1)
    assert plan["training_mixed_compact"]
    assert identity == {name: (id(p), p.requires_grad) for name, p in model.named_parameters()}
    actual = _gradient_record(model, graph)
    for got, expected in zip(actual[:3], reference[:3]):
        torch.testing.assert_close(got, expected, atol=3e-5, rtol=5e-4)
    for name, expected in reference[3].items():
        got = actual[3][name]
        assert (got is None) == (expected is None), name
        if got is not None:
            torch.testing.assert_close(got, expected, atol=4e-5, rtol=2e-3, msg=name)
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)


@cuda
@pytest.mark.parametrize("higher", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_live_radial_tp_all_mixed_targets(higher, cached, monkeypatch):
    _check_live_radial(higher, cached, monkeypatch)


@cuda
@pytest.mark.parametrize('higher', [False, True])
@pytest.mark.parametrize('bounded,fused,grouped,schedule', [
    (False, True, True, 'pruned'),
    (True, True, False, 'pruned'),
    (True, True, True, 'grouped'),
    (True, True, True, 'compact_grouped'),
    (True, False, True, 'pruned'),
])
def test_live_radial_first_projected_targets(higher, bounded, fused, grouped, schedule, monkeypatch):
    _check_live_radial(higher, True, monkeypatch, bounded=bounded,
        fused=fused, grouped=grouped, schedule=schedule)


def _check_live_radial(higher, cached, monkeypatch, bounded=False, fused=False,
                       grouped=False, schedule='legacy'):
    """A live bank tangent exercises the terms absent from coordinate-only VJP."""
    from mace_ictc.hp._radial_training import _RadialTP
    from mace_ictc.hp.coherent import _outputs, _reference
    torch.manual_seed(20261018)
    model = _build("persistent", channels=32 if cached else 4, lmax=3)
    configure_model(model, training=True, policy="generated")
    owner = model.interactions[-1]._hp_message.coherent
    p0, projection = owner.edge._prepare()
    phases = owner.heads*(owner.output_lmax+1 if owner.angular else 1)
    p = {**p0, "native_packed": True, "phase_heads": owner.heads,
         "phase_angular": owner.angular, "phase_count": phases,
         "mixed_threads": 32, "mixed_compact": True, "training_destination_first": 1}
    p.update(radial_cached=cached, mixed_cache=16 if cached else 0)
    p.update(radial_first_bounded=bounded, mixed_fuse_jvp=fused,
             mixed_grouped=grouped, first_schedule=schedule)
    n, e, hidden = 5, 13, 4
    src = torch.arange(e, device="cuda", dtype=torch.int64)%n
    dst = (src+torch.arange(e, device="cuda")//n+1)%n
    mask = torch.rand(e, device="cuda")
    shapes = ((n, p["xdim"]), (e, projection["hdim"]), (e, hidden),
              (owner.edge.gate_width, hidden), (e, phases), (e, phases))
    inputs = tuple((torch.randn(shape, device="cuda")*.2).requires_grad_() for shape in shapes)
    w = owner.edge.tp.weight
    probes = tuple(torch.randn(n, 3, width//(2*l+1), 2*l+1, device="cuda", requires_grad=True)
                   for l, _, width in _outputs(p))
    tangents = tuple(torch.randn_like(value) for value in inputs)

    def run(native):
        if native:
            parts = _RadialTP.apply(*inputs, w, src, dst, mask, p, projection, 5)
        else:
            output = _reference(inputs[0], inputs[1], w,
                torch.nn.functional.linear(inputs[2], inputs[3]), *inputs[4:],
                src, dst, mask, p, projection)
            parts = []
            for l, offset, width in _outputs(p):
                value = output.reshape(n, 3, p["odim"])[..., offset:offset+width]
                value = value.reshape(n, 3, -1, 2*l+1, p["c"]).transpose(-1, -2)
                parts.append(value.reshape(n, 3, width//(2*l+1), 2*l+1))
        first = torch.autograd.grad(parts, inputs, probes, create_graph=True)
        mixed = torch.autograd.grad(sum((a*b).sum() for a, b in zip(first, tangents)),
            (*inputs, *probes), create_graph=higher)
        if higher:
            third = torch.autograd.grad(sum(value.square().sum() for value in mixed), inputs)
        else:
            third = ()
        return tuple(value.detach() for value in (*parts, *first, *mixed, *third))

    expected = run(False)
    if not higher:
        import mace_ictc.hp._radial_training as radial_module
        def forbidden_reference(*args, **kwargs):
            raise AssertionError("Force-loss radial fusion must use its native mixed adjoint")
        monkeypatch.setattr(radial_module, "_reference", forbidden_reference)
    actual = run(True)
    for index, (got, ref) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(got, ref, atol=5e-5, rtol=5e-4,
                                   msg=f"Live radial value/derivative {index}, shape={tuple(got.shape)}")


@cuda
@pytest.mark.parametrize("scope", ["final", "persistent"])
def test_parameter_target_backward_keeps_optimizer_gradients(scope):
    from chorus.hp import backward_parameters
    torch.manual_seed(20261017)
    model = _build(scope, channels=8, lmax=2)
    configure_model(model, training=True, policy="generated", training_species_fusion=True)
    graph = make_fixed_graph(num_nodes=13, avg_degree=5, dtype=torch.float32,
                             device=torch.device("cuda"), seed=81)
    recorded = []
    for targeted in (False, True):
        model.zero_grad(set_to_none=True)
        energy, force, _ = compute_energy_forces(model, graph, create_graph=True)
        loss = energy.square()*.01+force.square().sum()
        if targeted:
            backward_parameters(loss, model)
        else:
            loss.backward()
        recorded.append({name: None if value.grad is None else value.grad.detach().clone()
                         for name, value in model.named_parameters() if value.requires_grad})
    for name, ref in recorded[0].items():
        got = recorded[1][name]
        assert (got is None) == (ref is None), name
        if got is not None:
            torch.testing.assert_close(got, ref, atol=3e-5, rtol=1e-3, msg=name)


@cuda
@pytest.mark.parametrize("scope", ["final", "persistent"])
def test_live_terminal_training_keeps_all_parameter_gradients(scope):
    torch.manual_seed(20261017)
    model = _build(scope, channels=8, lmax=2, layers=3)
    graph = make_fixed_graph(num_nodes=13, avg_degree=5, dtype=torch.float32,
                             device=torch.device("cuda"), seed=81)
    reference = _gradient_record(model, graph)
    identity = {name: (id(p), p.requires_grad) for name, p in model.named_parameters()}
    plan = configure_model(model, training=True, policy="generated",
        training_species_fusion=True, training_terminal_fusion=True)
    assert plan["counts"]["terminal_projections"] == 1
    assert identity == {name: (id(p), p.requires_grad) for name, p in model.named_parameters()}
    actual = _gradient_record(model, graph)
    for got, ref in zip(actual[:3], reference[:3]):
        torch.testing.assert_close(got, ref, atol=3e-5, rtol=5e-4)
    for name, ref in reference[3].items():
        got = actual[3][name]
        assert (got is None) == (ref is None), name
        if got is not None:
            torch.testing.assert_close(got, ref, atol=4e-5, rtol=2e-3, msg=name)



@cuda
@pytest.mark.parametrize("scope", ["final", "persistent"])
@pytest.mark.parametrize("stream_fusion", [False, True])
@pytest.mark.parametrize("gate_mode", ["original", "gate", "radial"])
def test_force_training_preserves_live_tp_coefficients(scope, stream_fusion, gate_mode):
    """TP coefficients stay trainable, including exactly zero coefficients."""
    torch.manual_seed(20261017)
    model = _build(scope)
    for layer in model.interactions:
        layer.tp.weight.requires_grad_(True)
        with torch.no_grad():
            layer.tp.weight.reshape(-1)[::3] = 0.
    graph = make_fixed_graph(num_nodes=17, avg_degree=5, dtype=torch.float32,
                             device=torch.device("cuda"), seed=91)
    identity = {name: (id(p), p.requires_grad) for name, p in model.named_parameters()}
    reference = _gradient_record(model, graph)
    info = configure_model(model, training=True, policy="generated", stream_fusion=stream_fusion,
                           gate_coefficient_fusion=gate_mode == "gate",
                           radial_coefficient_fusion=gate_mode == "radial")
    assert info["counts"]["trainable_tp_gate_folds"] == len(model.interactions)
    if gate_mode == "radial":
        assert info["counts"]["radial_coefficient_fused_heads"] == len(model.interactions)
    assert identity == {name: (id(p), p.requires_grad) for name, p in model.named_parameters()}
    actual = _gradient_record(model, graph)
    for got, expected in zip(actual[:3], reference[:3]):
        torch.testing.assert_close(got, expected, atol=2e-5, rtol=5e-4)
    for name, expected in reference[3].items():
        got = actual[3][name]
        assert (got is None) == (expected is None), name
        if got is not None:
            torch.testing.assert_close(got, expected, atol=3e-5, rtol=1e-3, msg=name)
