"""Compare old/new reference E/F and force-loss gradients with HP unavailable."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
from unittest.mock import patch


class NoHP(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "mace_ictc":
            raise ModuleNotFoundError("HP deliberately unavailable in reference qualification", name=fullname)
        return None


sys.meta_path.insert(0, NoHP())
import torch
import chorus
from chorus.synthetic import build_model, compute_energy_forces, make_fixed_graph


def record(model, graph):
    energy, force, atomic = compute_energy_forces(model, graph, create_graph=True)
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    loss = energy.square().sum() * 0.01 + force.square().sum()
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    return dict(energy=energy.detach(), force=force.detach(), atomic=atomic.detach(),
        gradients={name: None if value is None else value.detach()
                   for name, value in zip(parameters, gradients)})


parser = argparse.ArgumentParser()
parser.add_argument("--oracle", type=Path, required=True)
parser.add_argument("--write", action="store_true")
parser.add_argument("--report", type=Path)
args = parser.parse_args()
torch.set_num_threads(1)
oracle = {} if args.write else torch.load(args.oracle, map_location="cpu", weights_only=False)
results = {}
for scope in ("final", "persistent"):
    torch.manual_seed(20261019)
    model = build_model(channels=8, lmax=1, num_interaction=2, route="baseline",
        product_backend="ictd-bridge-u", dtype=torch.float32, device=torch.device("cpu"),
        correlation=2, ictd_fix_phase_mode="final-full-l-residual",
        ictd_fix_phase_amplitude="softplus", ictd_fix_phase_placement="pre-product-full-l",
        ictd_fix_phase_density_pairs="full-nonlinear", ictd_fix_phase_density_rank=4,
        ictd_fix_phase_scope=scope)
    graph = make_fixed_graph(num_nodes=11, avg_degree=4, dtype=torch.float32,
        device=torch.device("cpu"), seed=91)
    if not args.write:
        model.load_state_dict(oracle[scope]["state"], strict=True)
    actual = record(model, graph)
    if args.write:
        oracle[scope] = dict(state=model.state_dict(), outputs=actual)
    else:
        expected = oracle[scope]["outputs"]
        for field in ("energy", "force", "atomic"):
            torch.testing.assert_close(actual[field], expected[field], atol=2e-6, rtol=2e-5)
        assert actual["gradients"].keys() == expected["gradients"].keys()
        for name, gradient in actual["gradients"].items():
            previous = expected["gradients"][name]
            assert (gradient is None) == (previous is None), name
            if gradient is not None:
                torch.testing.assert_close(gradient, previous, atol=3e-6, rtol=3e-5, msg=name)
        from chorus.hp import configure_model
        from chorus.hp import _compat
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with patch.object(_compat, "_version", side_effect=_compat.PackageNotFoundError("ictc-o3-hp")):
            assert configure_model(model, policy="reference")["selected"] == "reference"
            fallback = configure_model(model, policy="auto")
            assert fallback["selected"] == "reference"
            assert any("not installed" in reason for reason in fallback["reasons"])
            try:
                configure_model(model, policy="generated")
            except ValueError as error:
                assert "not installed" in str(error)
            else:
                raise AssertionError("generated must fail when the backend is absent")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        restored = record(model, graph)
        torch.testing.assert_close(restored["force"], actual["force"], atol=0, rtol=0)
        results[scope] = dict(energy_force_and_parameter_gradients="PASS",
            missing_hp_reference_auto_generated="PASS", checkpoint_unchanged=True,
            force_max_abs_error=float((actual["force"] - expected["force"]).abs().max()))
assert not any(name == "mace_ictc" or name.startswith("mace_ictc.") for name in sys.modules)
if args.write:
    torch.save(oracle, args.oracle)
else:
    assert chorus.__version__ == "0.1.0"
    result = dict(status="PASS", version=chorus.__version__, module=chorus.__file__,
        hp_imported=False, reference_comparison="pre-HP committed CHORUS implementation", scopes=results)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
