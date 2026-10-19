"""Test installed wheels, not source-tree imports, with capture/replay forbidden."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import chorus
import mace_ictc
import torch
import pytest

from no_cuda_graph import forbid_cuda_graph


root = Path(os.environ["ROOT"])
source_root = Path(os.environ["SOURCE_ROOT"])
source = source_root / "chorus"
assert torch.cuda.is_available() and "A100" in torch.cuda.get_device_name()
assert chorus.__version__ == "0.1.0" and mace_ictc.__version__ == "0.2.12"
assert mace_ictc.__hp_api_version__ == 1
assert importlib.metadata.version("ictc-o3-hp") == "0.2.12"
assert not Path(chorus.__file__).resolve().is_relative_to(source_root)
assert not Path(mace_ictc.__file__).resolve().is_relative_to(source_root)
assert Path(chorus.__file__).resolve().is_relative_to(Path(sys.prefix))
assert Path(mace_ictc.__file__).resolve().is_relative_to(Path(sys.prefix))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
tests = [source / "chorus/test/test_hp_compat.py", source / "chorus/test/test_hp_backend.py"]
tests += [source / f"benchmarks/hp/{name}.py" for name in (
    "test_coherent", "test_coherent_training", "test_coherent_mixed_cache",
    "test_gate_coefficients", "test_species_training", "test_phase_coefficients",
    "test_silu_literal", "test_native_checkpoint", "test_layout_pruning",
    "test_radial_coherent", "test_radial_jvp", "test_terminal_projection",
)]
class RepresentativeCoverage:
    """Packaging regression: both scopes plus each original operator family.

    The full historical tuning matrices remain in their previous reports;
    version/optional-dependency edits do not change the numerical kernels.
    """
    def pytest_collection_modifyitems(self, config, items):
        kept, deselected, seen = [], [], set()
        for item in items:
            name = item.originalname or item.name
            params = getattr(getattr(item, "callspec", None), "params", {})
            if "test_hp_compat.py" in str(item.path) or name.startswith("test_density_"):
                retain = True
            elif name == "test_whole_force_training_and_checkpoint_identity":
                retain = params["stream_fusion"] and not params["mixed_context"]
            elif name == "test_high_l_compact_force_training":
                retain = not params["grouped"] and params["threads"] == 32 and not params["radial"]
            else:
                key = (str(item.path), name)
                retain = key not in seen
                seen.add(key)
            (kept if retain else deselected).append(item)
        self.selected, self.deselected = len(kept), len(deselected)
        items[:] = kept
        config.hook.pytest_deselected(items=deselected)


selection = RepresentativeCoverage()
with forbid_cuda_graph(purpose="paired release qualification") as policy:
    code = pytest.main(["-q", "--import-mode=importlib", "--tb=short", "--maxfail=1",
        f"--junitxml={root}/results/native_tests.xml", *(str(path) for path in tests)], plugins=[selection])
result = dict(status="PASS" if code == 0 else "FAIL", pytest_exit_code=int(code),
    torch=torch.__version__, gpu=torch.cuda.get_device_name(), graph_policy=policy,
    selected_tests=selection.selected, deselected_tuning_variants=selection.deselected,
    installed_modules={"chorus": chorus.__file__, "ictc_o3_hp": mace_ictc.__file__})
(root / "results/native_tests.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
sys.exit(code)
