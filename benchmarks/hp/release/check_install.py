"""Verify optional resolution and distinguish inherited environment problems."""
import json
import os
from pathlib import Path
import subprocess


root = Path(os.environ["ROOT"])
reports = {name: json.loads((root / f"results/install_{name}.json").read_text())
           for name in ("reference", "hp")}
installed = {name: {item["metadata"]["name"].lower().replace("_", "-"):
                        item["metadata"]["version"] for item in report["install"]}
             for name, report in reports.items()}
assert "ictc-o3-hp" not in installed["reference"]
assert installed["hp"]["ictc-o3-hp"] == "0.2.12", installed
assert installed["hp"]["chorus-mlip"] == "0.1.0", installed
hp_source = next(item["download_info"] for item in reports["hp"]["install"]
                 if item["metadata"]["name"].lower().replace("_", "-") == "ictc-o3-hp")
assert hp_source["vcs_info"]["commit_id"] == "9fdea47eef620013e9ea591f7d90534bb27671c8"
checks = {}
for name, executable in (("base", "/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python"),
                         ("hp", os.environ["HP_PYTHON"])):
    result = subprocess.run([executable, "-m", "pip", "check"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root / f"results/pip_check_{name}.log").write_text(result.stdout)
    checks[name] = set(result.stdout.splitlines()) if result.returncode else set()
new_problems = sorted(checks["hp"] - checks["base"])
assert not new_problems, new_problems
result = dict(status="PASS", resolved_packages=installed, hp_source=hp_source,
    inherited_base_pip_issues=sorted(checks["base"]), new_pip_issues=new_problems,
    hp_not_added_to_reference_install=True)
(root / "results/install_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
