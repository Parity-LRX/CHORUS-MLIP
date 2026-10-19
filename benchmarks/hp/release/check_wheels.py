"""Inspect paired wheel metadata without importing either implementation."""
import configparser
import email
import json
import os
from pathlib import Path
import tomllib
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


root = Path(os.environ["ROOT"])
source = Path(os.environ["SOURCE_ROOT"])
projects = {}
for key, filename in (("chorus", "chorus_mlip-0.1.0-*.whl"),
                      ("hp", "ictc_o3_hp-0.2.12-*.whl")):
    wheels = list(Path(os.environ["WHEELHOUSE"]).glob(filename))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        meta = email.message_from_bytes(wheel.read(next(
            name for name in names if name.endswith(".dist-info/METADATA"))))
        entries = configparser.ConfigParser()
        entries.read_string(wheel.read(next(
            name for name in names if name.endswith(".dist-info/entry_points.txt"))).decode())
        projects[key] = dict(name=meta["Name"], version=meta["Version"],
            requires=meta.get_all("Requires-Dist", []),
            scripts=dict(entries["console_scripts"]), files=names)

assert projects["chorus"]["version"] == "0.1.0"
assert projects["hp"]["version"] == "0.2.12"
dependencies = [Requirement(value) for value in projects["chorus"]["requires"]]
hp = [value for value in dependencies if canonicalize_name(value.name) == "ictc-o3-hp"]
assert len(hp) == 1
assert hp[0].extras == {"hp"} and not str(hp[0].specifier)
assert hp[0].url == "git+https://github.com/Parity-LRX/ICTC-O3-HP.git@9fdea47eef620013e9ea591f7d90534bb27671c8"
assert hp[0].marker is not None and hp[0].marker.evaluate({"extra": "hp"})
assert not hp[0].marker.evaluate({"extra": ""})
assert not hp[0].marker.evaluate({"extra": "full"})
before = tomllib.loads((root / "reference_before/pyproject.toml").read_text())["project"]
after = tomllib.loads((source / "chorus/pyproject.toml").read_text())["project"]
assert after["dependencies"] == before["dependencies"]
assert after["version"] == before["version"]
assert after["requires-python"] == before["requires-python"]
assert after["scripts"] == before["scripts"]
for extra, values in before["optional-dependencies"].items():
    assert after["optional-dependencies"][extra] == values, extra
assert set(projects["chorus"]["scripts"]).isdisjoint(projects["hp"]["scripts"])
assert all(name.startswith("ictc-hp-") for name in projects["hp"]["scripts"])
assert "chorus/hp/_compat.py" in projects["chorus"]["files"]
assert "mace_ictc/hp/coherent.py" in projects["hp"]["files"]
assert "mace_ictc/hp/_radial_first_training.py" in projects["hp"]["files"]
assert not any(name.startswith("mace_ictc/") for name in projects["chorus"]["files"])
result = dict(status="PASS", exact_hp_extra=str(hp[0]),
    base_dependencies_unchanged=True, existing_extras_unchanged=True,
    console_scripts_disjoint=True,
    wheels={key: {name: value for name, value in data.items() if name != "files"}
            for key, data in projects.items()})
(root / "results/wheel_metadata.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
