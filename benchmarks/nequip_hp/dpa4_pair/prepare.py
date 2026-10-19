"""Parameter-only pairing and fixed inputs; run only in the designated sbatch job."""
import argparse
import copy
from fractions import Fraction
import gc
import json
import os
from pathlib import Path

CASES = {"l2d2": (2, 2, 84), "l3d3": (3, 3, 64)}
SEED = 20261018


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def require_batch():
    if (not os.environ.get("SLURM_JOB_ID") or
            os.environ.get("SLURM_JOB_PARTITION") != "GPU-A100" or
            "XHPC-A100" not in os.environ.get("SLURM_JOB_NODELIST", "")):
        raise RuntimeError("This numerical preparation requires sbatch / GPU-A100 / XHPC-A100")


def nequip_table(root):
    import torch
    from common import config, periodic_graph, strict_fp32
    from nequip.model import model_from_config
    strict_fp32()
    tables = {}
    # The width dependence is quadratic on the declared multiple-of-four grid.
    # Independent extra probes must agree exactly, and the chosen width is
    # constructed/count-checked again before any performance test.
    probes = (16, 32, 64, 84, 128)
    for case, (lmax, layers, anchor) in CASES.items():
        for scope in ("final", "persistent"):
            counts = {}
            for width in probes:
                torch.manual_seed(SEED)
                model = model_from_config(config(scope, width, lmax, layers, oeq=False), initialize=True)
                counts[width] = sum(p.numel() for p in model.parameters())
                del model
                gc.collect()
            def interpolate(x):
                total = Fraction(0)
                for a in probes[:3]:
                    term = Fraction(counts[a])
                    for b in probes[:3]:
                        if a != b:
                            term *= Fraction(x-b, a-b)
                    total += term
                if total.denominator != 1:
                    raise AssertionError("Noninteger parameter interpolation")
                return int(total)
            for width in probes:
                if interpolate(width) != counts[width]:
                    raise AssertionError("Parameter formula failed an independent constructor probe")
            tables[f"{case}/{scope}"] = dict(lmax=lmax, layers=layers, anchor_channels=anchor,
                anchor_parameters=counts[anchor], constructor_probes=counts,
                width_grid=[dict(channels=c, parameters=interpolate(c)) for c in range(8, 193, 4)])
            print(json.dumps(dict(stage="nequip_parameter_table", case=case, scope=scope,
                                  anchor_parameters=counts[anchor])), flush=True)
    dump(root/"preparation/nequip_parameter_table.json", tables)
    for n in (128, 1024, 4096):
        for batch in range(3):
            data, geometry = periodic_graph(n, seed=SEED+batch)
            from nequip.data import AtomicDataDict as D
            data[D.ATOM_TYPE_KEY] = (data[D.ATOM_TYPE_KEY] + batch) % 4
            path = root/f"inputs/n{n}_b{batch}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(data={k:v.cpu() for k,v in data.items()}, geometry=geometry,
                            cutoff=4.0, seed=SEED+batch), path)
            del data
    print("NEQUIP_TABLE_AND_INPUTS_DONE", flush=True)


def dpa_pairs(root, template_path):
    import torch
    from deepmd.pt.model.model import get_model
    from deepmd.utils.argcheck import normalize
    table = json.loads((root/"preparation/nequip_parameter_table.json").read_text())
    template = json.loads(template_path.read_text())
    report = dict(status="RUNNING", job=os.environ["SLURM_JOB_ID"], seed=SEED,
        selection="Parameter counts only; DPA supported-width grid first, then nearest NequIP width. No timings used.",
        dpa_channel_grid=[32, 64, 96, 128], nequip_channel_grid=list(range(8,193,4)),
        maximum_parameter_mismatch=.05, cases={})
    for case, (lmax, layers, _) in CASES.items():
        candidates = []
        for width in report["dpa_channel_grid"]:
            spec = copy.deepcopy(template)
            spec["model"]["descriptor"].update(channels=width, lmax=lmax, mmax=1,
                n_blocks=layers, rcut=4.0, seed=SEED, use_amp=False, precision="float32")
            spec["model"]["descriptor"].pop("l_schedule", None)
            spec["model"]["descriptor"].pop("m_schedule", None)
            spec["model"]["fitting_net"].update(seed=SEED, precision="float32")
            spec["model"].update(use_compile=True, enable_tf32=False)
            torch.manual_seed(SEED)
            model = get_model(copy.deepcopy(normalize(copy.deepcopy(spec))["model"]))
            count = sum(p.numel() for p in model.parameters())
            candidates.append(dict(channels=width, parameters=count, config=spec))
            del model
            gc.collect()
        for scope in ("final", "persistent"):
            target = table[f"{case}/{scope}"]
            dpa = min(candidates, key=lambda r:(abs(r["parameters"]-target["anchor_parameters"]), r["channels"]))
            neq = min(target["width_grid"], key=lambda r:(abs(r["parameters"]-dpa["parameters"]), r["channels"]))
            mismatch = (dpa["parameters"]-neq["parameters"])/neq["parameters"]
            if abs(mismatch) > report["maximum_parameter_mismatch"]:
                raise RuntimeError("Declared parameter grid does not yield a close pair")
            directory = root/"pairs"/case/scope
            directory.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(SEED)
            model = get_model(copy.deepcopy(normalize(copy.deepcopy(dpa["config"]))["model"]))
            torch.save(model.state_dict(), directory/"dpa_initialization.pt")
            dump(directory/"dpa_config.json", dpa["config"])
            metadata = dict(case=case, scope=scope, lmax=lmax, layers=layers,
                nequip_channels=neq["channels"], nequip_parameters=neq["parameters"],
                dpa_channels=dpa["channels"], dpa_parameters=dpa["parameters"],
                signed_parameter_mismatch=mismatch, rank=16, cutoff=4.0, neighbors=50,
                anchor=target, dpa_candidates=[{k:v for k,v in r.items() if k!="config"} for r in candidates])
            dump(directory/"metadata.json", metadata)
            report["cases"][f"{case}/{scope}"] = metadata
            print(json.dumps(dict(stage="pair_selected", **{k:v for k,v in metadata.items() if k!="anchor"})), flush=True)
            del model
            gc.collect()
    report["status"] = "DONE"
    dump(root/"preparation/parameter_pairs.json", report)
    print("DPA_PAIRS_DONE", flush=True)


def save_nequip(root):
    import torch
    from common import config
    from nequip.model import model_from_config
    for case, (lmax, layers, _) in CASES.items():
        for scope in ("final", "persistent"):
            directory = root/"pairs"/case/scope
            metadata = json.loads((directory/"metadata.json").read_text())
            spec = config(scope, metadata["nequip_channels"], lmax, layers, oeq=False)
            torch.manual_seed(SEED)
            model = model_from_config(spec, initialize=True)
            count = sum(p.numel() for p in model.parameters())
            if count != metadata["nequip_parameters"]:
                raise AssertionError("Selected NequIP constructor disagrees with parameter-only selection")
            torch.save(model.state_dict(), directory/"nequip_initialization.pt")
            dump(directory/"nequip_config.json", spec)
            del model
            gc.collect()
    print("PREPARATION_DONE", flush=True)


if __name__ == "__main__":
    require_batch()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("nequip_table", "dpa_pairs", "save_nequip"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    args = parser.parse_args()
    if args.mode == "nequip_table": nequip_table(args.root)
    elif args.mode == "dpa_pairs": dpa_pairs(args.root, args.template)
    else: save_nequip(args.root)
