#!/usr/bin/env python3
import json, re, os
LRX = os.environ.get("LRX", "/home/ylzhang/lrx")
SIZES = [128, 256, 512, 1024, 2048]
CONDS = ["none", "elec", "elec-mp", "disp", "both"]

def parse_main(path):
    rows = {}
    pat = re.compile(r'\[([a-z-]+)\s*\]\s+([\w-]+)\s+N=\s*(\d+)\s+ms/step=\s*([\d.]+|FAIL)')
    for line in open(path):
        m = pat.search(line)
        if m:
            rows[(m.group(1), m.group(2), int(m.group(3)))] = (float(m.group(4)) if m.group(4) != "FAIL" else None)
    return rows

def parse_iso(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("{") and '"ms"' in line:
            j = json.loads(line); rows[(j["cond"], j["N"])] = j["ms"]
    return rows

def parse_dep(path):
    rows = {}
    if not os.path.exists(path): return rows
    pat = re.compile(r'RESULT (\w+?)_N(\d+) N=\d+ rc=(\d+) loop=(\S+) inner=(\d+) atoms_s=([\d.]+|NA)')
    for line in open(path):
        m = pat.search(line)
        if m:
            rows[(m.group(1), int(m.group(2)))] = (float(m.group(6)) if (m.group(6) != "NA" and m.group(3) == "0") else None)
    return rows

main = parse_main(f"{LRX}/bench_lr_rerun.log")
iso  = parse_iso(f"{LRX}/bench_lr_makefx_disp_isolated.log")
dep  = parse_dep(f"{LRX}/lr_pipeline_rerun.log"); dep.update(parse_dep(f"{LRX}/fig_nolr.log"))

results = []
def add(mode, cond, N, ms=None, atoms_s=None, note=""):
    if ms is None and atoms_s is None: return
    a = atoms_s if atoms_s is not None else N / (ms / 1000.0)
    m = ms if ms is not None else N / (a / 1000.0)
    results.append(dict(mode=mode, cond=cond, N=N, ms=m, atoms_s=a, sane=None, note=note))

for N in SIZES:
    for cond in CONDS:
        for mode in ["train", "infer", "aoti-infer", "md"]:
            add(mode, cond, N, ms=main.get((mode, cond, N)))
        if cond in ("disp", "both"):
            add("makefx-train", cond, N, ms=iso.get((cond, N)), note="compiled")
        else:
            add("makefx-train", cond, N, ms=main.get(("makefx-train", cond, N)))
        add("aoti-md", cond, N, ms=main.get(("aoti-md", cond, N)))
    # deployed-MD ladder (current code, rho=0.04, 8-probe cores)
    add("aoti-md", "disp-respa", N, atoms_s=dep.get(("disprespa", N)))
    add("aoti-md", "disp-c6",    N, atoms_s=dep.get(("c6", N)))

out = dict(backend="ictd-bridge-u", channels=64, lmax=2, num_interaction=2, degree=50, dtype="float32", results=results)
json.dump(out, open(f"{LRX}/bench_lr_current.json", "w"), indent=2)
print(f"wrote bench_lr_current.json: {len(results)} cells")
