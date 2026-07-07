#!/usr/bin/env python3
"""Throughput benchmark harness for the MACE-ICTC long-range modules (RTX 4090).

Mirrors the paper protocol: fp32, warmup + timed iters, atom-count sweep, reports
ms/step AND atoms/s. Reuses mace_ictc/synthetic.py building blocks.

MODES (per condition, per size):
  1. train      : eager full_train_step (fwd + force/loss double-backward)   [in-proc]
  2. makefx-train: faithful trainer --train-makefx-compile path on a tiny H5  [subprocess -> mode_makefx]
  3. infer      : eager forward_and_forces (energy+forces)                    [in-proc]
  4. aoti-infer : export .pt2 (export_aoti_core --dynamic), load, time forward [in-proc]
  5. md         : single eager energy+forces eval (= per-MD-step cost)        [in-proc, == infer]
  6. aoti-md    : LAMMPS lmp + pair_style mff/torch + .pt2 + C++ LR solver     [subprocess -> bench_lammps_md.sh]

CONDITIONS: none / elec / elec-mp / disp / both  (long-range kwargs passed via build_model **extra)

Usage:
  # in-process eager + aoti-infer modes (1,3,4,5):
  python bench_lr_throughput.py --modes train,infer,aoti-infer,md
  # one cell only:
  python bench_lr_throughput.py --modes train --sizes 256 --conditions disp
  # makefx single-cell worker (called by the harness via subprocess, or run directly):
  python bench_lr_throughput.py --mode-makefx --size 256 --condition disp

Scale up: edit SIZES / CONDITIONS / MODES at the top, then run the full thing.
makefx-train (mode 2) and aoti-md (mode 6) shell out; see run_makefx_subprocess /
the bench_lammps_md.sh template for those.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

DEFAULT_REPO = Path(__file__).resolve().parents[4]
REPO = os.environ.get("MACE_ICTC_REPO", str(DEFAULT_REPO))
PY = os.environ.get("PY", sys.executable)
LMP = os.environ.get("LMP", "lmp")
LAMMPS_MD_SH = os.environ.get("LAMMPS_MD_SH", str(Path(__file__).with_name("bench_lammps_md.sh")))
sys.path.insert(0, REPO)

# ----------------------------- SWEEP CONFIG -------------------------------- #
SIZES = [128, 256, 512, 1024, 2048]                  # atom counts (24GB caps heavy disp/both -> OOM)
CONDITIONS = ["none", "elec", "elec-mp", "disp", "both"]  # main-table long-range conditions
MODES = ["train", "makefx-train", "infer", "aoti-infer", "md", "aoti-md"]
PRODUCT_BACKEND = "ictd-bridge-u"                              # ICTC bridge-U backend key, or "cueq"

# Uniform fixed density for the position-dependent long-range paths (dispersion cutoff 9.0,
# electrostatics). v1 set box=20 for elec/both and None (open-cluster) for none/disp -> different
# coords/density per condition AND density growing with N. RHO_TARGET makes the box edge
# (N/RHO_TARGET)**(1/3) at EVERY (cond,N): identical seeded coords across conditions at a given N,
# constant density across N -> disp-neighbors/atom constant in N and identical across conditions.
RHO_TARGET = 0.04  # atoms/A^3, uniform across ALL conditions & sizes

# fixed model config
CHANNELS, LMAX, NUM_INTERACTION, DEGREE = 64, 2, 2, 50
DTYPE = "float32"
ITERS, WARMUP = 20, 5

# AOTI deploy (.pt2 for LAMMPS) backend: ICTC bridge-U (key `ictd-bridge-u`) is
# pure PyTorch, with no cuequivariance custom op baked in, so LAMMPS' libtorch
# loads it WITHOUT MFF_CUSTOM_OPS_LIB/embedded-Python.
# A cueq-exported .pt2 needs MFF_CUSTOM_OPS_LIB + MFF_LIBPYTHON + PYTHONHOME (see bench_lammps_md.sh).
# Use bridge-u for the deploy modes (aoti-infer / aoti-md) to keep them robust.
AOTI_BACKEND = "ictd-bridge-u"

# makefx-train slots: make_fx bakes leading dims (incl. the DISPERSION edge count, which is
# NOT held fixed by --pad-edges-to-max) into each traced graph, so disp/both need one slot per
# distinct dispersion-edge count. Set >= n_samples so disp/both compile instead of erroring out.
MAKEFX_N_SAMPLES = 16
MAKEFX_MAX_SLOTS = MAKEFX_N_SAMPLES + 2

# long-range kwargs per condition (PureCartesianICTDFix __init__ kwargs)
LR_ELEC = dict(long_range_mode="reciprocal-spectral-v1", long_range_boundary="periodic",
               long_range_reciprocal_backend="mesh_fft", long_range_mesh_size=32,
               long_range_source_channels=1, long_range_max_multipole_l=0)
# long_range_boundary="periodic" so disp-only ALSO does PERIODIC dispersion (dispersion_pbc=True)
# over the real fixed-density box -- otherwise the model defaults disp-only to open-cluster
# (dispersion_pbc=False) while both (periodic elec) uses periodic dispersion, so disp/both disagree
# on dispersion-edge count even with identical coords. Periodic here makes disp==both and gives
# true periodic-density scaling (disp-neighbors/atom constant in N). No electrostatic module is
# created (LR_DISP has no long_range_mode); the flag only flows to dispersion_pbc.
LR_DISP = dict(long_range_dispersion_mode="mbd-slq", dispersion_cutoff=9.0,
               long_range_boundary="periodic",
               dispersion_slq_num_probes=4, mbd_operator_backend="edge_sparse")
LR_DISP_C6 = dict(long_range_dispersion_mode="pairwise-c6", dispersion_cutoff=9.0,
                  long_range_boundary="periodic")
# Multipole electrostatics: q + dipole (l=1) + quadrupole (l=2) sources, mesh-FFT reciprocal with
# full-Ewald screening + PCS assignment (deploys faithfully via the matching C++ pcs solver). Same
# scalar-elec mesh/box as LR_ELEC; the extra cost is the 13-field (q|mu|Q) source emit + reciprocal sum.
LR_ELEC_MP = dict(LR_ELEC, long_range_max_multipole_l=2,
                  long_range_mesh_fft_full_ewald=True, long_range_assignment="pcs")
def lr_extra(cond: str) -> dict:
    if cond == "none": return {}
    if cond == "elec": return dict(LR_ELEC)
    if cond == "elec-mp": return dict(LR_ELEC_MP)
    if cond == "disp": return dict(LR_DISP)
    if cond == "disp-c6": return dict(LR_DISP_C6)
    if cond == "both": return {**LR_ELEC, **LR_DISP}
    raise ValueError(cond)

# --------------------------------------------------------------------------- #
import torch
from mace_ictc.training.train_loop import disable_tf32


def _setup():
    disable_tf32()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _dtype():
    return torch.float32 if DTYPE == "float32" else torch.float64


def make_graph(N, device, *, periodic_box=None):
    """Fixed-shape graph. periodic_box != None -> positions in a real box + PBC cell
    (needed so the periodic reciprocal-spectral path has a meaningful cell)."""
    from mace_ictc.synthetic import make_fixed_graph
    dt = _dtype()
    g = list(make_fixed_graph(num_nodes=N, avg_degree=DEGREE, dtype=dt, device=device))
    if periodic_box is not None:
        gen = torch.Generator(device="cpu").manual_seed(7)
        pos = (torch.rand(N, 3, generator=gen, dtype=torch.float64) * periodic_box).to(device, dt)
        cell = (torch.eye(3, dtype=torch.float64) * periodic_box).unsqueeze(0).to(device, dt)
        g[0] = pos
        g[6] = cell
    return tuple(g)


def build(cond, device):
    from mace_ictc.synthetic import build_model
    torch.manual_seed(0)
    return build_model(channels=CHANNELS, lmax=LMAX, num_interaction=NUM_INTERACTION,
                       route="baseline", product_backend=PRODUCT_BACKEND,
                       dtype=_dtype(), device=device, **lr_extra(cond))


# ---- in-process eager + aoti modes -------------------------------------- #
def _time(fn, device, iters=ITERS, warmup=WARMUP):
    from mace_ictc.synthetic import _time_section
    return _time_section(fn, device=device, iters=iters, warmup=warmup)


def bench_eager(cond, N, device, *, with_forces, train):
    """train=True -> full_train_step (mode 1); else forward_and_forces (modes 3/5)."""
    from mace_ictc.synthetic import forward_energy_atom
    # uniform fixed density (RHO_TARGET): ALL conditions, ALL N use the SAME seeded random-in-box
    # coords at constant density. none ignores the cell; disp/both use it for long-range neighbors.
    box = (N / RHO_TARGET) ** (1.0 / 3.0)   # uniform fixed density, ALL conditions, ALL N
    m = build(cond, device)
    if train:
        m.train()
    else:
        m.eval()
    g = make_graph(N, device, periodic_box=box)
    params = [p for p in m.parameters() if p.requires_grad]
    f_tgt = torch.zeros(N, 3, device=device, dtype=_dtype())
    e_tgt = torch.zeros((), device=device, dtype=_dtype())
    fw = 10.0

    def fwd_forces():
        pos = g[0].detach().clone().requires_grad_(True)
        e = forward_energy_atom(m, pos, (pos,) + tuple(g[1:])).sum()
        grad = torch.autograd.grad(e, pos, create_graph=train)[0]
        return e, -grad

    def full_train_step():
        for p in params:
            p.grad = None
        pos = g[0].detach().clone().requires_grad_(True)
        e = forward_energy_atom(m, pos, (pos,) + tuple(g[1:])).sum()
        grad = torch.autograd.grad(e, pos, create_graph=True)[0]
        forces = -grad
        loss = (e - e_tgt) ** 2 + fw * ((forces - f_tgt) ** 2).mean()
        loss.backward()
        return loss

    fn = full_train_step if train else (lambda: fwd_forces())
    # sanity number
    if train:
        sane = float(full_train_step().detach())
    else:
        e, f = fwd_forces()
        sane = float(f.abs().max().detach())
    ms = _time(fn, device)
    del m
    torch.cuda.empty_cache()
    return ms, sane


def bench_aoti_infer(cond, N, device):
    """Export .pt2 (dynamic) via export_aoti_core, load it, time the AOTI forward
    (energy+forces). The long-range ENERGY is deferred to C++ at deploy, so the
    .pt2 emits SOURCES only (short-range + source head) -> ~flat across conditions."""
    from mace_ictc.cli import export_aoti_core as eac
    out = f"/tmp/lr_aoti_{cond}_{N}.pt2"
    argv = ["--route", "baseline", "--channels", str(CHANNELS), "--lmax", str(LMAX),
            "--num-interaction", str(NUM_INTERACTION), "--degree", str(DEGREE),
            "--atoms", str(N), "--product-backend", AOTI_BACKEND, "--dtype", DTYPE,
            "--device", "cuda", "--dynamic", "--elements", "H,C,N,O", "--out", out,
            "--iters", str(ITERS), "--warmup", str(WARMUP)]
    ex = lr_extra(cond)
    if "long_range_mode" in ex:
        argv += ["--long-range-mode", "reciprocal-spectral-v1",
                 "--long-range-multipole-l", str(ex["long_range_max_multipole_l"]),
                 "--lr-mesh-size", str(ex["long_range_mesh_size"])]
    if "long_range_dispersion_mode" in ex:
        argv += ["--dispersion-mode", str(ex["long_range_dispersion_mode"]),
                 "--dispersion-cutoff", str(ex["dispersion_cutoff"])]
        if "mbd_operator_backend" in ex:
            argv += ["--mbd-operator-backend", ex["mbd_operator_backend"]]
    # Dispersion-ONLY: suppress the packed reciprocal_source so the .pt2 emits pure (E, force)
    # with the MBD/SLQ energy computed in-graph (newton-schulz SLQ is AOTI-deployable). The
    # sidecar then carries export_reciprocal_source=false, so the LAMMPS engine's
    # 'MBD edges + runtime reciprocal source' guard is skipped -> direct AOTI dispersion path.
    # (For "both" we keep the reciprocal_source: the combined elec+MBD AOTI path runs directly
    #  with a detached reciprocal_source, no fallback needed.)
    if cond == "disp":
        argv += ["--no-reciprocal-source"]
    # export_aoti_core.main() parses sys.argv; it ALSO times eager-vs-aoti and prints it.
    old = sys.argv
    sys.argv = ["export_aoti_core"] + argv
    rc = 1
    try:
        rc = eac.main()
    finally:
        sys.argv = old
    # The export log prints "[aoti] ... aoti ... ms"; we re-time the loaded package here ourselves
    # for a uniform number, by loading + running the .pt2 with a fixed graph.
    ms, sane = _time_loaded_pt2(out, cond, N, device)
    return ms, sane, out, rc


def _time_loaded_pt2(path, cond, N, device):
    """Load the exported .pt2 and time its forward. Inputs mirror the export example graph."""
    from mace_ictc.cli import export_aoti_core as eac
    runner = eac._aoti_load(path, device)
    box = (N / RHO_TARGET) ** (1.0 / 3.0)   # uniform fixed density, ALL conditions, ALL N
    g = make_graph(N, device, periodic_box=box)
    pos, A, batch, es, ed, esh, cell = g
    pos = pos.detach().clone()
    # AOTI core signature: (pos, A, batch, edge_src, edge_dst, edge_shifts, cell[, disp edges...])
    # try the plain 7-arg signature; if the model takes dispersion edges, pass empty ones.
    ex = lr_extra(cond)
    takes_disp = "long_range_dispersion_mode" in ex
    def call_plain():
        return runner(pos, A, batch, es, ed, esh, cell)
    def call_disp():
        z = torch.zeros(1, dtype=torch.long, device=device)
        zs = torch.zeros(1, 3, dtype=_dtype(), device=device)
        return runner(pos, A, batch, es, ed, esh, cell, z, z, zs)
    fn = call_plain
    try:
        r = call_plain()
    except Exception:
        fn = call_disp
        r = call_disp()
    e = r[0] if isinstance(r, (tuple, list)) else r
    f = r[1] if isinstance(r, (tuple, list)) and len(r) > 1 else None
    sane = float(f.abs().max()) if f is not None else float(e.sum())
    ms = _time(fn, device)
    return ms, sane


# ---- makefx-train (faithful trainer subprocess) ------------------------- #
def _write_tiny_h5(path, N, n_samples=24, box=None, cutoff=5.0, seed=0):
    """Write a tiny processed_<prefix>.h5 with the exact schema H5Dataset reads.
    All samples at the SAME atom count N so --makefx-buckets 1 -> one fixed shape."""
    import h5py, numpy as np
    if box is None:
        box = float((N / RHO_TARGET) ** (1.0 / 3.0))   # uniform fixed density, ALL conditions, ALL N
    def brute(pos, L, cut):
        Np = len(pos); src, dst, sh = [], [], []
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in (-1, 0, 1):
                    S = np.array([sx, sy, sz], float); disp = S * L
                    for a in range(Np):
                        r = np.linalg.norm(pos + disp - pos[a], axis=1)
                        for b in range(Np):
                            if r[b] < cut and not (a == b and sx == sy == sz == 0):
                                src.append(a); dst.append(b); sh.append(S)
        return (np.array(src, np.int64), np.array(dst, np.int64),
                np.array(sh, np.float64).reshape(-1, 3))
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        max_e = max_a = 0
        for idx in range(n_samples):
            rng = np.random.default_rng(seed)  # 1-SLOT identical samples -> fixed disp edges
            pos = rng.uniform(0, box, size=(N, 3))
            cell = np.eye(3) * box
            A = rng.choice([1, 6, 7, 8], size=N)
            i, j, S = brute(pos, box, cutoff)
            g = f.create_group(f"sample_{idx}")
            g.create_dataset("pos", data=pos.astype(np.float64))
            g.create_dataset("A", data=A.astype(np.int64))
            g.create_dataset("y", data=np.float64(rng.normal() * N))
            g.create_dataset("force", data=rng.normal(size=(N, 3)).astype(np.float64))
            g.create_dataset("edge_src", data=i)
            g.create_dataset("edge_dst", data=j)
            g.create_dataset("edge_shifts", data=S.astype(np.float64))
            g.create_dataset("cell", data=cell.astype(np.float64))
            st = rng.normal(size=(3, 3)); st = 0.5 * (st + st.T)
            g.create_dataset("stress", data=st.astype(np.float64))
            max_e = max(max_e, len(i)); max_a = max(max_a, N)
        f.attrs["max_edges"] = max_e
        f.attrs["max_atoms"] = max_a


def run_mode_makefx(cond, N):
    """Faithful makefx-train: build tiny H5, run the trainer with --train-makefx-compile,
    parse steady per-epoch time -> per-step ms. (This IS mode 2.) Returns (ms_per_step, note)."""
    workdir = tempfile.mkdtemp(prefix=f"makefx_{cond}_{N}_")
    n_samples, epochs, bs = MAKEFX_N_SAMPLES, 4, 1
    _write_tiny_h5(os.path.join(workdir, "processed_train.h5"), N, n_samples=n_samples)
    ex = lr_extra(cond)
    # disp/both: dispersion edge count varies per sample and is NOT padded by --pad-edges-to-max,
    # so make_fx needs one slot per distinct disp-edge count -> use MAKEFX_MAX_SLOTS (>= n_samples).
    slots = MAKEFX_MAX_SLOTS if "long_range_dispersion_mode" in ex else 2
    cmd = [PY, "-m", "mace_ictc.cli.train",
           "--data-dir", workdir, "--train-prefix", "train",
           "--epochs", str(epochs), "--batch-size", str(bs),
           "--channels", str(CHANNELS), "--lmax", str(LMAX),
           "--num-interaction", str(NUM_INTERACTION),
           "--product-backend", PRODUCT_BACKEND, "--dtype", DTYPE, "--device", "cuda",
           "--max-grad-norm", "10",
           "--train-makefx-compile", "--makefx-buckets", "1", "--makefx-max-slots", str(slots),
           "--pad-nodes-to-max", "--pad-edges-to-max",
           "--log-interval", "1000"]
    # long-range flags for the trainer (same kwarg names as build_model **extra)
    if "long_range_mode" in ex:
        cmd += ["--long-range-mode", "reciprocal-spectral-v1",
                "--long-range-boundary", "periodic",
                "--long-range-reciprocal-backend", "mesh_fft",
                "--long-range-mesh-size", str(ex["long_range_mesh_size"]),
                "--long-range-source-channels", "1",
                "--long-range-max-multipole-l", "0"]
    if "long_range_dispersion_mode" in ex:
        cmd += ["--long-range-dispersion-mode", "mbd-slq",
                "--dispersion-cutoff", str(ex["dispersion_cutoff"]),
                "--dispersion-slq-num-probes", str(ex["dispersion_slq_num_probes"]),
                "--mbd-operator-backend", ex["mbd_operator_backend"]]
    env = _repo_env({"OMP_NUM_THREADS": "1"})
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    out = p.stdout + "\n" + p.stderr
    # parse "[epoch N step M phase] ... (T.Ts)" lines -> per-step ms = T*1000/steps_per_epoch
    import re
    steps_per_epoch = n_samples // bs
    times = []
    for line in out.splitlines():
        mobj = re.search(r"\[epoch (\d+) step \d+ \w+\].*\(([\d.]+)s\)", line)
        if mobj:
            times.append((int(mobj.group(1)), float(mobj.group(2))))
    note = ""
    fb = "fall" in out.lower() and "eager" in out.lower()
    # makefx fallback signal for dispersion:
    for key in ("makefx", "dispersion", "torch_cluster", "eager fallback", "fell back"):
        for line in out.splitlines():
            if key in line.lower() and ("fall" in line.lower() or "eager" in line.lower()):
                note = line.strip()[:160]
                break
        if note:
            break
    if len(times) < 2:
        return None, f"PARSE-FAIL rc={p.returncode}; tail={out.strip()[-400:]}"
    # steady = min per-step over epochs >= 1 (epoch 0 includes compile)
    steady = [t for (e, t) in times if e >= 1]
    use = steady if steady else [t for (_, t) in times]
    ms_per_step = min(use) * 1000.0 / steps_per_epoch
    n_epochs_seen = len(times)
    return ms_per_step, (note or f"epochs={n_epochs_seen} steps/epoch={steps_per_epoch} rc={p.returncode}")


# ---- aoti-md (LAMMPS subprocess) ---------------------------------------- #
def run_mode_aoti_md(cond, N, pt2_path):
    """Run LAMMPS + pair_style mff/torch + the exported .pt2 (with C++ LR solver) for a
    short MD; parse Loop time/Performance -> ms/step. Returns (ms_per_step, note)."""
    ex = lr_extra(cond)
    disp_cut = ex.get("dispersion_cutoff", 0.0) if "long_range_dispersion_mode" in ex else 0.0
    cmd = ["bash", LAMMPS_MD_SH, pt2_path, str(N), str(disp_cut)]
    p = subprocess.run(cmd, env=_repo_env({"LMP": LMP, "PY": PY}), capture_output=True, text=True, timeout=1200)
    out = p.stdout + "\n" + p.stderr
    import re
    # LAMMPS prints "Loop time of X on P procs for S steps with N atoms"
    m = re.search(r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", out)
    if m:
        loop_s, steps = float(m.group(1)), int(m.group(2))
        ms = loop_s * 1000.0 / max(steps, 1)
        return ms, f"rc={p.returncode} loop={loop_s:.3f}s steps={steps}"
    return None, f"NO-LOOPTIME rc={p.returncode}; tail={out.strip()[-500:]}"


# --------------------------------------------------------------------------- #
def main():
    global PRODUCT_BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--backend", default=PRODUCT_BACKEND)
    # single-cell makefx worker (so it runs in a clean process)
    ap.add_argument("--mode-makefx", action="store_true")
    ap.add_argument("--mode-aoti-md", action="store_true")
    ap.add_argument("--size", type=int)
    ap.add_argument("--condition")
    ap.add_argument("--pt2")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    PRODUCT_BACKEND = args.backend
    _setup()
    device = torch.device("cuda")

    if args.mode_makefx:
        ms, note = run_mode_makefx(args.condition, args.size)
        print(json.dumps({"mode": "makefx-train", "cond": args.condition, "N": args.size,
                          "ms": ms, "atoms_s": (args.size / (ms / 1000.0) if ms else None),
                          "note": note}))
        return 0
    if args.mode_aoti_md:
        ms, note = run_mode_aoti_md(args.condition, args.size, args.pt2)
        print(json.dumps({"mode": "aoti-md", "cond": args.condition, "N": args.size,
                          "ms": ms, "atoms_s": (args.size / (ms / 1000.0) if ms else None),
                          "note": note}))
        return 0

    sizes = [int(x) for x in args.sizes.split(",")]
    conds = args.conditions.split(",")
    modes = args.modes.split(",")
    results = []

    def rec(mode, cond, N, ms, sane=None, note=""):
        a = (N / (ms / 1000.0)) if ms else None
        results.append(dict(mode=mode, cond=cond, N=N, ms=ms, atoms_s=a, sane=sane, note=note))
        mss = f"{ms:8.3f}" if ms is not None else "    FAIL"
        ass = f"{a:10.0f}" if a is not None else "        --"
        print(f"[{mode:12s}] {cond:5s} N={N:4d}  ms/step={mss}  atoms/s={ass}  "
              f"sane={sane}  {note}", flush=True)

    for N in sizes:
        for cond in conds:
            pt2_for_md = None
            for mode in modes:
                try:
                    if mode == "train":
                        ms, sane = bench_eager(cond, N, device, with_forces=True, train=True)
                        rec(mode, cond, N, ms, sane)
                    elif mode == "infer":
                        ms, sane = bench_eager(cond, N, device, with_forces=True, train=False)
                        rec(mode, cond, N, ms, sane)
                    elif mode == "md":
                        ms, sane = bench_eager(cond, N, device, with_forces=True, train=False)
                        rec(mode, cond, N, ms, sane, "eager E+F == infer")
                    elif mode == "aoti-infer":
                        ms, sane, pt2, rc = bench_aoti_infer(cond, N, device)
                        pt2_for_md = pt2
                        rec(mode, cond, N, ms, sane, f"sources-only(LR deferred) export_rc={rc}")
                    elif mode == "makefx-train":
                        # disp/both: the MBD-SLQ path is NOT make_fx-compilable (variable
                        # dispersion-edge count blows the slot cache); report the eager train
                        # number as eager-equivalent instead of burning time on slot errors.
                        if cond in ("disp", "both"):
                            ms, _ = bench_eager(cond, N, device, with_forces=True, train=True)
                            rec(mode, cond, N, ms, None, "eager-equivalent")
                        else:
                            cp = subprocess.run(
                                [PY, __file__, "--mode-makefx", "--condition", cond,
                                 "--size", str(N), "--backend", PRODUCT_BACKEND],
                                cwd=os.path.dirname(__file__) or ".",
                                env=_repo_env(), capture_output=True,
                                text=True, timeout=2000)
                            j = _last_json(cp.stdout)
                            if j:
                                rec(mode, cond, N, j["ms"], None, j["note"])
                            else:
                                rec(mode, cond, N, None, None,
                                    f"WORKER-FAIL rc={cp.returncode} {cp.stderr.strip()[-300:]}")
                    elif mode == "aoti-md":
                        if pt2_for_md is None:
                            # need an export first
                            _, _, pt2_for_md, _ = bench_aoti_infer(cond, N, device)
                        cp = subprocess.run(
                            [PY, __file__, "--mode-aoti-md", "--condition", cond,
                             "--size", str(N), "--pt2", pt2_for_md, "--backend", PRODUCT_BACKEND],
                            cwd=os.path.dirname(__file__) or ".",
                            env=_repo_env({"LMP": LMP, "PY": PY}), capture_output=True,
                            text=True, timeout=1500)
                        j = _last_json(cp.stdout)
                        if j:
                            rec(mode, cond, N, j["ms"], None, j["note"])
                        else:
                            rec(mode, cond, N, None, None,
                                f"WORKER-FAIL rc={cp.returncode} {cp.stderr.strip()[-300:]}")
                except Exception as exc:
                    import traceback
                    msg = str(exc)
                    is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                        "out of memory" in msg.lower() or "CUDA error: out of memory" in msg)
                    note = "OOM" if is_oom else f"EXC {type(exc).__name__}: {msg[:160]}"
                    rec(mode, cond, N, None, None, note)
                    if not is_oom:
                        traceback.print_exc()
                    # always recover the allocator so the next cell isn't poisoned
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(backend=PRODUCT_BACKEND, channels=CHANNELS, lmax=LMAX,
                           num_interaction=NUM_INTERACTION, degree=DEGREE, dtype=DTYPE,
                           results=results), f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


def _last_json(s):
    for line in reversed(s.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except Exception:
                continue
    return None


def _repo_env(extra=None):
    env = dict(os.environ)
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = REPO if not old else f"{REPO}:{old}"
    if extra:
        env.update(extra)
    return env


if __name__ == "__main__":
    raise SystemExit(main())
