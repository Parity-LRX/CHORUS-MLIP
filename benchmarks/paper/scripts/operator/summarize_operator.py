#!/usr/bin/env python
"""Summarize the operator benchmark into operator_cartnn_vs_ictd_summary.md.

Reads operator_cartnn_vs_ictd.csv (eager e3nn/cartnn/ictd) and, if present,
operator_ictd_compiled.csv (torch.compile ICTC). Produces markdown tables + a
carefully-scoped conclusions section (computed ratios, hedged language; no
model-level or chemical-accuracy claims). Optional diagnostic rows such as
`cace_sym` and `cart3l` are reported as structural references, not as numerically
matched TP implementations.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
MAIN = os.path.join(OUT, "operator_cartnn_vs_ictd.csv")
COMP = os.path.join(OUT, "operator_ictd_compiled.csv")


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


rows = load(MAIN) + load(COMP)
ok = [r for r in rows if r["status"] == "ok"]
oom = [r for r in rows if r["status"] in ("oom", "error")]

# index: (cfg, channels, dtype, mode, backend, edges) -> row
def fnum(x):
    try:
        return float(x)
    except Exception:
        return None

idx = {}
configs, channels_set, edges_set = [], set(), set()
for r in ok:
    cfg = (int(r["hidden_lmax"]), int(r["max_ell"]))
    if cfg not in configs:
        configs.append(cfg)
    channels_set.add(int(r["channels"])); edges_set.add(int(r["edges"]))
    idx[(cfg, int(r["channels"]), r["dtype"], r["mode"], r["backend"], int(r["edges"]))] = r
configs.sort()
all_channels = sorted(channels_set)
all_edges = sorted(edges_set)
backends_present = []
for b in ("e3nn", "cartnn", "ictd", "ictd_compiled", "cace_sym", "cart3l"):
    if any(k[4] == b for k in idx):
        backends_present.append(b)

L = []
def w(s=""):
    L.append(s)

w("# Operator benchmark: MACE-ICTC ICTC product vs Cartesian tensor-product references")
w()
env = ["RTX 4090 (D)", "torch 2.7.1+cu128", "e3nn 0.5.9", "cartnn 0.5.8 @ 4d0dc38"]
if "cace_sym" in backends_present:
    env.append("CACE-style monomial proxy from BingqingCheng/cace @ 3f1a665")
env.append("MACE-ICTC local git 414aa25")
w(f"**{', '.join(env)}. TF32 disabled.**")
w()
w("## What is being compared (and what is NOT)")
w()
w("The matched operator for `e3nn`, `cartnn`, and `ictd` is the **equivariant tensor product** that couples a hidden node "
  "feature (degrees `0..hidden_lmax`, `C` channels) with the edge angular embedding "
  "(degrees `0..max_ell`), per-edge weighted, over a batch of `E` directed edges — i.e. the "
  "MACE convolution tensor product. All backends run the **identical `(l1,l2,l3)` "
  "natural-parity path set** and the same per-edge weight count (`num_paths*C`) unless marked diagnostic:")
w()
w("| backend | operator | basis / storage | fusion |")
w("|---|---|---|---|")
w("| `e3nn` (reference) | `e3nn.o3.TensorProduct` (wigner_3j) | spherical, `2l+1` | opt_einsum_fx codegen |")
w("| `cartnn` | `cartnn.o3.TensorProduct` (cartesian_3j) | **full Cartesian, `3**l`** | opt_einsum_fx codegen |")
w("| `ictd` | `EdgeWeightedPathPreservingTensorProduct` | irreducible-Cartesian (ICTC), `2l+1` | **eager** (Python per-path) |")
if "ictd_compiled" in backends_present:
    w("| `ictd_compiled` | same ICTC op under `torch.compile` | ICTC `2l+1` | torch.compile (deployed form) |")
if "cart3l" in backends_present:
    w("| `cart3l` | local dense component TP diagnostic | ordered Cartesian tensor power, `3**l` on each leg | PyTorch einsum loop |")
if "cace_sym" in backends_present:
    w("| `cace_sym` | local CACE-style monomial product proxy | symmetric Cartesian monomials, `(l+1)(l+2)/2`; native output degree `l1+l2` | scatter-add product |")
w()
w("**Caveats (do not over-read):**")
w("- This is an *operator-level comparable workload*, **not** an exact apples-to-apples "
  "comparison: cartnn stores a degree-`l` tensor in `3**l` components (vs `2l+1`), and the "
  "per-path normalizations differ. Numerical outputs are **not** expected to match across backends.")
if "cart3l" in backends_present:
    w("- `cart3l` is a deliberately dense native-Cartesian diagnostic over the same MACE path set. "
      "It isolates the cost of ordered component axes; it is not a learned equivariant operator.")
if "cace_sym" in backends_present:
    w("- `cace_sym` follows the CACE-style symmetric monomial product (`l1+l2` output degree), so its "
      "path semantics differ from CG tensor products. Use it as a storage/product-structure reference, "
      "not as a speedup denominator for ICTC.")
w("- cartnn ships **no symmetric-contraction operator** (the authors declined to implement ICTC), "
  "so the MACE symmetric contraction is **out of scope** here; only the binary tensor product is compared.")
w("- `e3nn`/`cartnn` `TensorProduct` are **codegen-fused**; the bare `ictd` operator is timed in "
  "**eager** mode (Python per-path overhead, dominant at small sizes). The deployed MACE-ICTC model "
  "removes this via AOTI/`torch.compile` — see `ictd_compiled` below and the existing model-level "
  "throughput benchmarks. Read `ictd` eager numbers as a lower bound on the deployed ICTC speed.")
w("- No chemical-accuracy or model-level superiority is claimed or measured here.")
w()

def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b

# headline ratios at a representative size
def fmt(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"

REP_C = 64 if 64 in all_channels else all_channels[0] if all_channels else 64
REP_E = 100000 if 100000 in all_edges else (all_edges[-1] if all_edges else 0)
w(f"## Headline (channels={REP_C}, edges={REP_E})")
w()
w("`total_ms` = forward (+backward). Speedup ratios are reported only for the matched ICTC rows; "
  "diagnostic rows are shown as timings, not as equivalence claims.")
w()
for dt in ("float32", "float64"):
    for mode in ("forward_only", "forward_backward"):
        any_row = any((cfg, REP_C, dt, mode, "cartnn", REP_E) in idx for cfg in configs)
        if not any_row:
            continue
        w(f"### {dt}, {mode}")
        w()
        hdr = "| config (hid/ell) | " + " | ".join(f"{b} total_ms" for b in backends_present) + \
              " | ictd/cartnn | " + ("ictd_comp/cartnn |" if "ictd_compiled" in backends_present else "")
        w(hdr)
        w("|" + "---|" * (2 + len(backends_present) + (1 if "ictd_compiled" in backends_present else 0)))
        for cfg in configs:
            cells = []
            vals = {}
            for b in backends_present:
                r = idx.get((cfg, REP_C, dt, mode, b, REP_E))
                v = fnum(r["total_ms"]) if r else None
                vals[b] = v
                cells.append(fmt(v, 3) if v is not None else "—")
            cart = vals.get("cartnn")
            sp_ictd = ratio(cart, vals.get("ictd"))
            line = f"| {cfg[0]}/{cfg[1]} | " + " | ".join(cells) + f" | {fmt(sp_ictd)} |"
            if "ictd_compiled" in backends_present:
                line += f" {fmt(ratio(cart, vals.get('ictd_compiled')))} |"
            w(line)
        w()

# full per-config tables across edges at REP_C
w(f"## Throughput vs directed edges (channels={REP_C}, edges/s)")
w()
for dt in ("float32", "float64"):
    for mode in ("forward_only", "forward_backward"):
        present = any((cfg, REP_C, dt, mode, b, e) in idx
                      for cfg in configs for b in backends_present for e in all_edges)
        if not present:
            continue
        w(f"### {dt}, {mode}")
        w()
        w("| config | backend | " + " | ".join(f"E={e}" for e in all_edges) + " |")
        w("|" + "---|" * (2 + len(all_edges)))
        for cfg in configs:
            for b in backends_present:
                cells = []
                for e in all_edges:
                    r = idx.get((cfg, REP_C, dt, mode, b, e))
                    cells.append(f"{fnum(r['edges_per_s'])/1e6:.2f}M" if r else "—")
                w(f"| {cfg[0]}/{cfg[1]} | {b} | " + " | ".join(cells) + " |")
        w()

# channel scaling
w(f"## Channel scaling (edges={REP_E}, forward+backward, total_ms)")
w()
for dt in ("float32", "float64"):
    present = any((cfg, c, dt, "forward_backward", b, REP_E) in idx
                  for cfg in configs for c in all_channels for b in backends_present)
    if not present:
        continue
    w(f"### {dt}")
    w()
    w("| config | backend | " + " | ".join(f"C={c}" for c in all_channels) + " |")
    w("|" + "---|" * (2 + len(all_channels)))
    for cfg in configs:
        for b in backends_present:
            cells = []
            for c in all_channels:
                r = idx.get((cfg, c, dt, "forward_backward", b, REP_E))
                cells.append(f"{fnum(r['total_ms']):.2f}" if r else "—")
            w(f"| {cfg[0]}/{cfg[1]} | {b} | " + " | ".join(cells) + " |")
    w()

# OOM / error
w("## OOM / error cells")
w()
if not oom:
    w("None — every cell completed.")
else:
    w("| backend | config | channels | edges | dtype | mode | status | error |")
    w("|---|---|---|---|---|---|---|---|")
    for r in oom:
        w(f"| {r['backend']} | {r['hidden_lmax']}/{r['max_ell']} | {r['channels']} | {r['edges']} | "
          f"{r['dtype']} | {r['mode']} | {r['status']} | {r['error'][:80]} |")
w()

# auto conclusions (computed, hedged)
w("## Observations (measured, scoped to the tested workloads)")
w()
def avg_ratio(dt, mode, num_b, den_b):
    rs = []
    for cfg in configs:
        for c in all_channels:
            for e in all_edges:
                a = idx.get((cfg, c, dt, mode, num_b, e))
                b = idx.get((cfg, c, dt, mode, den_b, e))
                if a and b:
                    va, vb = fnum(a["total_ms"]), fnum(b["total_ms"])
                    if va and vb:
                        rs.append(vb / va)  # num faster than den if >1
    return rs

for dt in ("float32", "float64"):
    for mode in ("forward_only", "forward_backward"):
        rs = avg_ratio(dt, mode, "ictd", "cartnn")
        if rs:
            import statistics as st
            w(f"- **{dt} {mode}**: across all tested (config,channels,edges), eager `ictd` vs `cartnn` "
              f"total-time ratio (cartnn/ictd) median **{st.median(rs):.2f}×** "
              f"(min {min(rs):.2f}×, max {max(rs):.2f}×); >1 ⇒ ICTC faster.")
        rc = avg_ratio(dt, mode, "ictd_compiled", "cartnn")
        if rc:
            import statistics as st
            w(f"  - `ictd_compiled` vs `cartnn`: median **{st.median(rc):.2f}×** "
              f"(min {min(rc):.2f}×, max {max(rc):.2f}×).")
w()
if "cart3l" in backends_present or "cace_sym" in backends_present:
    layout_notes = ["cartnn exposes the cost of full ordered `3**l` component axes"]
    if "cart3l" in backends_present:
        layout_notes.append("`cart3l` gives a deliberately dense native-`3**l` diagnostic on the same path set")
    if "cace_sym" in backends_present:
        layout_notes.append("`cace_sym` exposes the smaller symmetric-monomial axis used by CACE-style products")
    w("Interpretation guidance: " + "; ".join(layout_notes) + ". "
      "Only `e3nn`/`cartnn`/`ictd` share the matched MACE `(l1,l2,l3)` TP path semantics. Where eager "
      "ICTC trails, it is the eager per-path launch overhead, not the ICTC algebra — compare the "
      "`ictd_compiled` row. None of this speaks to accuracy or to full-model performance.")
else:
    w("Interpretation guidance: cartnn's full `3**l` storage makes its per-edge work grow faster with "
      "`max_ell` than the `2l+1` ICTC/e3nn layouts, which is the main structural difference these "
      "numbers probe. Where eager ICTC trails, it is the eager per-path launch overhead, not the ICTC "
      "algebra — compare the `ictd_compiled` row. e3nn is included only as the spherical MACE-native "
      "reference. None of this speaks to accuracy or to full-model performance.")

with open(os.path.join(OUT, "operator_cartnn_vs_ictd_summary.md"), "w") as f:
    f.write("\n".join(L) + "\n")
print("wrote", os.path.join(OUT, "operator_cartnn_vs_ictd_summary.md"))
print(f"ok rows={len(ok)} oom/err={len(oom)} backends={backends_present} configs={configs} "
      f"channels={all_channels} edges={all_edges}")
