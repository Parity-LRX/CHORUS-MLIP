# MACE-ICTC paper supplement — final report

Task: (1) formula-level write-up of the ICTC algorithm contribution and its e3nn/MACE
numerical-parity mechanism; (2) an operator-level efficiency benchmark of the ICTC tensor
product vs the cartnn-style Cartesian tensor product, measured on an RTX 4090. No tracked
repo code was modified; nothing was committed. All artifacts live under
`/tmp/mace_ictc_cartnn_bench_20260614_220608/` (4090) mirrored locally at the same path.

---

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D, 24 GB (host `XHPC-4090-01`), driver 550.90.07 |
| torch | 2.7.1+cu128 (CUDA runtime 12.8) |
| e3nn | 0.5.9 |
| mace (mace-torch) | 0.3.16 (`/tmp/mace_torch_0_3_16` on PYTHONPATH) |
| cuequivariance / _torch | 0.10.0 / 0.8.1 |
| MACE-ICTC | local git **`414aa25`** (clean). The 4090 path `/home/ylzhang/lrx/MACE-ICTC` is a deployed **non-git** content copy of the same tree. |
| cartnn | **`github.com/xvzemin/cartnn`**, commit **`4d0dc381ffe76d62ccddb5cf8ab5030b270a5869`** (a fork of e3nn; pkg `cartnn` 0.5.8; pure-PyTorch). arXiv 2512.16882. |

cartnn was run **without installing into the FSCETP env** — it is pure Python and was placed
on `PYTHONPATH` exactly like `mace_torch_0_3_16`, so the FSCETP environment was left untouched
(its only runtime dep, `e3nn>=0.5.8`, is already satisfied by FSCETP's e3nn 0.5.9).

---

## 2. ICTC algorithm and e3nn/MACE parity (full write-up: `ictd_algorithm_and_parity_notes.md`)

Condensed; see the notes file for derivations, shapes, and code line references.

- **ICTC basis.** A degree-`l` feature is the harmonic (traceless) part of a rank-`l` symmetric
  Cartesian tensor, in a basis `B_l` orthonormalized under the rotation-invariant Gaussian-moment
  Gram `G_l`, then rescaled (`s_l`) to e3nn "component" RMS. Built purely from polynomial algebra —
  **no spherical-harmonic/e3nn call.**
- **Q transform.** A per-degree **orthogonal** `Q_l` (orthogonal-Procrustes fit on 8192 directions,
  `seed=20260426`) satisfies `direction_harmonics(n,l) @ Q_l == Y_l^{e3nn}(n)` to machine precision.
  Block-diagonal `Q`: `x_e3nn = x_ictd @ Q`, `x_ictd = x_e3nn @ Q^T` (right-multiply on the `m` axis;
  `Q^T` for the inverse). Energy/forces/virial are SO(3) invariants → unchanged by `Q`.
- **ICTC CG / U.** `build_cg_tensor` (polynomial multiply + harmonic projection, natural parity) and
  `build_full_cg_tensor_so3` (solves the infinitesimal SO(3) equivariance equations via SVD null
  space — covers antisymmetric paths). The symmetric-contraction `U` is then assembled recursively
  in the ICTC basis (`ictd_u_matrix_so3`) — the **pure-U** path, self-contained, no Wigner call.
- **bridge-U (the parity path).** Keep MACE's native `U` (from `_mace_cg.U_matrix_real`, e3nn) verbatim
  and wrap the native MACE symmetric contraction with the orthogonal `Q` bridge on its input/output
  (`x_e3nn=x_ictd@Q_in`, run native MACE, `out_ictd=out_e3nn@Q_out^T`). Because the contraction is
  multilinear and `Q` orthogonal, this is **algebraically identical to folding `Q` into `U`**
  (`U_ictd = Q_out · U_e3nn · (Q_in⊗…⊗Q_in)`) and reproduces mace-torch to float roundoff. The fold is
  realized at runtime by the feature bridge (the `U` buffers are not mutated → no double-fold on
  reload/AOTI).
- **Converter** (`convert_mace_to_ictd`): `o3.Linear` blocks → effective per-`l` matrix via identity
  probe (`W_eff=block(I)`, exact); `conv_tp` path permutation + per-path scalar calibrated empirically
  (residual `<1e-8`); radial MLP reparameterized (normalize2mom `K` fold); contraction weights copied
  1:1; first-layer `sc0` + scale/shift installed. Float64.
- **Backends:** `ictd-bridge-u` (default, exact-parity conversion target), `native-mace` (reference),
  `cueq` (perf, `<2e-5` fp32), `ictd-pure-u` (**diagnostic, not bit-exact** — converter refuses it).
  Parity is scoped to the **MACE-style baseline configs**, not "all MACE variants".

---

## 3. Parity tests (re-run on the 4090; log: `parity_tests.log`)

| test | result | tolerances |
|---|---|---|
| `test_angular_basis.py` (run as a **script**, not pytest) | **PASS** | f64 `dE=0, dF=4.2e-21`, feature `\|e3nn−ictd@Q\|=8.3e-17`; f32 `dE=0, dF=7.3e-12` |
| `test_mace_converter.py` | **PASS** (1 passed, 58.8 s) | whole-model bridge-U parity rel`\|dE\|<1e-9`, `\|dF\|<1e-6`; rotation-inv `<1e-6` |
| `test_cueq_product_backend.py` | **PASS** (4 passed, 1 skipped, 15.8 s) | cuEq fast vs reference `<2e-5` fp32 |

- `test_angular_basis.py` defines `run()`/`main()` and **no `test_*` functions**, so `pytest` reports
  "no tests ran" — it must be invoked as `python -m mace_ictc.test.test_angular_basis` (done; PASS).
  Not a failure; a harness quirk. *Suggested fix for the repo CI:* add `def test_angular_basis(): main()`.
- The single **skip** in the cueq test is the CPU case (cueq builds CUDA-only kernels when CUDA is
  present, so CPU reference-vs-fast is skipped by design); the CUDA cases run and pass.
- These cover angular, whole-model-conversion, and cueq-product parity. Sanity full-model harness
  (`sanity_full_model/`) ran clean (channels 16, atoms 256) confirming the benchmark harness works.

---

## 4. Existing vs newly produced results

- **Reused (referenced, not re-run):** the model-level throughput study in
  `MACE-ICTC/benchmark_results/` and `docs/figures/backend_{throughput,speedup}_benchmark_channels64.png`
  (RTX 4090, FP32, channels 64, avg directed degree 16 rescaled to 50-neighbor-equivalent atoms; five
  modes incl. e3nn / cuEq / ICTC eager / ICTC compiled / ICTC+cuEq). I did **not** re-run these.
- **Newly produced in this task (all on the 4090):** the three parity logs, the small sanity
  full-model run, and the **entire operator-level benchmark** (ICTC vs cartnn vs e3nn) below — these
  are new and were not part of the existing `benchmark_results/`.

---

## 5. Operator benchmark — ICTC product vs cartnn Cartesian product

*(full numbers: `operator_cartnn_vs_ictd.csv`, `operator_ictd_compiled.csv`;*
*summary: `operator_cartnn_vs_ictd_summary.md`; figures: `figures/operator_*.png`)*

**Matched operator:** the equivariant **tensor product** coupling a hidden node feature
(`0..hidden_lmax`, `C` channels) with the edge angular embedding (`0..max_ell`), per-edge weighted,
over `E` directed edges — i.e. the MACE convolution TP, on the **identical `(l1,l2,l3)` natural-parity
path set** for every backend: `ictd` (`EdgeWeightedPathPreservingTensorProduct`, ICTC `2l+1` basis,
eager), `cartnn` (`o3.TensorProduct` via `cartesian_3j`, full `3**l` Cartesian, codegen-fused), `e3nn`
(`o3.TensorProduct` via `wigner_3j`, spherical `2l+1`, codegen-fused — the MACE-native reference), and
`ictd_compiled` (the ICTC op under `torch.compile`, the deployed form).

**Scope / honesty (key):** this is an *operator-level comparable workload*, **not** exact
apples-to-apples — cartnn stores degree-`l` in `3**l` (vs `2l+1`) and the per-path normalizations
differ, so outputs are not expected to match. **cartnn ships no symmetric-contraction operator** (the
authors explicitly declined to implement ICTC), so the MACE symmetric contraction is **out of scope**;
only the binary tensor product is compared. No chemical-accuracy or model-level claim is made.

**Warmup validity (verified).** The ICTC operator populates `(device,dtype)`-keyed caches
(`_cg_cache_by_dev_dtype`, projector tensors) on its **first** forward — call #1 is 2–22× slower
than warm. A per-call probe (`warmup_curve.log`) shows the plateau is reached by **call #2–3** and
is stable to 0.1% (`median[6:] == median[21:]`). The harness discards `warmup=20` calls before
CUDA-event timing, so all measured ICTC numbers are on the warm plateau (probe warm values match the
CSV to <2%). cartnn/e3nn (codegen-fused) warm in 2–3 calls likewise.

**Headline (channels=64, edges=100000, total_ms = fwd[+bwd]; warm).**

| ratio | l1/1 | l1/2 | l2/2 | l2/3 | l3/3 |
|---|---|---|---|---|---|
| cartnn/e3nn, fp32 fwd | 1.00 | 1.00 | 1.21 | 1.20 | **2.80** |
| cartnn/e3nn, fp32 fwd+bwd | 1.00 | 1.00 | 1.23 | 1.21 | OOM(cartnn) |
| cartnn/e3nn, fp64 fwd+bwd | 1.00 | 1.00 | 1.28 | 1.27 | OOM |
| cartnn/ictd (>1 ⇒ ictd faster), fp32 fwd | 0.22 | 0.40 | 0.57 | 0.59 | **1.20** |
| ictd_compiled/ictd, fp32 fwd | 0.58 | 1.01 | 1.01 | 1.00 | 1.00 |

**What the operator benchmark shows (measured, scoped):**
1. **At low angular order (max_ell ≤ 2):** cartnn ≈ e3nn (the `3**l` vs `2l+1` difference is negligible
   relative to the channel work), while the eager **ICTC operator is 2–11× slower and the most
   memory-hungry** — the ICTC path-preserving conv-TP's overhead/inefficiency dominates here.
2. **At high angular order (l3/3):** cartnn's full `3**l` Cartesian storage becomes a severe penalty —
   cartnn is **2.8× slower than e3nn** and **slower than even eager ICTC** (a clean crossover), and it
   **OOMs at l3/3 fp64** (tried to allocate 11.6 GB for a single einsum) where ICTC (compiled) fits.
3. **e3nn (spherical `2l+1`, the MACE-native reference) is the fastest** of the three at every tested
   point; both Cartesian-basis operators (ICTC intrinsic, cartnn full) trail it at the operator level.
4. **`torch.compile` barely changes the standalone ICTC op** (it graph-breaks on the dict-in/out
   forward): `ictd_compiled ≈ ictd` everywhere except the tiny l1/1 case (0.58×) and reducing peak
   memory enough to complete l3/3 fp64 (287 ms) where eager OOMs. The deployed model's competitiveness
   comes from **full-graph AOTI at the model level** (existing `benchmark_results/`), not from this
   operator in isolation.
5. **OOM cells (RTX 4090 24 GB):** by backend — e3nn 35, cartnn 46, ictd 50 (of 720 cells); the large
   `E=500000`/`fp64`/high-`lmax` corner. cartnn OOMs more than e3nn (`3**l` memory); see
   `operator_cartnn_vs_ictd_summary.md` for the full per-cell list.

**Reconciliation with the model-level `benchmark_results/` (important — prevents misreading this bench).**
This operator bench measures the **isolated** conv tensor product and times the ICTC op **eager**
(`torch.compile` graph-breaks on its dict I/O) against e3nn's **codegen-fused** `o3.TensorProduct`. So
"e3nn fastest, ICTC slower" here is specifically the **un-fused ICTC operator at large scale** — the worst
case, and **not** what the model deploys. The existing model-level study (inference, channels=64,
hidden/ell=2/2) shows the very same eager regime *and* the deployment win:

| atoms | mace-e3nn (eager) | ICTC eager | ICTC **AOTI** | ICTC-eager / e3nn | **ICTC-AOTI / e3nn** |
|---|---|---|---|---|---|
| 512 | 34.2 ms | 14.6 | 3.76 | 2.35× | **9.10×** |
| 2048 | 41.6 | 34.1 | 13.8 | 1.22× | 3.01× |
| 4096 | 65.0 | 68.0 | 27.7 | **0.96×** | 2.35× |
| 8192 | 114.0 | 141.1 | 55.3 | **0.81×** | 2.06× |

- ICTC **eager** whole-model beats mace-e3nn at small atom counts but is **slower at ≥4096** (0.81–0.96×) —
  consistent with this operator bench (large edge counts ≈ large atom counts, where eager ICTC trails e3nn).
- The "MACE-ICTC ≫ mace-e3nn" result is **ICTC AOTI** (2–9×): whole-graph fusion removes the per-layer
  launch overhead that dominates eager mace-torch. (The baseline `mace-e3nn` is **eager**; mace's own
  accelerated `MACE cuEq` path is a flat ~20 ms floor — ICTC-AOTI beats it ≤2048 atoms, 5.4×, but loses
  ≥4096, 0.37×.)
- **Takeaway:** at this scale, end-to-end speed is governed by **whole-graph fusion**, not isolated
  per-operator FLOPs. The §5-headline numbers isolate the operator and time ICTC **eager** — they
  *under-represent* the deployed ICTC. The matched-fusion test below removes that bias.

### Matched-fusion comparison — the fair operator test (`operator_aoti_fwd.csv`, fig `operator_matched_fusion.png`)

Putting the ICTC op at the **same fusion level** as e3nn/cartnn — `torch.compile` on a thin **flat-I/O
wrapper** around the unmodified repo tp (the earlier "torch.compile ≈ no-op" was a graph break at the
**dict-I/O boundary**, *not* the op; a flat wrapper unblocks full fusion, ~3× over eager) — **flips the
verdict for `max_ell ≥ 2`** (channels=64, E=100000, forward-only, total_ms):

**fp32**, channels=64, E=100000, `ICTC compile` total_ms and ratios (>1 ⇒ slower than that backend):

| config | ICTC eager | **ICTC compile** | e3nn | cartnn | compile/e3nn | compile/cartnn |
|---|---|---|---|---|---|---|
| l1/1 | 6.9 | 2.5 | 1.3 | 1.3 | 1.87 | 1.87 |
| l1/2 | 10.4 | 3.4 | 3.8 | 3.8 | **0.90** | **0.90** |
| l2/2 | 32.7 | 10.7 | 13.5 | 16.4 | **0.79** | **0.65** |
| l2/3 | 42.4 | 14.0 | 18.7 | 22.5 | **0.75** | **0.62** |
| l3/3 | 89.8 | 33.9 | 34.8 | 97.6 | 0.97 | **0.35** |

**The result is dtype-dependent — state it precisely:**
- **fp32 (the inference/training regime):** fused ICTC is **faster than e3nn for mid-range angular order**
  (l1/2, l2/2, l2/3 → 0.75–0.90×), **~tied at l3/3** (0.97×), and loses only at the trivial l1/1 (1.87×).
  It beats **cartnn at every `max_ell ≥ 2`** (0.62–0.90×), **crushing it at l3/3** (0.35×, cartnn's `3**l`).
- **fp64:** e3nn's leaner Wigner kernels are **faster than ICTC across the board** (ICTC 1.35–7.2× e3nn);
  ICTC still beats cartnn at l3/3 (0.39×). So no "ICTC beats e3nn" claim holds in fp64.
- **Memory:** at l2/3 E=500000 fp32, eager ICTC **OOMs** while compiled ICTC fits (14.6 GB) and is fastest
  (75.0 vs e3nn 94.0, cartnn 113.2).
- The §5-headline "e3nn fastest, ICTC slower" reflected only the **un-fused** operator. At matched fusion the
  ICTC algebra is **competitive (fp32) — faster than e3nn for l1/2–l2/3** — which is consistent with the
  model-level AOTI win.
- Caveats: `torch.compile` (flat I/O) is the headline fused number; AOTI also fuses (~2× over eager) but was
  flaky in-process (4 export errors) and slower here. At E=500000 `torch.compile` shows variance (AOTI is the
  robust large-E number, e.g. l2/2 E5e5: AOTI 80.3 < cartnn 82.3, e3nn OOM). Per-config
  `torch._dynamo.reset()` is mandatory or dynamo's recompile limit silently falls back to eager.

**Honest framing for the paper (corrected after the matched-fusion test):** *"At matched fusion level
(`torch.compile`, forward-only, channels=64), the **fp32** ICTC tensor-product operator is faster than the
spherical e3nn reference for mid-range angular order (l1/2, l2/2, l2/3 → 0.75–0.90×), about tied at l3/3
(0.97×), and slower only at the trivial l1/1; it beats the cartnn Cartesian product at every `max_ell ≥ 2`
(0.62–0.90×, down to 0.35× at l3/3) with lower peak memory. In **fp64**, e3nn is faster than ICTC
throughout. The un-fused (eager) numbers in §5 are slower purely because of missing operator fusion, not the
ICTC algebra; consistently, the model-level MACE-ICTC advantage over eager mace-torch (2–9×) is whole-graph
AOTI fusion."* All of this is an operator-level **comparable workload** (same `(l1,l2,l3)` path set / edge
batch), **not** exact apples-to-apples (cartnn `3**l` vs `2l+1`), and makes **no** accuracy or training
claim. cartnn ships no contraction operator, so the MACE symmetric contraction was not benchmarked.

---

## 6. File manifest (`/tmp/mace_ictc_cartnn_bench_20260614_220608/`)

| file | what |
|---|---|
| `env.txt` | machine + library versions |
| `ictd_algorithm_and_parity_notes.md` | **deliverable 1** — formula-level ICTC + parity write-up |
| `parity_tests.log` | deliverable — parity test output (angular/converter/cueq) |
| `sanity_full_model/` , `sanity_full_model.log` | harness sanity run |
| `operator_bench.py` | operator benchmark harness (e3nn/cartnn/ictd) |
| `operator_bench_compiled.py` | torch.compile ICTC companion pass |
| `operator_cartnn_vs_ictd.csv` | **deliverable 2** — raw operator results |
| `operator_ictd_compiled.csv` | compiled-ICTC raw results |
| `operator_cartnn_vs_ictd_summary.md` | operator summary + caveats |
| `plot_operator_cartnn_vs_ictd.py` , `figures/operator_throughput.png` , `figures/operator_speedup.png` | plots |
| `summarize_operator.py` | summary generator |
| `src/cartnn/` | cloned cartnn @ 4d0dc38 |
| `operator_bench_aoti.py` , `operator_aoti_fwd.csv` , `operator_aoti_summary.md` | matched-fusion (torch.compile/AOTI) forward-only operator comparison |
| `plot_fp32_paper.py` , `figures/operator_fp32_matched_fusion.{pdf,png,svg}` | **paper fp32 figure** (vector PDF) |
| `paper_supplement.tex` | **paper-ready LaTeX**: ICTC methods + parity + fp32 operator-efficiency (with Table + figure) |
| `FINAL_REPORT.md` | this report |

---

## 7. Claims discipline (what is and isn't asserted)

**Asserted (measured / by construction):**
- "The bridge-U path is designed to preserve native MACE/e3nn numerical convention under the tested
  configurations" — and it does, to float roundoff (§3).
- "Under the measured operator workloads, ICTC is faster/slower than cartnn by X" — see §5 (per config).

**Not asserted:** that all MACE variants convert exactly; that all ICTC paths are e3nn-equivalent
(pure-U is diagnostic); that ICTC universally beats cartnn/CACE/TACE; any model-level accuracy or
training-superiority conclusion.
