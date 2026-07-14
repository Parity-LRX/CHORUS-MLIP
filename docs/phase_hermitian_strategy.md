# PEMP / Hermitian phase branch: implementation and research plan

## What is implemented

This branch adds an opt-in phase-modulated atomic-environment path to
`PureCartesianICTDFix`. The original interaction and product path is unchanged;
the new path is attached only to the last interaction and is added as a scalar
residual after the ordinary MACE product.

For a real equivariant edge message `b_ij`, the implementation predicts one
invariant scalar phase per edge and forms two real streams:

\[
x_i=\sum_j a_{ij}\cos\theta_{ij}\,b_{ij},\qquad
y_i=\sum_j a_{ij}\sin\theta_{ij}\,b_{ij}.
\]

It then computes, channel-wise and for every angular degree,

\[
\rho_i=\langle x_i,x_i\rangle+\langle y_i,y_i\rangle,
\]

followed by an element-conditioned scalar projection and a learnable residual
scale. Expanding the contraction gives the intended factor

\[
\sum_{jk}a_{ij}a_{ik}\cos(\theta_{ij}-\theta_{ik})
\langle b_{ij},b_{ik}\rangle.
\]

No complex PyTorch dtype is used. This is mathematically equivalent to a
complex feature for this Hermitian contraction, but remains compatible with the
existing real ICTD operators, autograd, make-fx, and checkpoint deployment.

The phase network consumes only the source and destination `l=0` node blocks
and radial edge features. Therefore its output is an E(3)-invariant scalar. In a
multi-interaction model the final-layer scalar node features already contain
environment information, unlike a first-layer species-and-distance-only gate.

## Configuration

Training flags:

```text
--phase-mode none|final-scalar-residual
--phase-hidden-channels 32
--phase-scale-init 0.05
--phase-amplitude unit|softplus
```

Example:

```bash
python -m mace_ictc.cli.train \
  <the existing data and training arguments> \
  --phase-mode final-scalar-residual \
  --phase-hidden-channels 32 \
  --phase-scale-init 0.05 \
  --phase-amplitude unit
```

Matched whole-model overhead on the 4090/FSCETP environment:

```bash
/home/ylzhang/micromamba/envs/FSCETP/bin/python \
  -m mace_ictc.bench.bench_phase_hermitian \
  --device cuda --dtype float32 --product-backend ictd-bridge-u \
  --channels 64 --hidden-lmax 1 --max-ell 2 \
  --atoms-list 128,512 --include-makefx
```

Matched rMD17 falsification run. With no overrides this uses the archived
baseline protocol: aspirin, ethanol and benzene; 64 channels; 300 epochs; and
seeds `20260616,20260617,20260618`:

```bash
PYTHON_BIN=/home/ylzhang/micromamba/envs/FSCETP/bin/python \
MACE_ICTC_REPO=/home/ylzhang/MACE-ICTC-Phase \
DATA_ROOT=/tmp/mace_ictd_public_md17 \
benchmarks/paper/scripts/training/run_phase_md17_matrix.sh
```

For a separately labelled smoke run, override `SEEDS`, `EPOCHS`, and optionally
`MAX_STEPS`; those results are not comparable with the archived 300-epoch
baseline.

`none` is the default and introduces no phase parameters or state-dict keys.
Phase mode is deliberately rejected when `--attn-heads` is nonzero in v1. Both
operators reweight neighbors, so combining them before separate ablations would
make the result hard to interpret.

## What this does not establish

The construction is globally U(1)-invariant at the Hermitian contraction, but
it is not yet a local gauge theory: there is no node-wise gauge transformation
law and no learned edge connection that transports phases between gauges.
Calling it a “learnable gauge structure” without this qualification would
overstate the mathematics.

The current contraction is a diagonal power spectrum in a learned channel
basis, not an explicit full `n x n'` density matrix. The preceding learned
message linear map can rotate channels and recover useful cross-channel
quadratic forms at O(C), whereas a literal full density matrix costs O(C²).
If the diagonal version is positive but saturates, a low-rank cross-channel
Hermitian contraction is the next justified extension.

The phase parameterization can also behave like a two-real-channel gate. U(1)
invariance and equivariance alone do not prove that the model learns chemically
meaningful “interference.” That claim requires matched controls and out-of-domain
evidence.

## Required ablations

Use identical data splits, seeds, parameter budgets, optimizer schedules, and
early-stopping rules. Report mean and confidence interval over at least five
seeds.

1. Existing model: `phase-mode=none`.
2. Hermitian unit phase: `final-scalar-residual`, `amplitude=unit`.
3. Hermitian learned amplitude plus phase: `amplitude=softplus`.
4. Parameter-matched real scalar gate residual.
5. Parameter-matched unconstrained two-real-channel Hermitian residual.
6. A wider/deeper baseline with the same parameter and wall-clock budget.

Controls 4–6 are not yet exposed as CLI modes in this first implementation and
must be added before making a representational claim. In particular, comparing
only against the smaller original model cannot distinguish the proposed
inductive bias from added capacity.

Primary metrics should include energy/force/stress MAE, learning curves versus
training-set size, seed variance, throughput, peak memory, and force continuity
near the cutoff. Test `rMD17` only as a debugging/sample-efficiency stage. The
stronger test is a fixed, realistically sized subset of QM7-X or a chemically
heterogeneous transition-metal/catalysis set with a genuine held-out chemistry
or coordination split. OC20 is useful only if compute and split design are
controlled; it is too large for the first falsification loop.

## Phase diagnostics

Analyze phase differences rather than absolute phases:

\[
\Delta\theta_{jk}=\operatorname{atan2}
(\sin(\theta_{ij}-\theta_{ik}),\cos(\theta_{ij}-\theta_{ik})).
\]

Track circular variance, correlation with neighbor distance/species/angles,
and stability across seeds. Do not begin with a regularizer that forces phases
to align: `1-cos(delta theta)` biases the model toward the real baseline and can
erase the effect being tested. Add weak regularization only in response to a
measured instability, and ablate it separately. Restricting raw theta to
`[-pi, pi]` is also unnecessary because only sine and cosine are consumed.

## Go/no-go criteria

Continue to a low-rank full Hermitian density only if the phase model improves
sample efficiency or held-out-chemistry force error beyond seed uncertainty and
the gain survives the parameter-matched controls. Stop or reframe the project if
the gain disappears against the unconstrained two-real-channel or widened
baseline, phases collapse across seeds, or deployment cost is disproportionate
to accuracy. In that case the honest conclusion is an additional quadratic
neighbor-correlation branch, not evidence for a special phase inductive bias.
