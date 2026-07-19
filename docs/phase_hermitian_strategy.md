# CHORUS: implementation and research plan

## What is implemented

CHORUS adds opt-in phase-coherent atomic-environment paths to
`PureCartesianICTDFix`. The original interaction and product path remains as a
residual backbone. With `phase-scope=final`, the phase path is attached only to
the last interaction. With `phase-scope=persistent`, every interaction produces
a charged doublet, the state is recurrently mixed across depth, and a neutral
Hermitian density is fed into every symmetric contraction. Both scopes can use
either a scalar density or a rank-reduced full-L equivariant density.

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

The scalar mode follows this with an element-conditioned scalar projection and
a learnable residual scale. Expanding the contraction gives the intended factor

\[
\sum_{jk}a_{ij}a_{ik}\cos(\theta_{ij}-\theta_{ik})
\langle b_{ij},b_{ik}\rangle.
\]

No complex PyTorch dtype is used. This is mathematically equivalent to a
complex feature for this Hermitian contraction, but remains compatible with the
existing real ICTD operators, autograd, make-fx, and checkpoint deployment.

The full-L mode first projects each complex l block from C channels to R latent
orbitals using the same real projection for both streams. It then retains every
natural-parity Hermitian CG path `(l1,l2)->L` supported by the backbone,
including both independent real and imaginary components for `l1<l2`. An
element-conditioned map returns every L block to C channels before adding the
complete residual to the ordinary pre-product message. This avoids a literal
O(C^2) channel density while allowing direct phase/main-trunk angular coupling.

The phase network consumes only the source and destination `l=0` node blocks
and radial edge features. Therefore its output is an E(3)-invariant scalar. In a
multi-interaction model the final-layer scalar node features already contain
environment information, unlike a first-layer species-and-distance-only gate.

For the persistent scope, let `z_t=x_t+i y_t` be the charged atomic state formed
at interaction `t`, and let `p_t` be that layer's incoming phase-modulated edge
aggregation. The implemented update is

\[
z_t=g_t A_t z_{t-1}+(1-g_t)B_t p_t,
\]

where `A_t` and `B_t` are SO(3)-equivariant channel maps shared by the real and
imaginary streams, and `g_t` is a learned sigmoid gate for each angular degree.
The first layer initializes `z_0=p_0`. A common phase rotation of every charged
source commutes with this recurrence, so each per-layer Hermitian contraction
is globally U(1)-invariant. The recurrence is an atomic memory across network
depth; it is not charged spatial message transport between node gauges.

## Configuration

Training flags:

```text
--phase-mode none|final-scalar-residual|final-full-l-residual
--phase-hidden-channels 32
--phase-scale-init 0.05
--phase-amplitude unit|softplus
--phase-placement post-product|pre-product-l0|pre-product-full-l|pre-and-post
--phase-density-rank 8
--phase-scope final|persistent
```

Example:

```bash
python -m mace_ictc.cli.train \
  <the existing data and training arguments> \
  --phase-mode final-scalar-residual \
  --phase-hidden-channels 32 \
  --phase-scale-init 0.05 \
  --phase-amplitude unit \
  --phase-placement post-product \
  --phase-scope final
```

`post-product` is the backward-compatible v1 path: the neutral Hermitian
feature is added after the final symmetric contraction. `pre-product-l0`
injects the same neutral feature into the scalar block of the final interaction
message before the ordinary MACE product. `pre-and-post` enables both paths as
a performance-oriented ablation. The pre-product modes let the configured
MACE correlation mix ordinary messages with the learned Hermitian density.
They do not by themselves propagate a charged feature between interaction
layers; that recurrence is enabled only by `--phase-scope persistent`.

`final-full-l-residual` requires `--phase-placement pre-product-full-l`. In the
current `max_ell=2` configuration it constructs neutral L=0,1,2 density blocks,
projects them back to the corresponding 64-channel main-trunk blocks, and then
runs the unchanged MACE symmetric contraction. `--phase-density-rank` controls
the latent channel rank (8 in the first experiment).

Persistent scalar mode requires `--phase-placement pre-product-l0`; persistent
full-L mode requires `--phase-placement pre-product-full-l`. These restrictions
ensure that every layer's neutral density actually reaches its symmetric
contraction. For example:

```bash
python -m mace_ictc.cli.train \
  <the existing data and training arguments> \
  --phase-mode final-full-l-residual \
  --phase-amplitude softplus \
  --phase-placement pre-product-full-l \
  --phase-density-rank 8 \
  --phase-scope persistent
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
MACE_ICTC_REPO=/home/ylzhang/CHORUS-MLIP \
DATA_ROOT=/tmp/mace_ictd_public_md17 \
benchmarks/paper/scripts/training/run_phase_md17_matrix.sh
```

For a separately labelled smoke run, override `SEEDS`, `EPOCHS`, and optionally
`MAX_STEPS`; those results are not comparable with the archived 300-epoch
baseline.

`none` is the default and introduces no CHORUS parameters or state-dict keys.
Phase mode is deliberately rejected when `--attn-heads` is nonzero in v1. Both
operators reweight neighbors, so combining them before separate ablations would
make the result hard to interpret.

## What this does not establish

The construction, including the persistent recurrence, is globally
U(1)-invariant at the Hermitian contractions, but it is not a local gauge
theory: there is no node-wise gauge transformation law and no learned edge
connection that transports phases between gauges. Independent phase rotations
of different layers are also not symmetries of the recurrence; the layers share
one global phase frame.
Calling it a “learnable gauge structure” without this qualification would
overstate the mathematics.

The scalar contraction is a diagonal power spectrum in a learned channel basis.
The full-L extension remains a rank-R factorization rather than an explicit
full `n x n'` density matrix. It therefore cannot represent an arbitrary C-by-C
Hermitian kernel unless R approaches C; this limitation is intentional and must
be included in any capacity-matched interpretation.

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

Continue scaling the low-rank full-L Hermitian density only if it improves
sample efficiency or held-out-chemistry force error beyond seed uncertainty and
the gain survives the parameter-matched controls. Stop or reframe the project if
the gain disappears against the unconstrained two-real-channel or widened
baseline, phases collapse across seeds, or deployment cost is disproportionate
to accuracy. In that case the honest conclusion is an additional quadratic
neighbor-correlation branch, not evidence for a special phase inductive bias.
