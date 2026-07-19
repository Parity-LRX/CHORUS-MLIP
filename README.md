# CHORUS

### U(1)-Structured Phase-Coherent Hermitian Aggregation for Equivariant Interatomic Potentials

**CHORUS** is a research framework for phase-coherent neighbor aggregation in
equivariant machine-learned interatomic potentials. Instead of assigning only a
real scalar weight to each neighbor, CHORUS predicts an invariant amplitude and
phase, coherently sums the resulting equivariant messages, and converts their
relative-phase correlations back to ordinary real irreducible representations
through Hermitian Clebsch-Gordan contractions.

The current reference implementation is built on our
[MACE-ICTC](https://doi.org/10.5281/zenodo.20690950) framework. MACE-ICTC
provides the ICTC representation, MACE-compatible many-body backbone, training,
conversion, compilation, and deployment stack. CHORUS contributes the coherent
aggregation operator, the full-\(L\) low-rank Hermitian density, and the
associated controls and benchmarks.

The name denotes **C**oherent **H**ermitian **O**(3) **R**epresentations with
**U**(1)-**S**tructured aggregation. The learned \(U(1)\) phase action is the
central operator rather than a secondary implementation option.

> **Status:** research preview. The primary final-layer full-\(L\) operator is
> implemented and benchmarkable. The persistent cross-layer charged stream is
> experimental. Interfaces and checkpoint metadata may change before the
> preprint release.

## Why coherent aggregation?

A conventional equivariant layer constructs a local environment by summing real
edge messages,

\[
A_i^{nlm}=\sum_{j\in\mathcal N(i)} b_{ij}^{nlm}.
\]

Positive scalar attention can change how strongly each neighbor contributes,
but it does not give different edges an independently learned relative phase.
CHORUS instead forms

\[
\psi_i^{nlm}
=
\sum_{j\in\mathcal N(i)}
a_{ij}e^{\mathrm i\theta_{ij}}b_{ij}^{nlm},
\qquad
b_{ij}^{nlm}
=
R_n(r_{ij})Y_l^m(\hat{\mathbf r}_{ij})h_j.
\]

The phase network consumes only invariant edge context: radial features and
the source and destination \(l=0\) node features. It therefore does not inject
an orientation-dependent scalar into the equivariant backbone.

CHORUS does not send \(\psi\) directly to the energy readout. It constructs a
neutral Hermitian density,

\[
\rho_i^{LM}
=
\operatorname{CG}_{LM}
\left(\psi_i\otimes\psi_i^\dagger\right),
\]

whose pair expansion contains

\[
\sum_{j,k}
a_{ij}a_{ik}
\cos\!\left(\theta_{ij}-\theta_{ik}\right)
\operatorname{CG}_{LM}
\left(b_{ij}\otimes b_{ik}\right).
\]

The \(j\ne k\) terms are the central mechanism: geometry-dependent neighbor
correlations are modulated by learned constructive or destructive
interference. The neutral full-\(L\) density is injected before the unchanged
symmetric contraction, so its \(L>0\) components can mix with the ordinary
equivariant trunk at the configured correlation order.

## Real-doublet implementation

The implementation does not require a complex PyTorch dtype. It represents
\(\psi=x+\mathrm i y\) using two real streams,

\[
x_i=\sum_j a_{ij}\cos\theta_{ij}\,b_{ij},
\qquad
y_i=\sum_j a_{ij}\sin\theta_{ij}\,b_{ij},
\]

and evaluates the Hermitian contractions in real arithmetic. The same
equivariant channel map is used for both streams. This preserves compatibility
with the existing ICTC operators, autograd, `make_fx`, AOTInductor, and the
deployment stack.

For the full-\(L\) path, \(C\) channels are first projected to \(R\) latent
orbitals, with the same projection applied to the two streams. Natural-parity
Hermitian CG paths are then contracted and mapped back to the trunk channels.
The default \(R=8\) factorization avoids explicitly materializing a
\(C\times C\) density matrix.

## Symmetry statement

CHORUS is designed to preserve the spatial symmetry of the underlying
interatomic potential:

- the learned edge coefficient is an \(E(3)\)-invariant scalar;
- both real streams transform under the same \(O(3)\) irreducible
  representations as the original edge message;
- Hermitian CG contractions return neutral real equivariant features;
- a common rotation of every charged doublet in its internal two-dimensional
  plane leaves the Hermitian density unchanged.

The last property is a **global U(1) invariance of the charged stream**, not a
complete local gauge theory. There is no node-wise gauge transformation law and
no learned edge connection that transports a phase between independent local
frames. The learned phase is also not an atomic charge, an electronic orbital,
or a physical quantum wavefunction.

## Data flow

```text
atomic graph
    │
    ├── radial basis × spherical/ICTC angular basis × neighbor features
    │
    ├── invariant coefficient network
    │       └── amplitude a_ij and phase θ_ij
    │
    ├── coherent aggregation
    │       └── ψ_i = Σ_j a_ij exp(i θ_ij) b_ij
    │
    ├── low-rank full-L Hermitian density
    │       └── CG(ψ_i ⊗ ψ_i†)
    │
    ├── real equivariant residual
    │
    ├── unchanged symmetric contraction and interaction backbone
    │
    └── atomic energy readout → total energy → conservative forces
```

## Implemented modes

| Mode | Purpose |
| --- | --- |
| `phase-mode=none` | Ordinary MACE-ICTC aggregation |
| `final-scalar-residual` | Scalar Hermitian residual for minimal experiments |
| `final-full-l-residual` | Low-rank full-\(L\) Hermitian residual; primary CHORUS operator |
| `phase-density-pairs=diagonal` | Retain only \(j=k\) self-density terms |
| `phase-density-pairs=full` | Retain \(j=k\) and coherent \(j\ne k\) correlations |
| `phase-coefficient=positive` | Positive real-gate control |
| `phase-coefficient=signed` | Signed real-gate control |
| `phase-coefficient=cartesian` | Unconstrained matched two-real-channel control |
| `phase-context=radial` | Distance-conditioned phase control |
| `attn-heads>0` | Neighbor-attention reference with CHORUS disabled |
| `phase-scope=persistent` | Experimental charged memory across interaction depth |

The controls are not interchangeable conclusions. In particular, the Cartesian
doublet has similar raw capacity to the polar parameterization and tests the
value of its inductive bias; the diagonal density removes cross-neighbor
coherence; and the current attention implementation represents a commonly used
positive normalized reweighting mechanism rather than every possible form of
attention.

## Installation

```bash
git clone https://github.com/Parity-LRX/CHORUS-MLIP.git
cd CHORUS-MLIP
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pyg]"    # torch-scatter / torch-cluster acceleration
pip install -e ".[cue]"    # cuEquivariance product backend
pip install -e ".[e0]"     # fitted-E0 CSV support
pip install -e ".[full]"   # all optional Python extras
```

The distribution is named `chorus-mlip`, while the Python namespace remains
`mace_ictc` for MACE-ICTC checkpoint, extension, and deployment compatibility.
The runtime requires Python >= 3.9, PyTorch >= 2.4, and
`e3nn >= 0.4.4, < 0.6`. Native checkpoint conversion has been validated against
`mace==0.3.16`; the standalone CHORUS training path does not require
`mace-torch`.

## Quick start

The primary final-layer, content-conditioned, full-\(L\) configuration is:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 64 --lmax 1 --max-ell 2 \
  --num-interaction 2 --correlation 3 \
  --function-type bessel --num-basis 8 \
  --product-backend ictd-bridge-u --angular-basis ictd \
  --phase-mode final-full-l-residual \
  --phase-amplitude softplus \
  --phase-coefficient polar \
  --phase-context content \
  --phase-density-pairs full \
  --phase-placement pre-product-full-l \
  --phase-scope final \
  --phase-density-rank 8 \
  --phase-hidden-channels 32 \
  --phase-scale-init 0.05 \
  --device cuda --dtype float32 \
  --checkpoint chorus.pth
```

Add the dataset-specific optimizer, loss, atomic-energy, batching, and stopping
arguments required by the training protocol. The archived rMD17 drivers contain
complete matched configurations.

To isolate the non-diagonal coherent terms, change only:

```bash
--phase-density-pairs diagonal
```

To run the attention reference:

```bash
--phase-mode none --attn-heads 4
```

CHORUS and attention are deliberately mutually exclusive in the current
implementation so their effects remain identifiable.

## Reproducible experiments

The main experiment drivers are:

- [`run_phase_md17_matrix.sh`](benchmarks/paper/scripts/training/run_phase_md17_matrix.sh):
  matched rMD17 mode/seed matrix;
- [`run_phase_confirmatory_multisystem.sh`](benchmarks/paper/scripts/training/run_phase_confirmatory_multisystem.sh):
  confirmatory molecular systems;
- [`run_phase_confirmatory_water.sh`](benchmarks/paper/scripts/training/run_phase_confirmatory_water.sh):
  liquid-water confirmation;
- [`analyze_md17_convergence.py`](benchmarks/paper/scripts/training/analyze_md17_convergence.py):
  checkpoint-aligned convergence and MAE summaries;
- [`bench_phase_hermitian.py`](mace_ictc/bench/bench_phase_hermitian.py):
  eager and `make_fx` throughput comparison.

Reference throughput command:

```bash
python -m mace_ictc.bench.bench_phase_hermitian \
  --device cuda --dtype float32 \
  --product-backend ictd-bridge-u \
  --channels 64 --hidden-lmax 1 --max-ell 2 \
  --atoms-list 128,512 --include-makefx
```

The archived GPU measurements use an RTX 4090 and the FSCETP environment.
Large checkpoints and trajectories are not stored in Git.

For accuracy comparisons, use identical data splits, seeds, parameter budgets,
optimizer schedules, and stopping rules. Select one checkpoint by validation
loss and report energy and force MAE from that same checkpoint. Selecting the
best energy and best force from different epochs produces an unattainable
composite model and is not the primary reporting rule.

The minimum comparison set for the central mechanism is:

1. ordinary MACE-ICTC;
2. CHORUS full-\(L\), full \(j,k\) density;
3. diagonal \(j=k\) density;
4. matched neighbor attention.

Positive, signed, Cartesian, radial-only, and persistent controls should be
reported as extended ablations. Current validation results are preliminary;
held-out test evaluation and uncertainty across seeds are required before
making a general accuracy claim.

## MACE-ICTC compatibility and deployment

CHORUS retains the underlying MACE-ICTC capabilities:

- ICTC/e3nn basis correspondence and strict supported-model conversion;
- H5 energy/force/stress training with SWA, EMA, resume, and optional
  `make_fx` compilation;
- ASE calculators, AOTInductor packages, and LAMMPS `USER-MFFTORCH`;
- optional cuEquivariance products;
- learned reciprocal-space electrostatics and anisotropic many-body
  dispersion.

Detailed backbone and deployment documentation remains available in the
[English user manual](docs/USER_MANUAL.md) and
[中文使用说明](docs/USER_MANUAL.zh-CN.md). These manuals retain the
MACE-ICTC name because they document the compatibility backend rather than the
new coherent operator.

## Limitations

- The current evidence does not establish that learned phases correspond to a
  unique chemical observable.
- A polar doublet can have similar capacity to a parameter-matched Cartesian
  two-real-channel model; performance differences must therefore be established
  empirically rather than inferred from notation.
- Rank-\(R\) full-\(L\) density is a factorization, not an unrestricted
  \(C\times C\) Hermitian kernel.
- The persistent stream is an across-depth atomic memory, not charged spatial
  transport between local gauges.
- Small molecular benchmarks are useful for falsification and sample-efficiency
  studies but are insufficient to establish universal MLIP superiority.
- The reference implementation is currently coupled to the MACE-ICTC
  backbone. Portability to other equivariant backbones is a design goal, not
  yet a benchmarked result.

## Repository layout

```text
CHORUS-MLIP/
  mace_ictc/                 # compatible implementation namespace
    models/                  # ICTC backbone and CHORUS operators
    cli/                     # training, conversion, and export
    training/                # force trainer and compilation
    evaluation/              # ASE integration
    bench/                   # operator and whole-model benchmarks
    test/                    # symmetry and numerical tests
  benchmarks/paper/          # experiment scripts and lightweight records
  docs/                      # CHORUS strategy and MACE-ICTC manuals
  lammps_user_mfftorch/      # LAMMPS deployment package
```

## Citation and lineage

The CHORUS preprint citation will be added when it is released. Until then,
please cite the MACE-ICTC software record:

```text
MACE-ICTC. Zenodo. https://doi.org/10.5281/zenodo.20690950
```

CHORUS and MACE-ICTC build on the MACE architecture:

> I. Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural
> Networks for Fast and Accurate Force Fields*, NeurIPS 2022.

See [`NOTICE.md`](NOTICE.md) for software lineage and attribution.

## License

The code is released under the MIT License. The license and attribution apply
to source code; separately distributed pretrained models or datasets may use
different terms.
