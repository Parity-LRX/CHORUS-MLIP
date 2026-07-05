# MACE-ICTC

**DOI:** [10.5281/zenodo.20690950](https://doi.org/10.5281/zenodo.20690950)

MACE-ICTC is a standalone implementation of MACE in an Irreducible Cartesian
Tensor Decomposition (ICTC) basis. It keeps the MACE interaction/readout
semantics, but stores equivariant angular features in compact Cartesian
irreducible blocks and uses fixed basis-change operators to communicate with
the original e3nn/MACE convention.

The repository is intended for three workflows:

- train MACE-style force fields directly in the ICTC basis;
- convert compatible native `mace-torch` checkpoints into MACE-ICTC checkpoints;
- export trained or converted models for ASE, AOTInductor, and LAMMPS deployment.

Full manuals:

- English: [docs/USER_MANUAL.md](docs/USER_MANUAL.md)
- 中文: [docs/USER_MANUAL.zh-CN.md](docs/USER_MANUAL.zh-CN.md)

## What Is Included

- MACE-compatible ICTC model: `PureCartesianICTDFix`.
- Fixed `Q`/`U` basis bridges for ICTC/e3nn correspondence.
- H5 training with energy/force/stress losses, SWA/EMA, resume, and optional
  `make_fx`/Inductor compilation.
- Native `mace-torch` checkpoint conversion for supported `ScaleShiftMACE`
  models.
- ASE, AOTInductor `.pt2`, and LAMMPS `USER-MFFTORCH` deployment.
- Optional cuEquivariance product backend.
- Long-range interactions: learned reciprocal-space electrostatics
  (periodic/slab, multipole sources) and anisotropic many-body dispersion (MBD),
  trained end-to-end and deployable to LAMMPS.
- Curated benchmark records under [benchmarks/paper](benchmarks/paper).

## Installation

```bash
cd /path/to/MACE-ICTC
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pyg]"    # torch-scatter / torch-cluster acceleration
pip install -e ".[cue]"    # cuEquivariance product backend
pip install -e ".[e0]"     # pandas support for fitted E0 CSV files
pip install -e ".[full]"   # all optional Python extras
```

Runtime expectations: Python >= 3.9, PyTorch >= 2.4, `e3nn >= 0.4.4, < 0.6`.
PyTorch >= 2.7 is recommended for AOTInductor and `make_fx`; CUDA is required
for cuEquivariance and the main GPU benchmark paths. An optional compiled ICTC
tensor-product extension can be built with:

```bash
MFF_BUILD_ICTD_TP_EXT=1 pip install -e .
```

The pure-Python/PyTorch path works without this extension. After installation,
the console scripts `mff-convert-mace`, `mff-export-aoti`, `mff-export-core`,
and `mff-lammps` are available. The examples below use `python -m ...` so they
also work from a source checkout before the script PATH is refreshed.

## Quick Check

```python
import torch
from mace_ictc.synthetic import build_model, make_fixed_graph, compute_energy_forces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(
    channels=64,
    lmax=2,
    num_interaction=2,
    correlation=2,
    route="baseline",
    product_backend="ictd-bridge-u",
    dtype=torch.float64,
    device=device,
)

graph = make_fixed_graph(
    num_nodes=128,
    avg_degree=24,
    dtype=torch.float64,
    device=device,
)

energy, forces, atomic_energy = compute_energy_forces(
    model, graph, create_graph=False
)
```

The model forward signature is:

```python
model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
```

where `A` is a tensor of atomic numbers, not species indices.

## Choosing a Runtime Mode

| Goal | Recommended mode |
| --- | --- |
| Conservative MACE parity or native MACE conversion | `product_backend=ictd-bridge-u`, `angular_basis=ictd` |
| Fast training | `product_backend=cueq`, `angular_basis=e3nn`, `--train-makefx-compile` |
| Fast AOTI inference with cuEq available | `python -m mace_ictc.cli.export_aoti_core --cueq-product --angular-basis e3nn` |
| Strictest deployment path | export the checkpoint without product replacement |
| Debug/reference comparisons | `native-mace` or `ictd-pure-u`, depending on the question being tested |

Important distinction:

- `ictd-bridge-u` is the main MACE-correspondence path. It preserves the source
  MACE product convention through the ICTC/e3nn basis bridge.
- `cueq + angular_basis=e3nn` is the main performance path. It folds compatible
  fixed angular operators once and uses cuEquivariance for the product block.
- `ictd-pure-u` is useful for diagnostics and ICTC-native experiments, but it is
  not the default exact native-MACE conversion path.

## Training From Scratch

The trainer consumes preprocessed H5 files such as:

```text
DATA/
  processed_train.h5
  processed_val.h5
```

Bridge-U training keeps the conservative MACE-correspondence product path:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 64 --lmax 2 --max-ell 2 \
  --num-interaction 2 --correlation 2 \
  --function-type bessel \
  --product-backend ictd-bridge-u \
  --epochs 300 --batch-size 4 \
  --energy-weight 1.0 --force-weight 10.0 --stress-weight 0.0 \
  --lr 1e-3 --min-lr 1e-6 --lr-scheduler plateau \
  --swa --start-swa 225 --swa-lr 1e-4 \
  --device cuda --dtype float64 \
  --checkpoint model_bridge_u.pth
```

The performance path uses cuEquivariance products and compiles the force step:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 64 --lmax 2 --max-ell 2 \
  --num-interaction 2 --correlation 2 \
  --function-type bessel \
  --product-backend cueq --angular-basis e3nn \
  --train-makefx-compile --makefx-buckets 6 \
  --epochs 300 --batch-size 8 \
  --energy-weight 1.0 --force-weight 10.0 --stress-weight 0.0 \
  --lr 1e-3 --min-lr 1e-6 --lr-scheduler cosine \
  --swa --start-swa 225 --swa-lr 1e-4 \
  --ema-decay 0.999 \
  --device cuda --dtype float32 \
  --checkpoint model_cueq_e3nn_makefx.pth
```

`--epochs` and `--max-steps` are independent stop conditions; if both are set,
training stops at the first one reached. MACE-style ScaleShift is enabled by
default through `--scaling rms_forces_scaling`. Stage Two/SWA, EMA, optimizer
parameters, loss weights, LR schedules, stress, resume, and E0 controls are
documented in [docs/USER_MANUAL.md](docs/USER_MANUAL.md).

## Convert a Native MACE Checkpoint

MACE-ICTC can import compatible native `mace-torch` `ScaleShiftMACE` objects:

```bash
python -m mace_ictc.cli.convert_mace \
  --mace-model mace.model \
  --out mace_ictc.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

The input must be a `torch.save(model)` object, not a raw `state_dict`.

Conservative AOTI export from the converted checkpoint:

```bash
python -m mace_ictc.cli.export_aoti_core \
  --checkpoint mace_ictc.pth \
  --elements H,C,N,O \
  --out mace_ictc.pt2 \
  --dynamic \
  --embed-e0
```

Performance-oriented cuEq export:

```bash
python -m mace_ictc.cli.export_aoti_core \
  --checkpoint mace_ictc.pth \
  --elements H,C,N,O \
  --out mace_ictc_cueq_e3nn.pt2 \
  --dynamic \
  --cueq-product \
  --angular-basis e3nn
```

Supported conversion is intentionally structure-based and strict. The current
converter targets the `ScaleShiftMACE` layout used in this repository's tests
and benchmarks, validated with `mace==0.3.16` and `e3nn<0.6`. Newer MACE
releases may work when the saved object layout remains compatible, but arbitrary
research forks, raw state dicts, pair-repulsion variants, and changed readout or
radial layouts are rejected rather than silently converted.

For exact MACE correspondence, the converted ICTC model must preserve the source
model's structural options, including `use_reduced_cg`.

OFF23 smoke test, RTX 4090, 2026-06-18: `MACE-OFF23_small.model` was converted
to bridge-U ICTC and compared against native `mace-torch` in float64. On a
benzene trajectory, same-frame maximum differences were `2.73e-12 eV` in energy
and `4.44e-15 eV/A` in force. A fresh float32 static-6 AOTI export loaded in
LAMMPS `mff/torch`; LAMMPS `pe=-6633.036` matched the Python checkpoint energy
`-6633.03613281 eV`, and LAMMPS `fmax=11.767612` matched the Python maximum
absolute force component `11.76760674 eV/A`. Some old OFF23 pickle files require
a matching historical `mace-torch`/`e3nn` environment for the loading step before
conversion.

## Long-Range Interactions (Electrostatics and Dispersion)

Beyond the message-passing cutoff, MACE-ICTC provides two complementary
long-range channels. Both are differentiable and trained end-to-end from the
same energy/force/stress losses — each module is initialized near zero, so
enabling it starts close to the short-range model and learns the correction.
Neither is a fixed-parameter analytic post-correction, and both deploy through
ASE, AOTInductor, and LAMMPS. See [docs/USER_MANUAL.md](docs/USER_MANUAL.md)
(section "Long-Range and Dispersion") for the full reference.

### Electrostatics (learned reciprocal-space correction)

A reciprocal-space correction driven by learned multipole sources predicted from
the final invariant descriptor, with periodic or slab boundaries and either a
direct k-space sum or an FFT mesh backend:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 64 --lmax 2 --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-boundary periodic \
  --long-range-reciprocal-backend direct_kspace \
  --long-range-kmax 4 \
  --long-range-source-channels 1 \
  --long-range-max-multipole-l 0 \
  --checkpoint model_elec.pth
```

Use `--long-range-boundary slab` for 2D-periodic interfaces, and
`--long-range-reciprocal-backend mesh_fft --long-range-mesh-size 32` for the FFT
mesh on larger cells. `--long-range-max-multipole-l` raises the source order
(e.g. `1` adds dipoles); `--long-range-source-channels` sets the number of
learned scalar source channels.

### Dispersion (anisotropic many-body dispersion)

A many-body dispersion term evaluated by matrix-free stochastic Lanczos
quadrature (no explicit eigendecomposition). The atomic polarizability is either
an isotropic scalar or an **anisotropic 3x3 tensor** built from the ICTC `l=2`
features, which keeps the dispersion energy rotationally equivariant and uses the
ICTC representation directly:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 64 --lmax 2 --num-interaction 2 \
  --long-range-dispersion-mode mbd-slq \
  --dispersion-cutoff 9.0 \
  --mbd-operator-backend edge_sparse \
  --dispersion-slq-num-probes 4 \
  --mbd-anisotropic \
  --checkpoint model_mbd.pth
```

Drop `--mbd-anisotropic` for the isotropic scalar polarizability;
`--mbd-operator-backend` selects the direct cutoff sum (`edge_sparse`, default)
or a reciprocal FFT dipole-field backend (`pme_fft`). The electrostatic and
dispersion channels can be enabled together.

### Deployment

For deployment the model emits a compact per-atom source tensor (for anisotropic
MBD a `[N, 8]` tensor: an effective frequency, the isotropic polarizability, and
the six unique components of the tensor polarizability), and the C++
`USER-MFFTORCH` solver reconstructs the coupling and evaluates the long-range
energy and forces at runtime. The reciprocal electrostatic correction and the
`edge_sparse` MBD path are `make_fx`/Inductor-compilable for training; the SLQ
dispersion runs eager.

Native MACE conversion preserves the source model; it does not add long-range
terms to an already trained MACE checkpoint.

## ASE, AOTI, and LAMMPS

ASE:

```python
from mace_ictc.evaluation.calculator import MyE3NNCalculator

atoms.calc = MyE3NNCalculator(checkpoint="model.pth", device="cuda")
```

AOTInductor:

```bash
python -m mace_ictc.cli.export_aoti_core \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model.pt2 \
  --dynamic
```

LAMMPS support is in [lammps_user_mfftorch](lammps_user_mfftorch). After building
LAMMPS with the provided `USER-MFFTORCH` package and LibTorch/AOTI support:

```text
pair_style   mff/torch  5.0 cuda
pair_coeff   * * /path/to/model.pt2 H C N O
```

Read [lammps_user_mfftorch/README.md](lammps_user_mfftorch/README.md) and
[lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)
for build details.

## Benchmarks and Reproducibility

The curated benchmark archive lives in [benchmarks/paper](benchmarks/paper). It
contains lightweight scripts, CSV/JSON summaries, validation logs, and SVG
figures used by the technical report. Large generated binaries such as
checkpoint snapshots, AOTI packages, PNG/PDF figure exports, and MD trajectory
arrays are kept out of Git and should be distributed through GitHub Releases.

Representative RTX 4090 results are shown below. They are meant to summarize
the observed regimes, not replace the full artifact tables.

- Isolated tensor product, FP32, `C=64`, `E=100k`, forward+backward:
  compiled ICTC is `45.5 ms` at `(L_h,L_e)=(2,2)` versus e3nn `50.8 ms`
  and cartnn `62.5 ms`; at `(3,3)`, compiled ICTC is `128.5 ms` versus
  e3nn `174.7 ms`, while cartnn OOMs.
- Whole-model synthetic training, `lmax=max_ell=1`, 8192 atoms, avg degree 16:
  `ICTC+cuEq compiled` reaches `46.3 ms/step`, about `1.9x` faster than
  MACE e3nn (`89.0 ms`) and slower than native MACE cuEq (`36.5 ms`).
- Whole-model synthetic inference, same setting: `ICTC+cuEq compiled` reaches
  `15.3 ms/step`, about `1.5x` faster than MACE e3nn (`22.6 ms`) and slightly
  faster than native MACE cuEq (`16.9 ms`).
- Matched 300-epoch training runs on revised benzene/ethanol/aspirin and Cheng
  water show lower final force RMSE for ICTC modes than for the matched MACE
  e3nn/cuEq baselines under the archived protocol. This is a controlled
  protocol comparison, not a universal accuracy claim.

Replot the archived figures from the repository root:

```bash
python benchmarks/paper/scripts/plot_benchmark_figures.py
python benchmarks/paper/scripts/training/plot_paper_training_figures.py
```

Selected RTX 4090 throughput figures:

![Backend throughput benchmark](docs/figures/backend_throughput_benchmark_channels64.png)

![Backend speedup benchmark](docs/figures/backend_speedup_benchmark_channels64.png)

These benchmarks are controlled computational records, not universal
chemical-accuracy leaderboards. The synthetic whole-model runs fix graph
workloads to expose backend throughput; the matched training runs fix small
protocols to compare parameterization/backend behavior under the same data and
optimizer settings.

## Validation Scope

The repository contains tests and benchmark records for:

- ICTC/e3nn frame correspondence;
- converted native-MACE energy and force agreement for supported models;
- rotation invariance/equivariance checks;
- fused contraction agreement against reference contraction paths;
- representation-layer SO(2) and spin-coupled double-cover prototype checks;
- long-MD checkpoint-correspondence diagnostics for selected models.

The strongest exactness statement is for the supported MACE correspondence path:
compatible native MACE checkpoints can be represented in MACE-ICTC with the same
energy/force behavior up to expected floating-point tolerances. This should not
be read as a guarantee for arbitrary MACE forks, unsupported model structures,
or every possible ICTC-native experimental backend.

## Repository Layout

```text
MACE-ICTC/
  mace_ictc/
    cli/                 # training, conversion, export, LAMMPS helpers
    data/                # preprocessing, H5 datasets, batching
    evaluation/          # ASE calculator wrappers
    interfaces/          # checkpoint and deployment wrappers
    models/              # ICTC model, irreps, products, radial, long-range
    training/            # ForceTrainer and make_fx compile helpers
    utils/               # graph, scatter, config, checkpoint metadata
  lammps_user_mfftorch/  # LAMMPS USER-MFFTORCH package
  benchmarks/paper/      # curated paper benchmark records and scripts
  docs/                  # user manuals and figures
```

## Citation

If you use MACE-ICTC, cite the Zenodo DOI:

```text
MACE-ICTC DOI: 10.5281/zenodo.20690950
```
