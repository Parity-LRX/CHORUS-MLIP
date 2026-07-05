# MACE-ICTC User Manual

This manual describes the standalone MACE-ICTC repository: what each subsystem is for, which runtime mode to choose, and how to train, convert, export, benchmark, and deploy models.

Chinese version: [USER_MANUAL.zh-CN.md](USER_MANUAL.zh-CN.md)

## 1. What This Repository Contains

MACE-ICTC is a standalone implementation of MACE in the Irreducible Cartesian Tensor Decomposition (ICTC) basis. It keeps the MACE model class and deployment stack independent of the original FSCETP tree.

The central technical point is the ICTC basis construction: fixed `Q`/`U` operators re-express MACE/e3nn angular algebra in irreducible Cartesian tensor blocks while preserving the original MACE interaction/readout semantics. The repository focuses on basis conversion, native `mace-torch` parity, and deployment.

The repository includes:

- `mace_ictc/models/`: the `PureCartesianICTDFix` model, ICTC irreps, tensor product helpers, MACE-compatible symmetric contractions, radial basis functions, optional ZBL and long-range modules.
- `mace_ictc/training/`: the energy/force/stress trainer and the `make_fx` + Inductor force-step compiler.
- `mace_ictc/data/`: extended-XYZ parsing, H5 dataset loading, graph padding, bucket sampling, and collate functions.
- `mace_ictc/cli/`: command-line tools for training, MACE conversion, AOTInductor export, TorchScript export, and LAMMPS helper generation.
- `mace_ictc/interfaces/`: checkpoint loading, LAMMPS MLIAP wrappers, and deployment-facing compatibility code.
- `mace_ictc/evaluation/`: ASE calculator wrappers.
- `mace_ictc/bench/`: benchmark harnesses comparing MACE-ICTC modes against native `mace-torch`.
- `mace_ictc/test/`: numerical and smoke tests.
- `lammps_user_mfftorch/`: a LAMMPS `USER-MFFTORCH` package with C++ pair styles and LibTorch/AOTI integration.

The core model forward signature is:

```python
model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
```

where:

- `pos`: Cartesian coordinates, shape `[N, 3]`.
- `A`: atomic numbers, not species indices.
- `batch`: graph id for each atom, shape `[N]`.
- `edge_src`, `edge_dst`: directed edge indices.
- `edge_shifts`: integer periodic image shifts, shape `[E, 3]`.
- `cell`: cell tensor, shape `[B, 3, 3]`.
- return value: per-atom interaction energies, normally shape `[N, 1]`.

Atomic reference energies E0 are handled outside the core model in training/export wrappers.

## 2. Installation

Minimal editable install:

```bash
cd /path/to/MACE-ICTC
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pyg]"   # torch-scatter and torch-cluster acceleration
pip install -e ".[cue]"   # cuEquivariance product backend
pip install -e ".[e0]"    # pandas support for fitted E0 CSV files
pip install -e ".[full]"  # all optional dependencies
```

Important runtime expectations:

- Python >= 3.9.
- PyTorch >= 2.4; PyTorch >= 2.7 is recommended for `make_fx` and AOTInductor workflows.
- `e3nn < 0.6` for compatibility with current `mace-torch`.
- CUDA is required for the cuEquivariance and serious AOTI/Inductor benchmark paths.

Optional compiled ICTC tensor-product extension:

```bash
MFF_BUILD_ICTD_TP_EXT=1 pip install -e .
```

For CUDA extension build:

```bash
MFF_BUILD_ICTD_TP_EXT=1 MFF_BUILD_ICTD_TP_CUDA=1 pip install -e .
```

The Python fallback path works without this extension.

## 3. Command-Line Tools

Installed console scripts:

| Command | Python entry point | Purpose |
|---|---|---|
| `mff-convert-mace` | `mace_ictc.cli.convert_mace` | Convert a native `mace-torch` `ScaleShiftMACE` checkpoint to MACE-ICTC. |
| `mff-export-aoti` | `mace_ictc.cli.export_aoti_core` | Export an AOTInductor `.pt2` core for Python/C++/LAMMPS deployment. |
| `mff-export-core` | `mace_ictc.cli.export_libtorch_core` | Export a TorchScript core. Mostly for legacy LibTorch deployment. |
| `mff-lammps` | `mace_ictc.cli.lammps_interface` | Generate helper files for LAMMPS-style deployment. |

Direct module commands are also supported:

```bash
python -m mace_ictc.cli.train --help
python -m mace_ictc.cli.convert_mace --help
python -m mace_ictc.cli.export_aoti_core --help
```

## 4. Core Concepts

### 4.1 ICTC Basis Versus e3nn/MACE Basis

Original MACE uses e3nn spherical features. MACE-ICTC stores equivariant features in an ICTC Cartesian basis. The two bases are related by fixed orthogonal per-`l` matrices `Q`.

For invariant outputs such as total energy, forces, and virial, the basis choice should not change the physical result. For equivariant intermediate features, use:

```python
model.to_mace_basis(x)
model.to_ictd_basis(x)
```

or the lower-level helpers in `mace_ictc.mace_basis`.

### 4.2 `angular_basis`

`angular_basis` controls which basis the model computes in internally:

| Value | Meaning | When to use |
|---|---|---|
| `ictd` | Default ICTC internal basis. | Canonical parity mode, bridge-U mode, safest baseline. |
| `e3nn` | Fold fixed angular operators once so the internal equivariant features are in the original MACE/e3nn convention. | cuEq product performance path; AOTI export with `--cueq-product`. |

Important constraints:

- `ictd-bridge-u` does not expose an e3nn-fold path. If you request `angular_basis=e3nn` without replacing the product backend, export will keep `ictd` and print a warning.
- `cueq` product supports `angular_basis=e3nn`.
- Training with `angular_basis=e3nn` saves already-folded fixed buffers. Checkpoint reload restores the runtime Q blocks and product flags without folding a second time.

### 4.3 Product Backends

| Backend | Description | Recommended use |
|---|---|---|
| `ictd-bridge-u` | Uses MACE/e3nn symmetric-contraction U tensors with the ICTC/e3nn basis bridge folded into the U tensors. | Canonical MACE parity and high-`max_ell` conversion. |
| `cueq` | Uses cuEquivariance for the product/symmetric contraction. | Performance training and inference, especially with `--angular-basis e3nn`. |
| `native-mace` | Calls MACE's native symmetric contraction in the product block. | Debug/reference path. |
| `ictd-pure-u` | Uses ICTC-generated U tensors directly. | Diagnostic path; not the primary high-`max_ell` production path. |

### 4.4 `use_reduced_cg`

`--use-reduced-cg` is a structural option for the product/symmetric contraction. It changes the CG/path layout and weight shapes.

Rules:

- When converting an existing native MACE checkpoint, follow the original `mace_model.use_reduced_cg`. Do not choose it manually.
- Native `mace-torch` training also has this option, named `--use_reduced_cg`.
- For from-scratch MACE-ICTC training, only enable it if you intentionally want the reduced-CG architecture.
- It is not a guaranteed stable-step throughput accelerator. In recent 4090 tests for `cueq + angular_basis=e3nn + make_fx`, stable step-time changed only by roughly -1% to +2%, while compile time improved more.

## 5. Which Mode Should I Use?

| Goal | Recommended mode |
|---|---|
| Exact MACE conversion/parity baseline | `ictd-bridge-u`, `angular_basis=ictd`, usually `dtype=float64`. |
| From-scratch training that should match MACE architecture semantics | `ictd-bridge-u`, `function-type=bessel`, MACE-style ScaleShift enabled. |
| Fast training | `cueq`, `angular_basis=e3nn`, `--train-makefx-compile`, bucketed shapes. |
| Fast AOTI inference from an ICTC checkpoint | `mff-export-aoti --cueq-product --angular-basis e3nn` when cuEq custom ops are deployable. |
| Most conservative deployment | Export the checkpoint without cuEq replacement, keep `angular_basis=checkpoint` or `ictd`. |

Complete canonical parity training command:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend ictd-bridge-u \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --max-steps 200000 \
  --batch-size 4 \
  --loss smooth_l1 \
  --loss-beta 0.5 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 0.001 \
  --min-lr 0.000001 \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr-scheduler plateau \
  --lr-factor 0.8 \
  --scheduler-patience 50 \
  --warmup-batches 1000 \
  --warmup-start-ratio 0.1 \
  --swa \
  --start-swa 225 \
  --swa-lr 0.0001 \
  --swa-energy-weight 1000.0 \
  --swa-force-weight 100.0 \
  --swa-stress-weight 0.0 \
  --ema-decay 0.0 \
  --checkpoint-state-source swa \
  --device cuda \
  --dtype float64 \
  --checkpoint model_bridge_u.pth
```

Complete high-performance training command:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --makefx-max-slots 8 \
  --pad-nodes-to-max \
  --pad-edges-to-max \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --max-steps 200000 \
  --batch-size 8 \
  --loss smooth_l1 \
  --loss-beta 0.5 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 0.001 \
  --min-lr 0.000001 \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr-scheduler cosine \
  --warmup-batches 1000 \
  --warmup-start-ratio 0.1 \
  --swa \
  --start-swa 225 \
  --swa-lr 0.0001 \
  --swa-energy-weight 1000.0 \
  --swa-force-weight 100.0 \
  --swa-stress-weight 0.0 \
  --ema-decay 0.999 \
  --ema-start-step 1000 \
  --checkpoint-state-source swa \
  --device cuda \
  --dtype float32 \
  --checkpoint model_cueq_e3nn_makefx.pth
```

The numeric values above are safe starting points, not chemistry-independent optimum
hyperparameters. For strict comparison against a native MACE run, match the dataset
split, seed, loss weights, optimizer, scheduler, batch construction, dtype, and
ScaleShift/E0 settings.

### 5.1 Multi-GPU Training

MACE-ICTC training supports PyTorch `DistributedDataParallel` through the training
CLI. The default `--ddp auto` enables DDP when the process environment has
`WORLD_SIZE>1`, so a normal `torchrun` launch is enough:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m mace_ictc.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --pad-nodes-to-max \
  --pad-edges-to-max \
  --batch-size 4 \
  --device cuda \
  --ddp auto \
  --checkpoint model_ddp.pth
```

On Slurm, request the GPUs through the scheduler and let `srun` or `torchrun`
create one process per GPU. A minimal pattern is:

```bash
#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

source /path/to/conda.sh
conda activate mff

srun python -m mace_ictc.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --pad-nodes-to-max \
  --pad-edges-to-max \
  --batch-size 4 \
  --device cuda \
  --ddp auto \
  --checkpoint model_ddp.pth
```

DDP behavior:

- Each process uses `cuda:$LOCAL_RANK`; do not manually give every rank the same
  `--device cuda:0`.
- `--batch-size` is per rank. The effective global batch is approximately
  `batch_size * world_size`.
- Rank 0 performs validation logging and checkpoint writing; checkpoint keys are
  saved without a `module.` prefix.
- With `--makefx-buckets`, the bucket sampler is DDP-aware. Keep padding/bucketing
  enabled when using `--train-makefx-compile`; otherwise every raw graph shape can
  trigger a separate compile.
- `--train-makefx-compile` also works under DDP, but the first compiled bucket has
  a noticeable compile cost on every rank. It is useful for long runs and stable
  bucketed shapes, not for very short smoke tests.
- Use `--ddp on` when you want the command to fail unless DDP is active. Use
  `--ddp off` for single-process debugging.

## 6. Data Pipeline

The trainer consumes preprocessed H5 files:

```text
DATA/
  processed_train.h5
  processed_val.h5       # optional
  processed_train.counts.npz / bucket sidecars, when generated
```

The parser in `mace_ictc.data.preprocessing` supports extended XYZ-style data with:

- atomic species / atomic numbers,
- Cartesian positions,
- forces,
- total energy,
- cell and PBC flags,
- optional stress or virial.

Programmatic preprocessing entry point:

```python
from mace_ictc.data.preprocessing import save_to_h5_parallel

save_to_h5_parallel(
    prefix="train",
    max_radius=5.0,
    num_workers=8,
    data_dir="DATA",
)
```

The dataset loader is `mace_ictc.data.datasets.H5Dataset`; batching uses `mace_ictc.data.collate.collate_fn_h5`.

For `make_fx` training, prefer size bucketing:

```bash
--train-makefx-compile --makefx-buckets 6
```

This groups similar atom/edge counts so Inductor compiles once per bucket instead of once per raw shape.

## 7. Training Details

Training CLI:

```bash
python -m mace_ictc.cli.train --help
```

Key architecture arguments:

| Argument | Meaning |
|---|---|
| `--channels` | Hidden channel count. |
| `--lmax` | Hidden feature maximum angular order. |
| `--max-ell` | Edge spherical-harmonics cutoff. Defaults to `--lmax`. |
| `--num-interaction` | Number of MACE interaction/product blocks. |
| `--correlation` | MACE product correlation order, also called `save_contraction_order`. |
| `--function-type` | Radial basis type; use `bessel` for MACE-like parity. |
| `--product-backend` | Product backend: usually `ictd-bridge-u` or `cueq`. |
| `--angular-basis` | `ictd` or `e3nn`; use `e3nn` with `cueq` for the performance path. |
| `--use-reduced-cg` | Enable reduced-CG product layout. Must match source MACE when converting. |
| `--long-range-mode` | Optional learned scalar reciprocal correction. Use `reciprocal-spectral-v1` to enable; default is `none`. |

Optimization, step, and seed controls:

| Argument | Meaning |
|---|---|
| `--seed` | Seeds Python, NumPy, PyTorch, DataLoader shuffle, and bucket sampling. It improves reproducibility but does not force deterministic CUDA kernels. |
| `--epochs` | Maximum epoch count. |
| `--max-steps` | Optional optimizer-step cap. If set, training stops once this global step is reached, even mid-epoch. |
| `--batch-size` | Graphs per batch or per bucketed batch. |
| `--optimizer` | `adamw` or `adam`. |
| `--lr`, `--min-lr`, `--weight-decay` | Initial/max learning rate, scheduler floor, and AdamW weight decay. LR is clamped to `[--min-lr, --lr]`. |
| `--adam-beta1`, `--adam-beta2`, `--adam-eps`, `--amsgrad` | Adam/AdamW numerical parameters. |
| `--lr-scheduler` | `plateau`, `exp`, `cosine`, `step`, or `none`. `ReduceLROnPlateau` and `ExponentialLR` aliases are accepted. |
| `--warmup-batches`, `--warmup-start-ratio` | Linear warmup length and initial LR multiplier. |
| `--lr-factor`, `--scheduler-patience` | Plateau factor and patience. |
| `--lr-scheduler-gamma` | Exponential scheduler gamma. |
| `--lr-decay-step`, `--lr-decay-factor` | Legacy step scheduler parameters. |
| `--max-grad-norm` | Optional gradient clipping threshold. |

Energy/force/stress loss:

```text
total = energy_weight * loss(E)
      + force_weight  * loss(F)
      + stress_weight * loss(stress)
```

| Argument | Meaning |
|---|---|
| `--loss` | `smooth_l1` (default) or `mse`. |
| `--loss-beta` | SmoothL1 beta for energy, force, and stress when `--loss smooth_l1`. |
| `--energy-weight`, `--force-weight`, `--stress-weight` | Weights in the total loss. Stress is disabled by default (`--stress-weight 0`). |
| `--force-shift-value` | Multiplies the reference force before the force loss. Keep at `1.0` unless reproducing a legacy run. |

When stress is enabled, stress is computed from the strain derivative.

MACE-style Stage Two / SWA:

| Argument | Meaning |
|---|---|
| `--swa`, `--stage-two` | Enables mace-torch-style Stage Two: switch loss weights, lower LR, and save averaged weights. |
| `--start-swa`, `--start-stage-two`, `--swa-start-epoch` | First epoch using Stage Two. If `--swa` is set and no start is given, defaults to `max(1, epochs * 3 // 4)`, matching mace-torch behavior. |
| `--swa-start-step` | Optional global-step trigger for Stage Two. |
| `--swa-lr`, `--stage-two-lr` | Stage Two/SWA LR. It must satisfy `--min-lr <= --swa-lr <= --lr`. |
| `--swa-energy-weight`, `--swa-force-weight`, `--swa-stress-weight` | Stage Two loss weights. MACE-like energy/force defaults are `1000` and `100`; stress defaults to `0` when stress is off and `10` when stress is on. |
| `--swa-anneal-epochs`, `--swa-anneal-strategy` | SWALR annealing controls. |
| `--ema-decay` | Enables exponential moving average when greater than `0`, for example `0.999`. Saved as `e3trans_ema_state_dict`. |
| `--ema-start-step` | First global optimizer step eligible for EMA updates. |
| `--checkpoint-state-source` | `auto`, `raw`, `ema`, or `swa`. Deploy loaders use `default_state_source`; `auto` prefers EMA, then SWA, then raw. |

This now mirrors the relevant mace-torch Stage Two behavior for this trainer:
loss weights switch at the Stage Two boundary, the main LR scheduler is suspended,
the optimizer LR is moved to `--swa-lr`, and averaged weights are saved as
`e3trans_swa_state_dict`.

ScaleShift behavior:

- Default: `--scaling rms_forces_scaling`.
- Also available: `std_scaling`, `no_scaling`.
- Override scale/shift explicitly with `--atomic-inter-scale` and `--atomic-inter-shift`.
- Use `--no-atomic-inter-shift` to keep scaling but force zero interaction-energy shift.

E0 behavior:

- Pass `--atomic-energy-keys` and `--atomic-energy-values` for explicit reference energies.
- If omitted, the training CLI uses its built-in H/C/N/O defaults.
- Export can embed E0 into the deployed core with `--embed-e0`.

Long-range correction:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 64 --lmax 2 --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-boundary periodic \
  --long-range-reciprocal-backend direct_kspace \
  --long-range-kmax 4 \
  --long-range-source-channels 1 \
  --checkpoint model_lr.pth
```

This enables the currently supported long-range module, `reciprocal-spectral-v1`. It maps the final
per-atom invariant descriptor to learned latent scalar sources and adds a reciprocal-space energy
term. The contribution is initialized near zero, so the model starts close to the short-range baseline
and learns the correction from the same energy/force/stress losses. It is not a fixed-charge analytic
Ewald term and it does not require explicit charge labels.

Key options:

| Argument | Meaning |
|---|---|
| `--long-range-boundary periodic` | Fully periodic reciprocal solve. Required by `direct_kspace`. |
| `--long-range-boundary slab` | Slab boundary; use with `--long-range-reciprocal-backend mesh_fft`. |
| `--long-range-reciprocal-backend direct_kspace` | Direct k-space sum inside the model/exported core. Best for small `kmax` tests. |
| `--long-range-reciprocal-backend mesh_fft` | FFT mesh path for larger periodic/slab systems. |
| `--long-range-kmax` | Integer k-lattice cutoff for `direct_kspace`. |
| `--long-range-mesh-size` | Mesh resolution for `mesh_fft`. |
| `--long-range-source-channels` | Number of latent scalar source channels. |
| `--no-long-range-neutralize` | Disable per-graph source neutralization. Usually leave neutralization on. |
| `--long-range-include-k0` | Include the k=0 mode. Usually leave off with neutralized sources. |
| `--long-range-green-mode` | `poisson` or `learned_poisson`. |

The checkpoint stores the long-range hyperparameters in `model_hyperparameters`, so
`LAMMPS_MLIAP_MFF.from_checkpoint` and `mff-export-aoti --checkpoint model_lr.pth ...` rebuild the
same architecture. Native MACE conversion does not add a long-range module to an already trained
MACE checkpoint; train or fine-tune in MACE-ICTC with `--long-range-mode reciprocal-spectral-v1` when
this correction is needed.

## 8. Native MACE Conversion

Use this path after training a model with native `mace-torch`. The input must be a torch-saved
`ScaleShiftMACE` object, not only a raw `state_dict`:

```bash
mff-convert-mace \
  --mace-model mace.model \
  --out mace_ictc.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

Recommended parity path:

- Convert with `--product-backend ictd-bridge-u`.
- Use `--dtype float64` for the tightest energy/force comparison.
- Export the converted checkpoint without changing the product backend.
- Use `--embed-e0` when the deployed `.pt2` should return absolute energies.

```bash
mff-export-aoti \
  --checkpoint mace_ictc.pth \
  --elements H,C,N,O \
  --out mace_ictc.pt2 \
  --dynamic \
  --embed-e0
```

For faster cuEq product inference from the same converted checkpoint:

```bash
mff-export-aoti \
  --checkpoint mace_ictc.pth \
  --elements H,C,N,O \
  --out mace_ictc_cueq_e3nn.pt2 \
  --dynamic \
  --cueq-product \
  --angular-basis e3nn
```

The cuEq export path replaces the product blocks and folds compatible angular operators to the e3nn
convention. It requires cuEquivariance custom op registration in the deployment runtime. Keep E0
outside this core if your exporter/runtime cannot combine `--embed-e0` with `--cueq-product`.

Supported versions and model variants:

- The converter targets the `mace-torch` `ScaleShiftMACE` object layout used by this repository's
  tests and benchmarks, validated with `mace==0.3.16` and `e3nn<0.6`.
- Newer `mace-torch` releases may work if the saved object layout and `extract_config_mace_model`
  output remain compatible, but they are not automatically covered.
- Checkpoint loading still depends on Python pickle compatibility. Very old pretrained MACE models
  may need the matching historical `mace-torch`/`e3nn` environment to load; after a model object is
  loadable, conversion is governed by the structural checks below.
- This is not a converter for arbitrary native-MACE-like implementations, raw state dicts, or custom
  research forks unless they save a compatible `ScaleShiftMACE`.

Conversion constraints are intentionally strict. Unsupported variants are rejected rather than silently converted. Current supported assumptions include:

- `ScaleShiftMACE`,
- Bessel radial basis,
- `radial_MLP=[64,64,64]`,
- MACE parity hidden irreps with contiguous `l=0..L` and uniform channel count,
- uniform correlation across layers,
- no pair repulsion or distance transform,
- SiLU MACE-style scalar readout with `MLP_irreps=16x0e`,
- first interaction `RealAgnosticInteractionBlock` or `RealAgnosticResidualInteractionBlock`,
- later interactions `RealAgnosticResidualInteractionBlock`,
- `max_ell >= hidden_irreps.lmax`,
- `num_interactions >= 2`.

Backend differences:

- `ictd-bridge-u`: recommended conversion backend; folds the MACE/e3nn U convention through the ICTC
  basis bridge and is the main parity path.
- `native-mace`: debug/reference backend; useful for diagnosing MACE-side contraction behavior.
- `cueq`: performance-oriented product backend; use especially at export time with
  `--cueq-product --angular-basis e3nn`.
- `ictd-pure-u`: diagnostic ICTC-generated-U path; it is not the native MACE exact-conversion path.

The converter reads `mace_model.use_reduced_cg` and rebuilds MACE-ICTC with the same reduced-CG setting.
Users should not guess or override that flag for imported native MACE models.
Conversion preserves the source MACE architecture; it does not add a learned long-range module.

### 8.1 OFF23 pretrained conversion and `mff/torch` smoke test

Public pretrained MACE checkpoints are often saved as pickled Python model objects. The converter can
only start after Python can load the source object. For older OFF23 checkpoints, this may require a
historical `mace-torch`/`e3nn` environment for the loading and conversion step; the converted ICTC
checkpoint can then be loaded by the current MACE-ICTC runtime.

**Setting up that historical loading environment.** Current `e3nn` (0.5.x / 0.6.x) cannot deserialize
OFF23 checkpoints: `torch.load` raises `ValueError: too many values to unpack (expected 2)` from
`e3nn/util/codegen/_mixin.py` (the compiled-module pickle format changed between e3nn versions). Load
and convert with the versions OFF23 was serialized with — **`e3nn==0.4.4` + `mace-torch==0.3.16`** —
installed into isolated directories and prepended on `PYTHONPATH` so they shadow the environment's
newer `e3nn`/`mace`, while the environment still supplies `torch` (any recent version works for the
deserialization):

```bash
# one-time: stage the legacy deserialization deps anywhere persistent
pip install --target=$HOME/compat_e3nn044/e3nn_0_4_4        "e3nn==0.4.4"
pip install --target=$HOME/compat_e3nn044/mace_torch_0_3_16 "mace-torch==0.3.16"

# load + convert with them prepended (mace_ictc also importable via PYTHONPATH or install):
PYTHONPATH=$HOME/compat_e3nn044/mace_torch_0_3_16:$HOME/compat_e3nn044/e3nn_0_4_4 \
  python -m mace_ictc.cli.convert_mace \
    --mace-model /path/to/MACE-OFF23_small.model \
    --out MACE-OFF23_small_ictd_bridge_u_f64.pth \
    --product-backend ictd-bridge-u --dtype float64
```

The converted `.pth` is a plain state_dict and loads in the normal (current-`e3nn`) environment for
training and export — only the *loading of the pickled source model* needs the legacy deps. Note: the
fresh-build converter parity test (`mace_ictc/test/test_mace_converter.py`) builds its MACE in-process
and is unaffected; this legacy environment is needed only to load *saved* foundation checkpoints.

Example conversion of an OFF23 small model:

```bash
mff-convert-mace \
  --mace-model /path/to/MACE-OFF23_small.model \
  --out MACE-OFF23_small_ictd_bridge_u_f64.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

Float64 is the recommended audit format for native-MACE parity. For LAMMPS deployment, export a
float32 AOTInductor core with a static atom count when the target MD cell has fixed `N`. Create the
deployment checkpoint with the same conversion command and `--dtype float32`, then export:

```bash
mff-export-aoti \
  --checkpoint MACE-OFF23_small_ictd_bridge_u_f32.pth \
  --elements H,C,N,O,F,P,S,Cl,Br,I \
  --atoms 6 \
  --degree 5 \
  --static-n \
  --dtype float32 \
  --device cuda \
  --embed-e0 \
  --out MACE-OFF23_small_ictd_bridge_u_f32_static6.pt2
```

Minimal LAMMPS input:

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch 4.5 cuda
pair_coeff * * MACE-OFF23_small_ictd_bridge_u_f32_static6.pt2 H C N O

thermo 1
thermo_style custom step temp pe etotal fmax
run 0
```

On the 4090 validation host, both `build-mfftorch` and `build-mfftorch-kk` compiled and loaded the
`.pt2` OFF23 core. The ordinary build and the Kokkos build produced the same LAMMPS energies to the
printed precision. For a fresh static-6 export, the LAMMPS `run 0` result was:

| Quantity | LAMMPS `mff/torch` | Python checkpoint |
|---|---:|---:|
| energy (eV) | `-6633.036` | `-6633.03613281` |
| max absolute force component (eV/A) | `11.767612` | `11.76760674` |

LAMMPS `fmax` in this thermo output is the maximum absolute force component, not the maximum force
vector norm. Compare it against `max(abs(forces))`, not against `max(norm(forces_i))`.

The same converted float64 checkpoint was compared against native `mace-torch` on a benzene
same-frame trajectory. The maximum absolute energy difference was `2.73e-12 eV`; the maximum force
component difference was `4.44e-15 eV/A`. This checks the conversion bridge. The AOTI export should
still be checked separately because compiler lowering changes floating-point operation order.

## 9. AOTInductor Export

Basic export:

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model.pt2 \
  --dynamic \
  --embed-e0
```

Performance-oriented export:

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model_cueq_e3nn.pt2 \
  --dynamic \
  --cueq-product \
  --angular-basis e3nn \
  --assume-cutoff-edges \
  --preserve-edge-order \
  --fuse-selector-message-linear \
  --inductor-max-autotune
```

For the performance-oriented cuEq export, keep atomic E0 outside the exported core unless your current
exporter/runtime supports combining `--embed-e0` with product replacement. The conservative bridge-U
export above is the simplest path when an E0-embedded absolute-energy `.pt2` is required.

Important export options:

| Option | Meaning |
|---|---|
| `--dynamic` | Export with dynamic atom/edge dimensions where supported. |
| `--static-n` | Keep atom count static. Useful for fixed-N MD when dynamic export is problematic. |
| `--embed-e0` | Add atomic reference energies into the exported energy. |
| `--cueq-product` | Replace product blocks with cuEq product blocks during export. |
| `--angular-basis e3nn` | Fold fixed angular operators to e3nn basis for fold-capable products. |
| `--assume-cutoff-edges` | Assume caller already filtered edges inside cutoff; skips model-side edge mask. |
| `--preserve-edge-order` | Assume caller passes stable edge order; skips model-side destination sort. |
| `--fuse-selector-message-linear` | Fuse selected message linears where supported. |
| `--inductor-max-autotune` | Slower compile, potentially faster kernel choices. Benchmark before relying on it. |

`strict=False` export fallback:

- The exporter first tries strict export when appropriate.
- If strict export fails due to exporter limitations, it can retry non-strict export.
- Correctness is still checked by compiling/loading the `.pt2` and comparing numerical outputs.

## 10. ASE and Python Inference

The ASE wrapper is `mace_ictc.evaluation.calculator.MyE3NNCalculator`.

Typical use:

```python
import torch
from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from mace_ictc.evaluation.calculator import MyE3NNCalculator

wrapper = LAMMPS_MLIAP_MFF.from_checkpoint(
    "model.pth",
    element_types=["H", "C", "N", "O"],
    device="cuda",
)

atoms.calc = MyE3NNCalculator(
    model=wrapper.wrapper.model,
    atomic_energies_dict={1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
    device=torch.device("cuda"),
    max_radius=5.0,
)
```

For production MD, prefer exported AOTI/LAMMPS paths after numerical validation.

## 11. LAMMPS Deployment

LAMMPS support lives in:

```text
lammps_user_mfftorch/
```

Read:

- `lammps_user_mfftorch/README.md`
- `lammps_user_mfftorch/docs/BUILD_AND_RUN.md`

The package provides:

- `pair_style mff/torch`
- `pair_style mff/torch/kk`

Current `.pt2` LAMMPS deployment supports energy and forces. Physical tensor
outputs are not exposed as a supported public LAMMPS interface.

General workflow:

1. Train or convert a checkpoint.
2. Export an AOTI `.pt2` or TorchScript core depending on the target LAMMPS integration.
3. Build LAMMPS with `USER-MFFTORCH` and LibTorch.
4. Use the exported model in a LAMMPS input script.

Minimal example:

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O

fix 1 all nve
run 100
```

The element order in `pair_coeff` must match the export/load order.

### 11.1 Multi-GPU LAMMPS Runs

`USER-MFFTORCH` is an MPI pair style. Multi-GPU use is therefore normally
one MPI rank per GPU, not one LAMMPS process controlling all GPUs. Build LAMMPS
with MPI, `USER-MFFTORCH`, LibTorch, and optionally Kokkos/CUDA.

For the non-Kokkos pair style:

```bash
export MFF_DEBUG_BUNDLE=1   # optional: prints requested and selected devices
mpirun -np 2 /path/to/lmp -in in.mfftorch
```

with an input using:

```lammps
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O
```

For the Kokkos GPU data path:

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lmp -k on g 2 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
```

and either write the Kokkos style explicitly:

```lammps
pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O
```

or use `pair_style mff/torch` and let `-sf kk` map it to the Kokkos variant when
the build supports that mapping.

Device mapping details:

- Plain `mff/torch` selects the local CUDA device from MPI/Slurm local-rank
  variables such as `SLURM_LOCALID`, `LOCAL_RANK`, `OMPI_COMM_WORLD_LOCAL_RANK`,
  or `MPI_LOCALRANKID`.
- `mff/torch/kk` currently maps ranks to Kokkos GPUs correctly for a single-node
  run with one MPI rank per GPU. Treat multi-node Kokkos runs as requiring an
  explicit local-rank validation before production.
- With `MFF_DEBUG_BUNDLE=1`, the engine prints the requested device and the
  selected device. Check that different local ranks select different GPUs.
- For static-N `.pt2` exports, make sure the exported `--atoms`/`--degree`
  capacity covers local atoms plus ghosts on every MPI rank. N-dynamic `.pt2`
  exports are more convenient when the deployment environment supports them.
- Compare `run 0` energy and forces between `-np 1` and `-np N` before long MD.
  Small fp32 differences can occur from different edge ordering and accumulation,
  but large differences indicate a decomposition, cutoff, or exported-capacity
  problem.

## 12. Long-Range and Dispersion (Train → Export → Deploy)

MACE-ICTC learns two families of long-range correction, both in-network (no external Ewald or
libMBD) and both LAMMPS-deployable through the `mff/torch` pair style. This section is the
end-to-end reference: what is available, how to train it, how to export it, and how to run it in
LAMMPS.

The unifying mechanism: the network emits **per-atom latent sources** from its descriptors. At
training time the long-range energy is computed inline and learned from the same
energy/force/stress losses (no charge labels). At deploy time the source is exported and the
long-range energy is either kept in the compiled graph (pairwise-C6 dispersion) or **deferred to a
dedicated C++ solver** in LAMMPS (reciprocal electrostatics, MBD), so it stays separable and scales.

### 12.1 What is available

| Family | Mode (train flag) | Physics | Deploy route |
|---|---|---|---|
| Electrostatics | `--long-range-mode reciprocal-spectral-v1` | learned latent scalar charge, reciprocal-space | C++ reciprocal solver (in-core for `direct_kspace`) |
| Electrostatics (multipole) | `… --long-range-reciprocal-backend mesh_fft --long-range-max-multipole-l {1,2}` | learned monopole/dipole/quadrupole, mesh-FFT | C++ mesh-FFT reciprocal solver |
| Dispersion (pairwise) | `--long-range-dispersion-mode pairwise-c6` | learned C6 + Becke–Johnson damping, r⁻⁶ | in-graph (rides inside the `.pt2`) |
| Dispersion (many-body) | `--long-range-dispersion-mode mbd-slq` | MBD@rsSCS coupled-dipole, matrix-free Tr[√C] | C++ MBD solver |
| Dispersion (dense) | `--long-range-dispersion-mode mbd` | dense QHO eigensolve | reference/validation only |

MBD-SLQ has two further axes:

- **Operator backend** `--mbd-operator-backend edge_sparse|pme_fft`. `edge_sparse` (default) sums the
  damped dipole tensor over the cutoff dispersion graph (direct, O(E), fastest at small/medium N);
  `pme_fft` is a reciprocal-only FFT matvec for large periodic boxes. **Match this across train and
  deploy** — the C++ solver runs the matching operator.
- **Polarizability rank** `--mbd-anisotropic`. Off = isotropic scalar α (emits an `[N,2]` source
  `(ω, α)`); on = **anisotropic l=2 tensor** α (emits `[N,8]` `(ω, α_iso, 6×B)`, coupling W=ω·B). The
  tensor is built from the l=2 node block, so it needs `--lmax ≥ 2`. It is ICTC-distinctive and nearly
  free (Section 12.5).

Electrostatics and dispersion are independent and combine in one model (e.g. multipole l=2 + MBD).

### 12.2 Training

MBD-SLQ, isotropic, default `edge_sparse` backend:

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 128 --lmax 2 --num-interaction 2 \
  --long-range-dispersion-mode mbd-slq \
  --dispersion-cutoff 8.0 \
  --mbd-operator-backend edge_sparse \
  --checkpoint model_mbd.pth
```

Anisotropic (l=2 tensor) MBD — add `--mbd-anisotropic` (needs `--lmax >= 2`):

```bash
python -m mace_ictc.cli.train ... \
  --long-range-dispersion-mode mbd-slq --dispersion-cutoff 8.0 \
  --mbd-operator-backend edge_sparse --mbd-anisotropic \
  --checkpoint model_mbd_aniso.pth
```

Pairwise-C6 dispersion (cheapest, in-graph):

```bash
python -m mace_ictc.cli.train ... --long-range-dispersion-mode pairwise-c6 --dispersion-cutoff 8.0 --checkpoint model_c6.pth
```

Multipole electrostatics (mesh-FFT, dipole+quadrupole) — the mesh-FFT multipole path requires
full-Ewald screening:

```bash
python -m mace_ictc.cli.train ... \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-reciprocal-backend mesh_fft --long-range-mesh-size 32 \
  --long-range-max-multipole-l 2 --long-range-mesh-fft-full-ewald \
  --long-range-assignment pcs --checkpoint model_mp.pth
```

Key dispersion/MBD flags:

| Flag | Default | Meaning |
|---|---|---|
| `--long-range-dispersion-mode` | `none` | `none` / `pairwise-c6` / `mbd` / `mbd-slq` |
| `--dispersion-cutoff` | `8.0` | dispersion neighbor cutoff (Å); ~8 ≈ 5% MBD-energy convergence; `0` reuses the model edge list |
| `--mbd-operator-backend` | `edge_sparse` | `edge_sparse` (direct) / `pme_fft` (reciprocal); match at deploy |
| `--mbd-anisotropic` | off | l=2 tensor polarizability (`[N,8]` source); needs `lmax ≥ 2` |
| `--mbd-pme-mesh-size` | `32` | PME mesh for `pme_fft` |
| `--dispersion-slq-num-probes` | `8` | Hutchinson probes for Tr[√C] |
| `--dispersion-slq-lanczos-steps` | `16` | Lanczos steps per probe |

The checkpoint stores every long-range hyperparameter in `model_hyperparameters`, so export rebuilds
the identical architecture.

### 12.3 Export

Two cores; both are parity-correct — choose by target:

| | AOTI `.pt2` (production) | TorchScript `.pt` (portable) |
|---|---|---|
| Tool | `mace_ictc.cli.export_aoti_core` | `mace_ictc.cli.export_libtorch_core` |
| Speed | Inductor-fused fwd+bwd, ~3.7–5.4× faster | C++ autograd, no fusion |
| N | static-N or N-dynamic | N-flexible (one core, any N) |
| Long-range | full (defers reciprocal/MBD to C++; C6 in-graph) | minimal |

Export from a trained checkpoint (inherits the long-range config, see Section 9):

```bash
python -m mace_ictc.cli.export_aoti_core --checkpoint model_mbd_aniso.pth --out model.pt2
```

The synthetic/combined export builds a long-range config at export time (for testing) and accepts:

| Flag | Meaning |
|---|---|
| `--dispersion-mode mbd-slq` | dispersion family |
| `--dispersion-cutoff 8.0` | dispersion cutoff (Å) |
| `--mbd-operator-backend edge_sparse\|pme_fft` | MBD operator (match training) |
| `--mbd-anisotropic` | anisotropic l=2 tensor (`[N,8]` source) |
| `--lr-mesh-size 32` | mesh size for mesh-FFT / `pme_fft` |
| `--long-range-mode`, `--long-range-multipole-l` | electrostatics block |

```bash
python -m mace_ictc.cli.export_aoti_core --route baseline --channels 128 --lmax 2 \
  --num-interaction 2 --dtype float32 --device cuda \
  --dispersion-mode mbd-slq --dispersion-cutoff 8.0 --mbd-operator-backend edge_sparse \
  --mbd-anisotropic --out model_aniso.pt2
```

The export writes a `model.pt2.json` sidecar with the deploy metadata; for MBD it includes
`long_range_mbd_source_channels` (2 isotropic / 8 anisotropic), `mbd_operator_backend`,
`long_range_mbd_beta`, and `long_range_mbd_coupling_scale`, which the C++ engine reads.

> **Deploy-defer**: the AOTI forward of an MBD or reciprocal model emits the per-atom source but
> **defers the long-range energy to the C++ solver**, so the forward stays ~constant regardless of the
> long-range method. The long-range cost appears only in the MD step (Section 12.5).

### 12.4 LAMMPS deployment

Pair-style grammar (`pair_mff_torch.cpp`):

```
pair_style mff/torch <model_cutoff> [cpu|cuda] [dispersion <disp_cutoff>]
pair_coeff * * <model.pt2> <elem_1> [<elem_2> ...]
```

A dispersion/MBD model **must** be given the `dispersion <disp_cutoff>` keyword: it builds the LAMMPS
ghost neighbor list (to `disp_cutoff`) that the C++ MBD/dispersion solver reuses. Use the **same
cutoff you trained with**. Everything else — operator backend, anisotropic source width, β, coupling
scale, probe counts — is baked into the `.pt2` metadata, so the LAMMPS input is identical for
isotropic and anisotropic MBD (the source width comes from metadata):

```lammps
units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data system.data
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch 5.0 cuda dispersion 8.0
pair_coeff * * model_mbd_aniso.pt2 H C N O

fix 1 all nve
run 1000
```

- **Pairwise-C6** also takes `dispersion <cutoff>` (to supply its neighbor list); the energy itself
  rides in-graph inside the `.pt2`.
- **Reciprocal electrostatics** needs no extra keyword — the engine runs its reciprocal solver from
  metadata and the mesh.
- **Mesh assignment is honored at deploy.** The C++ reciprocal solver reads `long_range_assignment`
  (`cic` / `pcs`) from the deploy metadata and runs the matching spread + sinc-window deconvolution
  stencil, so a multipole model trained with `--long-range-assignment pcs` reproduces its in-model
  energy/forces in LAMMPS (verified to ~2e-6 relative on OFF23-large l=2). Train and deploy stay
  matched automatically through the metadata — no extra flag.
- The `pair_coeff` element order must match the export `--elements` order; `NULL` skips a type.

Matching rules and constraints:

- **Backend must match** train↔deploy (`edge_sparse` vs `pme_fft`); the C++ runs whichever the
  metadata names.
- **Single-image cutoff** (`edge_sparse`): `2·dispersion_cutoff ≤` the smallest box face height
  (nearest image); otherwise the pair style errors — raise the box or lower the cutoff. `pme_fft` has
  no such limit.
- The deploy MBD energy uses a C++ Chebyshev trace estimator vs training's Lanczos/Newton–Schulz: the
  *operator* matches exactly, but the stochastic Tr[√C] estimator differs (not bit-exact; the same for
  isotropic and anisotropic).
- `pair_style mff/torch/kk` is the Kokkos/GPU variant; same grammar.

### 12.5 Cost and how to choose

Anisotropic vs isotropic MBD — 512 atoms, ch128, lmax2, `edge_sparse`, cutoff 8 (RTX 4090):

| Mode | Isotropic | Anisotropic | Δ |
|---|---|---|---|
| Train (fwd + force-bwd + loss-bwd) | 304.9 ms | 304.9 ms | +0.0% |
| Inference (fwd + force) | 109.3 ms | 109.6 ms | +0.3% |
| MD (deploy) | 35.3 ms/step | 37.3 ms/step | +5.7% |

Anisotropic adds the l=2 readout plus a per-atom 3×3 W matmul in each SLQ matvec: essentially free to
train/infer (MBD is launch/autograd overhead-bound) and ~6% in MD. (The eager train/infer absolutes
include an O(N²) dispersion search that real training avoids with precomputed edges; the Δ is clean.)

Decision rules:

- **Dispersion**: `pairwise-c6` is cheapest (in-graph, ~free) and a fine default when many-body effects
  are not needed; `mbd-slq` for many-body screening / polarization response.
- **MBD backend**: `edge_sparse` for small/medium systems (faster through ~8k atoms); `pme_fft` for
  large periodic boxes.
- **Anisotropic**: enable when the l=2 representation should drive directional polarizability (the
  ICTC-distinctive path); the cost is small.
- **Core**: AOTI `.pt2` for throughput; TorchScript `.pt` for N-flexible / portable deployment.

### 12.6 Training stability and warm-start (long-range heads)

**Warm-start is the recommended way to add ANY native long-range head** — MBD dispersion,
reciprocal electrostatics, or multipole electrostatics. Train (or convert) a short-range backbone
first, then add the long-range physics on top with a non-strict load (`--finetune`): the backbone is
warm-started and only the new head starts fresh. This converges fastest and is the most stable path
for all three heads. The same flag works for every head — the only thing that changes is which
`--long-range-*` flags you pass.

`--finetune` loads weights **non-strictly** (the missing long-range-head keys stay fresh, unexpected
keys are ignored) with a fresh optimizer from epoch 0. The architecture flags must still describe the
*full* model (backbone + the long-range head). To continue the full model afterwards, resume normally
with `--resume-checkpoint model.pth --resume-training-state` (strict load, optimizer restored).

**Why warm-start is safe: every head starts gentle.** A freshly-added head must contribute ~zero
energy at init so it does not destabilize the warm-started backbone, while staying trainable. Each
head has a built-in gentle start:

- **Electrostatics** (`--long-range-mode reciprocal-spectral-v1`, the default `direct_kspace`
  backend): the energy is gated by a learnable `energy_scale` initialized to **0**, so the
  contribution is exactly zero at init and ramps up as training proceeds (it bootstraps because the
  latent sources are non-zero, so `dE/d energy_scale ≠ 0`).
- **Multipole electrostatics** (`mesh_fft` + `--long-range-max-multipole-l {1,2}`): the reciprocal
  energy is *quadratic* in the q/μ/Q sources, so a learnable **gate** (`--long-range-multipole-gate-init`,
  default `0.1`) scales the emitted sources — the head starts at ~`gate²` of full strength (gentle)
  yet remains trainable. The gate is folded into the exported source, so train and deploy stay
  consistent. Lower it (e.g. `0.02`) for an especially stiff backbone; set `1.0` for the ungated
  from-scratch behavior.
- **MBD** is a coupled-dipole solve on a *learned* polarizability, so early in training the coupling
  matrix `C` can drift toward the polarization-catastrophe edge. Two numerical guards are built in and
  always on: a **detached spectral rescaling** keeps `C` strictly positive-definite at every step (the
  `Tr[√C]` estimator never sees a non-PD operator, so there is no hard NaN from the spectrum), and the
  anisotropic l=2 readout uses a **smooth norm** so the second-order (force-loss) gradient stays finite
  at initialization.

**Recipes** (all start from `backbone.pth` = a trained MACE-ICTC checkpoint with NO long-range, or a
converted MACE checkpoint):

```bash
# (a) MBD dispersion
python -m mace_ictc.cli.train --data-dir DATA \
  --channels 128 --lmax 2 --num-interaction 2 \
  --long-range-dispersion-mode mbd-slq --mbd-anisotropic --dispersion-cutoff 8.0 \
  --resume-checkpoint backbone.pth --finetune \
  --max-grad-norm 10 --lr 1e-3 \
  --checkpoint model_mbd_aniso.pth

# (b) electrostatics (learned latent scalar charges, reciprocal space)
python -m mace_ictc.cli.train --data-dir DATA \
  --channels 128 --lmax 2 --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --resume-checkpoint backbone.pth --finetune \
  --max-grad-norm 10 --lr 1e-3 \
  --checkpoint model_es.pth

# (c) multipole electrostatics (learned monopole/dipole/quadrupole, mesh-FFT)
python -m mace_ictc.cli.train --data-dir DATA \
  --channels 128 --lmax 2 --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-reciprocal-backend mesh_fft --long-range-mesh-fft-full-ewald \
  --long-range-max-multipole-l 2 --long-range-assignment pcs \
  --long-range-multipole-gate-init 0.1 \
  --resume-checkpoint backbone.pth --finetune \
  --max-grad-norm 10 --lr 1e-3 \
  --checkpoint model_multipole.pth
```

Electrostatics and dispersion are independent and can be combined in one model (e.g. multipole l=2 +
MBD) by passing both sets of flags. `--long-range-max-multipole-l` must not exceed the model `lmax`.

**From-scratch MBD** works too, but is more delicate:

- Always keep `--max-grad-norm 10` (the project standard). If the loss spikes, lower the LR or the
  dispersion cutoff — do **not** raise the clip.
- Two-interaction models (receptive field ≈ 2×cutoff) are generally stable once past the first ~20
  epochs; the rescaling fires gently and convergence is smooth.
- **Single-interaction** (`--num-interaction 1`) MBD is sensitive at `lr 1e-3`: the coarse receptive
  field makes α/ω swing batch-to-batch, the rescaling fires hard, and the loss can spike (NaN-free but
  convergence-wrecking). Warm-start it, or drop the LR.

**Eager vs. compiled for MBD.** `--train-makefx-compile` gives ~3× step throughput and is fine for the
throughput benchmark and medium runs (tens of epochs). For **long / production MBD training prefer
eager** (omit `--train-makefx-compile`): the compiled path holds one graph per atom-count bucket, and a
deep `torch.compile` multi-graph double-backward interaction can surface a *stochastic* NaN late in
long multi-shape runs (mitigated — not eliminated — by disabling AOTAutograd donated buffers). This is
a training-only, multi-graph issue: **AOTI deployment is single-graph and unaffected.**

## 13. Benchmarking

Main benchmark harness:

```bash
python -m mace_ictc.bench.bench_mace_ictc_vs_mace \
  --device cuda \
  --dtype float32 \
  --channels 64 \
  --atoms-list 256,1024,4096 \
  --configs 1:1,2:2,2:3 \
  --train-iters 5 \
  --infer-iters 20 \
  --out-dir /tmp/mace_ictc_bench
```

The benchmark reports rows for training and inference modes where supported. Treat it as a kernel/backend throughput harness, not a chemistry validation benchmark.

Recommended comparisons:

- Native `mace-torch` e3nn backend.
- Native `mace-torch` cuEq backend.
- MACE-ICTC bridge-U eager/make_fx/AOTI.
- MACE-ICTC cuEq product eager/make_fx/AOTI.
- Optional pure-U diagnostic path.

Always separate:

- first compile time,
- steady-state step time,
- ASE/Python overhead,
- neighbor-list overhead,
- LAMMPS throughput.

## 14. Tests and Validation

Core smoke tests:

```bash
python -m mace_ictc.test.test_training_smoke
python -m pytest mace_ictc/test/test_angular_basis.py -q
python -m pytest mace_ictc/test/test_export_aoti_core.py -q
```

MACE converter validation:

```bash
python -m mace_ictc.test.test_mace_converter
```

cuEq product tests:

```bash
python -m pytest mace_ictc/test/test_cueq_product_backend.py -q
python -m mace_ictc.test.test_cueq_makefx_training
```

Use a CUDA machine for the cuEq and make_fx tests.

Expected parity levels depend on dtype and backend:

- Float64 bridge-U conversion should reach near machine precision for energy/forces.
- Float32 cuEq paths should be judged with float32 tolerances.
- AOTI and make_fx paths must be compared against eager outputs after compile/load.

## 15. Common Pitfalls

### Bridge-U and `angular_basis=e3nn`

Bridge-U does not have an e3nn fold path. Use:

```bash
--product-backend ictd-bridge-u --angular-basis ictd
```

For e3nn-folded product inference, use:

```bash
--cueq-product --angular-basis e3nn
```

### cuEq Product Replacement

When replacing bridge-U products with cuEq products, only learnable MACE contraction weights should be copied. Fixed bridge-U `U_matrix_*` buffers already contain the ICTC/e3nn basis fold and must not be copied into cuEq.

The export path handles this.

### `use_reduced_cg`

This is a model-structure choice, not a harmless speed flag. If converting native MACE, follow the source checkpoint. If training from scratch, choose intentionally and keep it in checkpoint metadata.

### ScaleShift and E0

MACE-style ScaleShift affects interaction energy. E0 reference energies are added separately. To compare against native MACE or deploy absolute energies, make sure:

- atomic energy keys/values match,
- scale/shift match,
- `avg_num_neighbors` matches,
- export uses `--embed-e0` when the deployment expects absolute energy.

### `max_ell` Versus `lmax`

- `lmax`: hidden feature angular cutoff.
- `max_ell`: edge spherical-harmonics cutoff.

Native MACE often allows `max_ell >= hidden_lmax`. Higher `max_ell` can be much more expensive, especially for product contractions and force training.

### Dynamic Shapes

Dynamic AOTI and make_fx paths are sensitive to PyTorch/Inductor version. If a dynamic export fails, try:

- fixed-N export with `--static-n`,
- fewer dynamic dimensions,
- smaller buckets,
- PyTorch 2.7+,
- disabling optional fusion/autotune flags.

### MBD / dispersion training instability

If MBD training spikes or NaNs, it is almost always one of: (1) training from scratch instead of
warm-starting (add MBD on a trained backbone with `--resume-checkpoint … --finetune` — see Section
12.6); (2) a missing or raised gradient clip (keep `--max-grad-norm 10`; lower the LR or cutoff
instead); (3) single-interaction MBD at `lr 1e-3` (warm-start or drop the LR); or (4) a long
`--train-makefx-compile` run (use eager for long/production MBD training; AOTI deployment is
unaffected). The polarizability solve is kept positive-definite at every step, so a hard NaN is rare
and usually points at one of these. See Section 12.6.

## 16. Development Notes

Local code style is intentionally conservative:

- Prefer existing model/backend abstractions.
- Keep MACE parity tests before changing angular basis or product code.
- Do not treat passing smoke tests as proof of MACE parity; run direct MACE-vs-ICTC converter tests for parity claims.
- When touching checkpoint metadata, verify `LAMMPS_MLIAP_MFF.from_checkpoint` strict reload.
- When touching `angular_basis=e3nn`, verify both eager forward and checkpoint reload to avoid double-folding fixed buffers.

Useful files:

| File | Why it matters |
|---|---|
| `mace_ictc/models/pure_cartesian_ictd_fix.py` | Main model and product backends. |
| `mace_ictc/mace_basis.py` | Orthogonal ICTC/e3nn basis conversion. |
| `mace_ictc/interfaces/mace_converter.py` | Native MACE to MACE-ICTC weight conversion. |
| `mace_ictc/cli/export_aoti_core.py` | AOTI export, cuEq product replacement, angular-basis export logic. |
| `mace_ictc/training/makefx_compile.py` | `make_fx` force-step compilation. |
| `mace_ictc/training/train_loop.py` | Trainer, checkpoint metadata, ScaleShift/E0 loss handling. |
| `mace_ictc/interfaces/lammps_mliap.py` | Deployment checkpoint reload and wrapper logic. |
| `mace_ictc/test/test_mace_converter.py` | Native MACE to MACE-ICTC conversion parity tests. |
