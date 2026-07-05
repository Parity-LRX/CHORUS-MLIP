# LAMMPS USER-MFFTORCH

`USER-MFFTORCH` is a LAMMPS package for running exported MACE-ICTC models through
LibTorch/AOTInductor from C++. It is intended for production MD after a model has
already been trained or converted in Python and exported as a deployment core.

Full build notes: [docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md).

## What It Provides

Pair styles:

- `pair_style mff/torch`: C++ + LibTorch/AOTI inference path. Use this first for
  build validation and numerical smoke tests.
- `pair_style mff/torch/kk`: Kokkos/CUDA data-preparation path plus LibTorch/AOTI
  inference. Use this for GPU MD when the Kokkos build is available.

The recommended model artifact is an AOTInductor `.pt2` core exported by
`mff-export-aoti` or `python -m mace_ictc.cli.export_aoti_core`. Legacy
TorchScript `core.pt` support is still kept for older LibTorch deployments, but
new validation uses `.pt2`.

Current supported deployment output is energy and force. Physical tensor outputs
are not a supported public LAMMPS interface in the current `.pt2` deployment
path; do not rely on `compute mff/torch/phys` for production runs.

## Install Into LAMMPS

From the MACE-ICTC repository root:

```bash
bash scripts/install_user_mfftorch_into_lammps.sh /path/to/lammps
```

Manual copy, if needed:

- `lammps_user_mfftorch/src/USER-MFFTORCH/` to `LAMMPS/src/USER-MFFTORCH/`
- `lammps_user_mfftorch/cmake/Modules/Packages/USER-MFFTORCH.cmake` to
  `LAMMPS/cmake/Modules/Packages/USER-MFFTORCH.cmake`

Some LAMMPS versions also require adding `USER-MFFTORCH` to package lists in
`LAMMPS/cmake/CMakeLists.txt`. The full guide covers the exact edits.

## Build

Example CUDA/Kokkos build:

```bash
cd /path/to/lammps
export LIBTORCH_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake -S cmake -B build-mfftorch-kk \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D PKG_USER-MFFTORCH=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON

cmake --build build-mfftorch-kk -j
```

For a non-Kokkos validation build, drop the Kokkos options and use a separate
build directory, for example `build-mfftorch`.

At runtime, make sure the PyTorch shared libraries are visible:

```bash
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
```

## Export a Model

Basic `.pt2` export from a MACE-ICTC checkpoint:

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model.pt2 \
  --dynamic \
  --dtype float32 \
  --device cuda \
  --embed-e0
```

For fixed-size MD systems, a static-N export is often the most conservative
LAMMPS smoke-test target:

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --atoms 256 \
  --degree 32 \
  --static-n \
  --dtype float32 \
  --device cuda \
  --embed-e0 \
  --out model_static256.pt2
```

Rules that matter:

- `--elements` must match the element order passed to LAMMPS `pair_coeff`.
- `--embed-e0` makes LAMMPS `pe` include atomic reference energies.
- Static export capacity is controlled by `--atoms` and `--degree`; a LAMMPS
  run that exceeds the exported capacity is invalid.
- Always check the exporter's eager-vs-compiled energy/force comparison before
  running LAMMPS.

## LAMMPS Input

Minimal example:

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O

velocity all create 300 42
fix 1 all nve
run 100
```

Notes:

- `5.0` is the cutoff in Angstrom.
- The `pair_coeff` element list maps LAMMPS atom types to element symbols. Use
  `NULL` for atom types that should be skipped.
- Both `mff/torch` and `mff/torch/kk` can load `.pt2`; the `/kk` variant uses
  Kokkos for the GPU data path.

## Multi-GPU MPI Runs

LAMMPS multi-GPU execution uses MPI decomposition: run one MPI rank per GPU.
Do not expect one LAMMPS rank to drive all GPUs.

Plain LibTorch/AOTI path:

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lmp -in in.mfftorch
```

with:

```lammps
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O
```

Kokkos GPU data path:

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lmp -k on g 2 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
```

with either:

```lammps
pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O
```

or `pair_style mff/torch` when `-sf kk` mapping is available in the build.

The plain `mff/torch` engine selects the local CUDA device from local-rank
environment variables such as `SLURM_LOCALID`, `LOCAL_RANK`,
`OMPI_COMM_WORLD_LOCAL_RANK`, or `MPI_LOCALRANKID`. The Kokkos variant maps
ranks to Kokkos GPUs correctly for the common single-node, one-rank-per-GPU
case. Validate multi-node Kokkos runs before production.

With `MFF_DEBUG_BUNDLE=1`, initialization prints the requested and selected
device. Check that different local ranks select different GPUs. Before a long
MD run, compare `run 0` energy and forces for `-np 1` and `-np N`. Small fp32
roundoff differences are possible; large differences usually mean a domain
decomposition, cutoff, ghost, or static `.pt2` capacity issue.

## OFF23 Smoke Test

On an RTX 4090 validation host, `MACE-OFF23_small.model` was converted to
bridge-U MACE-ICTC, exported as a float32 static-6 `.pt2`, and loaded by both
`build-mfftorch` and `build-mfftorch-kk`.

Fresh LAMMPS `run 0` result:

| Quantity | LAMMPS `mff/torch` | Python checkpoint |
|---|---:|---:|
| Energy (eV) | `-6633.036` | `-6633.03613281` |
| Max absolute force component (eV/A) | `11.767612` | `11.76760674` |

The corresponding float64 conversion bridge check against native `mace-torch`
on a benzene same-frame trajectory gave:

- max energy difference: `2.73e-12 eV`
- max force-component difference: `4.44e-15 eV/A`

LAMMPS thermo `fmax` is the maximum absolute force component. Compare it against
`max(abs(forces))` from Python, not against the maximum per-atom force-vector
norm.

Older public OFF23 pickle files may require a matching historical
`mace-torch`/`e3nn` environment for the initial load/conversion step. Once
converted, the MACE-ICTC checkpoint can be exported by the current runtime.
