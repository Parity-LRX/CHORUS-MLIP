# OFF23 Small Pretrained Check

Remote host: `XHPC-4090-01` via `ssh -p 18022 ylzhang@10.10.3.21`

Software path:

- MACE-ICTC: `/home/ylzhang/lrx/MACE-ICTC`
- Python: `/home/ylzhang/micromamba/envs/FSCETP/bin/python`
- `mace-torch`: `/tmp/mace_torch_0_3_16` (`mace==0.3.16`)
- Current default `e3nn`: `0.5.9`
- Legacy native-OFF compatibility `e3nn`: `/tmp/e3nn_0_4_4` (`e3nn==0.4.4`)

Model:

- Source: `/home/ylzhang/.cache/mace/MACE-OFF23_small.model`
- Converted checkpoint: `/tmp/mace_ictc_pretrained/off23_small_ictd_bridge_u_float32.pth`
- Species: `H,C,N,O,F,P,S,Cl,Br,I`
- MACE config extracted from checkpoint:
  - `num_interactions=2`
  - `hidden_irreps=96x0e`
  - `max_ell=3`
  - `correlation=3`
  - `avg_num_neighbors=15.652482986450195`
  - `use_reduced_cg=False`
  - `radial_type=bessel`, `radial_MLP=[64, 64, 64]`

## Loader Compatibility

The OFF23 small pickle is not directly runnable under the current `e3nn==0.5.9`:

- direct `torch.load(..., weights_only=False)` fails because old `CodeGenMixin.__codegen__` stores compiled modules as raw bytes, while current e3nn expects `(buffer_type, buffer)` pairs;
- after a temporary codegen loader patch, native MACE forward still fails because old pickled `SphericalHarmonics` and `Activation` objects miss runtime fields expected by e3nn 0.5.9.

For native-MACE reference output, the model was run with isolated `e3nn==0.4.4` from `/tmp/e3nn_0_4_4`. MACE-ICTC conversion and `.pt2` export were run with the normal current environment.

## Native MACE vs Converted MACE-ICTC

The comparison used synthetic fixed-edge graphs with 50 directed edges per atom. These are kernel/parity stress graphs, not physical MD configurations; random short distances make absolute energies and forces very large.

| atoms | directed edges | native MACE ms | ICTC wrapper ms | speedup | rel energy diff | rel force diff |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 6,400 | 12.232 | 11.391 | 1.074x | 7.729e-07 | 1.118e-06 |
| 512 | 25,600 | 13.617 | 13.905 | 0.979x | 3.079e-07 | 1.126e-06 |

## AOTI `.pt2` Export

Common export settings:

- `atoms=512`
- `degree=50`
- `vary_atoms=128`
- `vary_degree=40`
- `--assume-cutoff-edges`
- `--preserve-edge-order`
- `--fuse-selector-message-linear`
- dynamic N/E enabled by default
- FP32, TF32 not enabled

| exported core | output | AOTI strict | numerical match | equivariance | vary E | vary N | eager ms | AOTI ms | speedup | peak CUDA memory |
|---|---|---:|---|---|---|---|---:|---:|---:|---:|
| bridge-U AOTI | `/tmp/mace_ictc_pretrained/off23_small_ictd_bridge_u_n512_d50.pt2` | true | pass, `dE_rel=4.31e-06`, `dF_rel=3.29e-06` | pass | pass | pass | 11.228 | 4.343 | 2.59x | 0.71 GB |
| cuEq-product AOTI, `angular_basis=e3nn` | `/tmp/mace_ictc_pretrained/off23_small_ictd_cueq_e3nn_n512_d50.pt2` | false | pass, `dE_rel=3.47e-06`, `dF_rel=3.07e-06` | pass | pass | pass | 13.757 | 4.703 | 2.93x | 0.64 GB |

Important caveat: for this OFF23 small model, `hidden_lmax=0`. On this case, bridge-U AOTI was slightly faster than cuEq-product AOTI at the tested size. Do not generalize the synthetic high-`lmax` cuEq conclusion to this scalar-hidden pretrained model.

Current exporter limitation observed in this run: `--embed-e0` and `--cueq-product` cannot currently be combined because the E0 wrapper is applied before product replacement, and the wrapper does not expose `.products`. The bridge-U AOTI path above was exported with `--embed-e0`; the cuEq-product AOTI path was exported without `--embed-e0`.

## LAMMPS Smoke / Throughput

LAMMPS executables found on the remote host:

- ordinary build: `/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch/lmp`
- Kokkos/AOTI build: `/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch-kk/lmp`

The ordinary build attempted to load `.pt2` through the TorchScript path and failed with `file in archive is not in a subdirectory: version`. The Kokkos/AOTI build loaded the same bridge-U `.pt2` successfully.

For cuEq-product `.pt2`, LAMMPS needs the cuEquivariance custom op libraries and an embedded Python runtime:

```bash
PYTHONHOME=/home/ylzhang/micromamba/envs/FSCETP
PYTHONPATH=/home/ylzhang/micromamba/envs/FSCETP/lib/python3.11:/home/ylzhang/micromamba/envs/FSCETP/lib/python3.11/site-packages
MFF_LIBPYTHON=/home/ylzhang/micromamba/envs/FSCETP/lib/libpython3.11.so
MFF_CUSTOM_OPS_LIB=/home/ylzhang/micromamba/envs/FSCETP/lib/python3.11/site-packages/cuequivariance_ops/lib/libcue_ops.so:/home/ylzhang/micromamba/envs/FSCETP/lib/python3.11/site-packages/cuequivariance_ops_torch/_ext/cuequivariance_ops_torch_ext.cpython-311-x86_64-linux-gnu.so
```

LAMMPS results below use `pair_style mff/torch 4.5 cuda` through `build-mfftorch-kk/lmp`.

| atoms | full neighbors | avg neighbors/atom | core | steps | loop time s | steps/s | katom-step/s | pair ms/step | notes |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 128 | 3,282 | 25.64 | bridge-U AOTI | 50 | 0.0736007 | 679.342 | 86.956 | 1.459 | `--embed-e0`, dynamic `.pt2` |
| 128 | 3,282 | 25.64 | cuEq-product AOTI | 50 | 0.216624 | 230.814 | 29.544 | 4.318 | no E0 embedding, custom-op env required |
| 512 | 40,960 | 80.00 | bridge-U AOTI | 100 | 0.669346 | 149.400 | 76.493 | 5.817 | zero initial velocity, `timestep=1e-6` |
| 512 | 40,960 | 80.00 | cuEq-product AOTI | 100 | 0.685753 | 145.825 | 74.662 | 5.979 | zero initial velocity, custom-op env required |

The 512-atom grid is an artificial H/C/N/O periodic grid used to keep atom count and neighbor count controlled. With the default timestep it lost atoms because forces were too large, so the throughput run used zero initial velocity and `timestep=1e-6`. Treat these rows as LAMMPS integration/throughput checks, not stable MD production trajectories.
