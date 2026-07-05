# MACE-OFF23 Small MD Parity Check

Remote host: `XHPC-4090-01` via `ssh -p 18022 ylzhang@10.10.3.21`

Model artifacts:

- Native source model: `/home/ylzhang/.cache/mace/MACE-OFF23_small.model`
- MACE-ICTC FP32 checkpoint: `/tmp/mace_ictc_pretrained/off23_small_ictd_bridge_u_float32.pth`
- MACE-ICTC FP64 checkpoint generated for this check: `/tmp/mace_ictc_pretrained/off23_small_ictd_bridge_u_float64.pth`

System:

- Benzene grids generated from `ase.build.molecule("C6H6")`
- Non-periodic boxes, 0.25 fs VelocityVerlet, 100 K initial velocities
- OFF23 small was used because it is an organic pretrained model; the benzene grids are implementation-parity systems, not liquid-state production benchmarks.

Native reference:

- Native MACE was run with `mace==0.3.16` and `e3nn==0.4.4` from `/tmp/mace_torch_0_3_16` and `/tmp/e3nn_0_4_4`.
- MACE-ICTC was run from `/home/ylzhang/lrx/MACE-ICTC` in the current FSCETP environment.

Main FP64 parity results:

| case | atoms | steps | max same-frame energy error | max same-frame energy error (meV/atom) | RMS same-frame force error | max independent-trajectory RMS position error |
|---|---:|---:|---:|---:|---:|---:|
| `benzene_1` | 12 | 100 | `2.73e-12 eV` | `2.27e-10` | `2.31e-15 eV/A` | `8.63e-16 A` |
| `benzene_128_grid` | 1536 | 10 | `3.96e-09 eV` | `2.58e-09` | `2.52e-15 eV/A` | `1.58e-16 A` |

FP32 observation:

- Forces still agree at about `1e-6 eV/A` RMS.
- Total-energy absolute differences grow with atom count because FP32 atomic-energy and accumulation differences are extensive. For example, `benzene_128_grid` reached `7.25 eV` max total-energy difference while the force RMS difference stayed `1.43e-6 eV/A`.
- For strict numerical-parity claims, report the FP64 bridge-U result. Treat FP32 as a deployment-precision result, not a bit-level parity result.

