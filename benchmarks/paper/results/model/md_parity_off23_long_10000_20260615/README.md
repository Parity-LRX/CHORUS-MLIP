# MACE-OFF23 Long MD Parity Check

Remote host: `XHPC-4090-01` via `ssh -p 18022 ylzhang@10.10.3.21`

Purpose:

- Validate long-trajectory numerical correspondence between a native `mace-torch` pretrained model and its MACE-ICTC bridge-U conversion.
- The reported errors are implementation-parity errors on identical frames and independently integrated trajectories, not dataset accuracy metrics.

Model and runtime:

- Native source model: `/home/ylzhang/.cache/mace/MACE-OFF23_small.model`
- MACE-ICTC checkpoint: `/tmp/mace_ictc_pretrained/off23_small_ictd_bridge_u_float64.pth`
- Native MACE runtime: `mace==0.3.16`, `e3nn==0.4.4`
- MACE-ICTC runtime: `/home/ylzhang/lrx/MACE-ICTC`
- Device: RTX 4090
- Precision: FP64

Systems and protocol:

- Systems: ethanol, acetic acid, acetamide, benzene from ASE molecule geometries.
- Dynamics: non-periodic VelocityVerlet, 0.25 fs, 100 K initial velocities, 10000 steps.
- For each system, native MACE and MACE-ICTC were independently integrated from the same initial positions and velocities.
- MACE-ICTC was also evaluated on every native-MACE frame to separate same-frame numerical error from any trajectory divergence.

Summary:

| system | atoms | steps | max same-frame energy error (eV) | RMS same-frame force error (eV/A) | max independent-trajectory RMS position error (A) | native ms/step | ICTC ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| ethanol | 9 | 10000 | `3.64e-12` | `1.90e-15` | `6.79e-12` | `18.03` | `14.22` |
| acetic acid | 8 | 10000 | `2.73e-12` | `2.23e-15` | `1.63e-12` | `18.17` | `14.05` |
| acetamide | 9 | 10000 | `2.73e-12` | `2.30e-15` | `4.78e-11` | `18.12` | `14.17` |
| benzene | 12 | 10000 | `3.64e-12` | `2.22e-15` | `2.54e-12` | `18.02` | `14.14` |

Interpretation:

- In FP64, the converted MACE-ICTC bridge-U model reproduces the native MACE model to numerical precision over 10000-step trajectories for these representative organic systems.
- The independent trajectories remain essentially identical because force differences stay near machine precision.
- The timing columns include ASE calculator overhead, neighbor construction, Python dispatch, and force evaluation. They should not be used as pure operator throughput numbers.
- These systems are intentionally within the MACE-OFF organic chemistry domain. Water or liquid-water benchmarks should use a pretrained model whose training domain explicitly covers those systems; otherwise failures would not isolate implementation parity.

