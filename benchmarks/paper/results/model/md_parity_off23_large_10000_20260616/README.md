# MACE-OFF23 Large-System MD Parity Check

Remote host: `XHPC-4090-01` via `ssh -p 18022 ylzhang@10.10.3.21`

Purpose:

- Validate native-MACE vs MACE-ICTC bridge-U numerical correspondence on a larger organic system.
- This is a long implementation-parity run, not a liquid-state physical benchmark.

Model and runtime:

- Native source model: `/home/ylzhang/.cache/mace/MACE-OFF23_small.model`
- MACE-ICTC checkpoint: `/tmp/mace_ictc_pretrained/off23_small_ictd_bridge_u_float64.pth`
- Native runtime: `mace==0.3.16`, `e3nn==0.4.4`
- MACE-ICTC runtime: `/home/ylzhang/lrx/MACE-ICTC`
- Device: RTX 4090
- Precision: FP64

System and protocol:

- System: 64 benzene molecules in a sparse non-periodic grid.
- Atom count: 768.
- Dynamics: VelocityVerlet, 0.25 fs, 10000 steps.
- Recording/evaluation stride: 20 steps, giving 501 recorded frames.
- Native MACE and MACE-ICTC were independently integrated from the same initial positions and velocities.
- MACE-ICTC was additionally evaluated on the recorded native-MACE frames.

Result:

| system | atoms | steps | recorded frames | max same-frame energy error (eV) | max same-frame energy error (meV/atom) | RMS same-frame force error (eV/A) | max independent-trajectory RMS position error (A) | native ms/step | ICTC ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| benzene_64_grid | 768 | 10000 | 501 | `4.77e-09` | `6.21e-09` | `2.38e-15` | `2.76e-11` | `109.75` | `92.57` |

Interpretation:

- FP64 bridge-U conversion remains numerically aligned with native MACE for a 768-atom, 10000-step trajectory.
- The total-energy absolute error is larger than in single-molecule systems because total energy is extensive, but the per-atom error remains negligible and force errors stay near machine precision.
- Timing includes ASE calculator overhead, neighbor construction, Python dispatch, and autograd force evaluation. It should not be interpreted as isolated tensor-product throughput.

