# NequIP OpenEquivariance cost controls on A100: extended matrix

Strict-FP32 energy/force benchmark on one NVIDIA A100, extending
`nequip_oeq_depth_lmax_scaling_a100_20260815` to every Table 3
configuration: lmax=1 and lmax=3 Final/Persistent, three-interaction
Final/Persistent, and the widened C72 three-interaction Off control.
OpenEquivariance is enabled and audited for every configuration.

Script: `benchmarks/paper/scripts/throughput/run_nequip_depth_lmax_scaling_extended_oeq_a100.sbatch`
(Slurm array job 56067, eight tasks). Raw machine-readable records are in
`raw/`.

## 2048-atom endpoints (katoms/s, GiB)

| Configuration | Parameters | Inference | Training | Inference memory | Training memory | Inference 4096 |
|---|---:|---:|---:|---:|---:|---:|
| L1 Off | 217,384 | 282.52 | 87.50 | 0.618 | 1.648 | 331.20 |
| L1 Final R16 | 262,796 | 171.64 | 57.18 | 0.788 | 2.149 | 197.65 |
| L1 Persistent R16 | 286,706 | 138.26 | 46.43 | 0.847 | 2.468 | 157.53 |
| Lmax3 Final R16 | 1,232,526 | 27.78 | 10.84 | 6.541 | 12.736 | 28.47 |
| Lmax3 Persistent R16 | 1,337,336 | 23.33 | 9.08 | 6.803 | 13.825 | 23.89 |
| L2 Final R16, 3 layers | 852,621 | 51.44 | 18.77 | 2.957 | 6.673 | 54.26 |
| L2 Persistent R16, 3 layers | 1,100,125 | 32.00 | 11.99 | 3.477 | 9.224 | 33.79 |
| Off, widened C72, 3 layers | 835,820 | 72.95 | 24.83 | 2.445 | 5.502 | 76.49 |

Configs come from `nequip_chorus_runs/lmax_controls_c64_r16_20260815`
(L1, L3), `depth2_scope_c64_r16_20260801` (depth-3 via `--num-layers 3`)
and `extended_c72_r16_20260729` (widened, lmax=2, 3 layers).
