# MACE-ICTC depth and angular-resolution cost controls: extended matrix

Strict-FP32 MakeFX benchmark on one NVIDIA A100 40 GB, extending
`mace_depth_lmax_scaling_a100_20260815` to every Table 3 configuration.
All models use C128 (widened control C140), correlation order 3, a fixed
directed degree of 32, and the same synthetic-graph protocol. Inference
includes energy and conservative forces; training is a complete
energy--force optimiser update including force double backward.

Script: `benchmarks/paper/scripts/throughput/run_mace_depth_lmax_scaling_extended_a100.sbatch`
(Slurm job 56075). The two `lmax3` training cells initially recorded a
MakeFX compile error at 2048 atoms; a rerun with `require_makefx=False`
(job 56080, `run_mace_lmax3_train2048_retry_a100.sbatch` plus diagnostic
copy) showed the compile step itself raises CUDA OOM and the eager fallback
also OOMs, so those cells are reported as OOM.

## 2048-atom endpoints (katoms/s, GiB)

| Configuration | Parameters | Inference | Training | Inference memory | Training memory | Inference 4096 |
|---|---:|---:|---:|---:|---:|---:|
| L1 Off | 411,072 | 109.505 | 38.652 | 2.095 | 3.922 | 110.646 |
| L1 Final R16 | 482,756 | 88.458 | 29.388 | 2.469 | 5.682 | 89.982 |
| L1 Persistent R16 | 619,978 | 77.480 | 25.957 | 2.690 | 6.677 | 78.699 |
| Lmax3 Final R16 | 1,277,896 | 8.707 | OOM | 24.316 | OOM | OOM |
| Lmax3 Persistent R16 | 1,657,172 | 8.009 | OOM | 25.505 | OOM | OOM |
| L2 Final R16, 3 interactions | 1,381,510 | 14.397 | 5.358 | 12.908 | 30.845 | 14.383 |
| L2 Persistent R16, 3 interactions | 1,857,048 | 11.375 | OOM | 14.744 | OOM | 11.464 |
| Off, widened C140 | 762,692 | 25.408 | 10.988 | 8.671 | 16.352 | 25.451 |

Checkpoints: T1X checkpoints from `table4_lmax_controls_c128_2layer_r16`
(L1, L3 Final), `chorus_table4_lmax_controls_c128_2layer_r16_20260814`
(L3 Persistent), `chorus_depth3_scope_20260801` (depth-3 Final/Persistent)
and `ictc_c140_matched_all_20260729` (widened), staged on the A100 under
`chorus_lmax_depth_extended_assets_20260818`.
