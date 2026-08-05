# CHORUS-Persistent C128/R16 formal benchmark

All five systems completed under the fixed formal configuration:
128 channels, \(l_{\max}=2\), \(L_{\max}=2\), two interactions,
correlation order 3, Hermitian density rank 16, full-nonlinear gate and
persistent phase scope. Runs used strict float32, no TF32, no EMA, a
5 Å cutoff, Energy:Force \(=1:100\), seed 20260616 and MakeFX.

Every model was selected by the minimum validation Force MAE, with the
earliest step breaking ties. Test data were not used for checkpoint
selection. The metrics in each row therefore belong to one selected
checkpoint; Energy and Force results have not been spliced across states.
Transition1x Energy values below include the fixed train-only elemental
residual calibration. Forces are unchanged by that calibration.

## Selected-checkpoint test results

| System | Selected step | Energy MAE (meV/atom) | Energy RMSE (meV/atom) | Force MAE (meV/Å) | Force RMSE (meV/Å) |
|:--|--:|--:|--:|--:|--:|
| Transition1x, train-only calibrated | 84,348 | 21.559 | 30.529 | 82.342 | 162.255 |
| xxMD MAL | 40,250 | 13.409 | 26.303 | 156.332 | 465.487 |
| xxMD STI | 14,400 | 24.364 | 37.676 | 146.185 | 500.757 |
| MD22 Buckyball | 44,400 | 0.184 | 0.233 | 6.297 | 8.525 |
| 3BPA 300 K | 39,424 | 0.153 | 0.197 | 8.457 | 13.055 |
| 3BPA 600 K | 39,424 | 0.344 | 0.494 | 21.982 | 45.389 |
| 3BPA 1200 K | 39,424 | 3.174 | 4.582 | 120.166 | 262.343 |

## Transition1x calibration audit

The selected raw checkpoint has test Energy MAE/RMSE of
26.970/35.973 meV/atom and Force MAE/RMSE of
82.342/162.255 meV/Å. A closed-form elemental residual was fitted using
only the 50,000 training structures. It changes the test Energy MAE/RMSE to
21.559/30.529 meV/atom and leaves every force metric unchanged.

## Execution record

- 4090 Transition1x training, selected-checkpoint test and train-only
  calibration completed at 2026-07-30 19:10 CST.
- A100 Slurm job 54040 completed MAL and Buckyball with exit code 0.
- A100 Slurm job 54041 completed STI and all three 3BPA temperatures with
  exit code 0.
- These jobs ran concurrently, so their wall time and throughput are not
  admissible for the formal speed comparison.
- Exact eV-valued metrics, validation checkpoint counts and source paths are
  retained in `results.json`.
