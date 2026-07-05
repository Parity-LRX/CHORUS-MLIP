# MD17 convergence analysis

All force RMSE values are in eV/A and all energy RMSE values are in eV/atom.
Scalar validation losses are intentionally not compared because MACE-ICTC and mace-torch log different internal loss normalizations.
Partial rows are included so long-running jobs can be monitored before all modes finish.

## Per-run summary

| dataset | mode | seed | status | final epoch | best F | best F epoch | E at best F | best E | best E epoch |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| revised_benzene | ictd_bridge_u_eager | 20260616 | complete | 299 | 0.0033 | 295 | 0 | 0 | 295 |
| revised_benzene | ictd_bridge_u_eager | 20260617 | complete | 299 | 0.0031 | 286 | 0.0003 | 0 | 209 |
| revised_benzene | ictd_bridge_u_eager | 20260618 | complete | 299 | 0.0035 | 296 | 0.0004 | 0 | 281 |
| revised_benzene | ictd_bridge_u_makefx | 20260616 | complete | 299 | 0.0032 | 297 | 0.0003 | 0.0001 | 85 |
| revised_benzene | ictd_bridge_u_makefx | 20260617 | complete | 299 | 0.0031 | 264 | 0.0001 | 0 | 229 |
| revised_benzene | ictd_bridge_u_makefx | 20260618 | complete | 299 | 0.0035 | 287 | 0.0003 | 0.0001 | 40 |
| revised_benzene | ictd_cueq_makefx | 20260616 | complete | 299 | 0.0032 | 297 | 0.0002 | 0.0001 | 59 |
| revised_benzene | ictd_cueq_makefx | 20260617 | complete | 299 | 0.0029 | 295 | 0.0002 | 0 | 246 |
| revised_benzene | ictd_cueq_makefx | 20260618 | complete | 299 | 0.0035 | 287 | 0.0001 | 0 | 274 |
| revised_benzene | mace_e3nn | 20260616 | complete | 299 | 0.0112 | 299 | 0.00108 | 0.00015 | 219 |
| revised_benzene | mace_e3nn | 20260617 | complete | 299 | 0.01012 | 287 | 0.00071 | 0.00013 | 232 |
| revised_benzene | mace_e3nn | 20260618 | complete | 299 | 0.0108 | 297 | 0.00016 | 0.00013 | 292 |
| revised_benzene | mace_cueq | 20260616 | complete | 299 | 0.01346 | 297 | 0.00038 | 0.00016 | 273 |
| revised_benzene | mace_cueq | 20260617 | complete | 299 | 0.01176 | 299 | 0.00056 | 0.00014 | 224 |
| revised_benzene | mace_cueq | 20260618 | complete | 299 | 0.01269 | 290 | 0.00055 | 0.00015 | 234 |

## Aggregate by mode

| dataset | mode | runs | complete | best F mean | best F std | best E mean | best E std | mean log10 F |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| revised_benzene | ictd_bridge_u_eager | 3 | 3 | 0.0033 | 0.0002 | 0 | 0 | -2.13398 |
| revised_benzene | ictd_bridge_u_makefx | 3 | 3 | 0.00326667 | 0.000208167 | 6.66667e-05 | 5.7735e-05 | -2.13786 |
| revised_benzene | ictd_cueq_makefx | 3 | 3 | 0.0032 | 0.0003 | 3.33333e-05 | 5.7735e-05 | -2.13996 |
| revised_benzene | mace_e3nn | 3 | 3 | 0.0107067 | 0.000546016 | 0.000136667 | 1.1547e-05 | -1.75175 |
| revised_benzene | mace_cueq | 3 | 3 | 0.0126367 | 0.000851254 | 0.00015 | 1e-05 | -1.70066 |

## Force convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.02 | 0.015 | 0.01 | 0.0075 | 0.005 | 0.0035 |
|---|---|---:|---:|---:|---:|---:|---:|
| revised_benzene | ictd_bridge_u_eager | 25 (3/3) | 33.3333 (3/3) | 49.3333 (3/3) | 77 (3/3) | 139 (3/3) | 259 (3/3) |
| revised_benzene | ictd_bridge_u_makefx | 27.6667 (3/3) | 33 (3/3) | 50.3333 (3/3) | 81.3333 (3/3) | 146.333 (3/3) | 255.333 (3/3) |
| revised_benzene | ictd_cueq_makefx | 24.3333 (3/3) | 31.3333 (3/3) | 49.6667 (3/3) | 77.6667 (3/3) | 136.333 (3/3) | 255.667 (3/3) |
| revised_benzene | mace_e3nn | 67.6667 (3/3) | 129 (3/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_benzene | mace_cueq | 82.6667 (3/3) | 192 (3/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |

## Energy convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.005 | 0.002 | 0.001 | 0.0005 | 0.0002 |
|---|---|---:|---:|---:|---:|---:|
| revised_benzene | ictd_bridge_u_eager | 20 (3/3) | 25.3333 (3/3) | 31.6667 (3/3) | 31.6667 (3/3) | 40 (3/3) |
| revised_benzene | ictd_bridge_u_makefx | 19 (3/3) | 26 (3/3) | 26.3333 (3/3) | 30.6667 (3/3) | 43.6667 (3/3) |
| revised_benzene | ictd_cueq_makefx | 20.6667 (3/3) | 27 (3/3) | 30 (3/3) | 33.6667 (3/3) | 49 (3/3) |
| revised_benzene | mace_e3nn | 55 (3/3) | 58.6667 (3/3) | 63.6667 (3/3) | 63.6667 (3/3) | 109.333 (3/3) |
| revised_benzene | mace_cueq | 66 (3/3) | 88 (3/3) | 100 (3/3) | 110.667 (3/3) | 184.333 (3/3) |
