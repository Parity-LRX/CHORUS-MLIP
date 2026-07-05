# MD17 convergence analysis

All force RMSE values are in eV/A and all energy RMSE values are in eV/atom.
Scalar validation losses are intentionally not compared because MACE-ICTC and mace-torch log different internal loss normalizations.
Partial rows are included so long-running jobs can be monitored before all modes finish.

## Per-run summary

| dataset | mode | seed | status | final epoch | best F | best F epoch | E at best F | best E | best E epoch |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| revised_aspirin | ictd_bridge_u_eager | 20260616 | complete | 299 | 0.0267 | 291 | 0.0021 | 0.0005 | 241 |
| revised_aspirin | ictd_bridge_u_eager | 20260617 | complete | 299 | 0.0266 | 293 | 0.0036 | 0.0005 | 215 |
| revised_aspirin | ictd_bridge_u_eager | 20260618 | complete | 299 | 0.0265 | 295 | 0.0013 | 0.0005 | 217 |
| revised_aspirin | ictd_bridge_u_makefx | 20260616 | complete | 299 | 0.0274 | 292 | 0.0006 | 0.0005 | 229 |
| revised_aspirin | ictd_bridge_u_makefx | 20260617 | complete | 299 | 0.0266 | 292 | 0.0007 | 0.0005 | 214 |
| revised_aspirin | ictd_bridge_u_makefx | 20260618 | complete | 299 | 0.0274 | 284 | 0.0024 | 0.0005 | 258 |
| revised_aspirin | ictd_cueq_makefx | 20260616 | complete | 299 | 0.0276 | 289 | 0.0022 | 0.0005 | 261 |
| revised_aspirin | ictd_cueq_makefx | 20260617 | complete | 299 | 0.0268 | 292 | 0.0009 | 0.0005 | 246 |
| revised_aspirin | ictd_cueq_makefx | 20260618 | complete | 299 | 0.0276 | 296 | 0.0028 | 0.0005 | 274 |
| revised_aspirin | mace_e3nn | 20260616 | complete | 299 | 0.05141 | 294 | 0.00228 | 0.00145 | 283 |
| revised_aspirin | mace_e3nn | 20260617 | complete | 299 | 0.04796 | 295 | 0.00131 | 0.00119 | 272 |
| revised_aspirin | mace_e3nn | 20260618 | complete | 299 | 0.05056 | 290 | 0.00551 | 0.00158 | 283 |
| revised_aspirin | mace_cueq | 20260616 | complete | 299 | 0.04749 | 298 | 0.00263 | 0.00104 | 296 |
| revised_aspirin | mace_cueq | 20260617 | complete | 299 | 0.04832 | 292 | 0.00487 | 0.00117 | 288 |
| revised_aspirin | mace_cueq | 20260618 | complete | 299 | 0.05014 | 287 | 0.00642 | 0.00257 | 253 |

## Aggregate by mode

| dataset | mode | runs | complete | best F mean | best F std | best E mean | best E std | mean log10 F |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| revised_aspirin | ictd_bridge_u_eager | 3 | 3 | 0.0266 | 0.0001 | 0.0005 | 0 | -1.3686 |
| revised_aspirin | ictd_bridge_u_makefx | 3 | 3 | 0.0271333 | 0.00046188 | 0.0005 | 0 | -1.36684 |
| revised_aspirin | ictd_cueq_makefx | 3 | 3 | 0.0273333 | 0.00046188 | 0.0005 | 0 | -1.36504 |
| revised_aspirin | mace_e3nn | 3 | 3 | 0.0499767 | 0.00179745 | 0.00140667 | 0.000198578 | -1.13504 |
| revised_aspirin | mace_cueq | 3 | 3 | 0.04865 | 0.00135547 | 0.00159333 | 0.000848312 | -1.1517 |

## Force convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.02 | 0.015 | 0.01 | 0.0075 | 0.005 | 0.0035 |
|---|---|---:|---:|---:|---:|---:|---:|
| revised_aspirin | ictd_bridge_u_eager | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_aspirin | ictd_bridge_u_makefx | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_aspirin | ictd_cueq_makefx | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_aspirin | mace_e3nn | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_aspirin | mace_cueq | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |

## Energy convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.005 | 0.002 | 0.001 | 0.0005 | 0.0002 |
|---|---|---:|---:|---:|---:|---:|
| revised_aspirin | ictd_bridge_u_eager | 62.6667 (3/3) | 86.3333 (3/3) | 101 (3/3) | 224.333 (3/3) | - (0/3) |
| revised_aspirin | ictd_bridge_u_makefx | 55.6667 (3/3) | 87 (3/3) | 118 (3/3) | 233.667 (3/3) | - (0/3) |
| revised_aspirin | ictd_cueq_makefx | 60.6667 (3/3) | 68.6667 (3/3) | 103.333 (3/3) | 260.333 (3/3) | - (0/3) |
| revised_aspirin | mace_e3nn | 143.667 (3/3) | 178.667 (3/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_aspirin | mace_cueq | 183.333 (3/3) | 212 (2/3) | - (0/3) | - (0/3) | - (0/3) |
