# MD17 convergence analysis

All force RMSE values are in eV/A and all energy RMSE values are in eV/atom.
Scalar validation losses are intentionally not compared because MACE-ICTC and mace-torch log different internal loss normalizations.
Partial rows are included so long-running jobs can be monitored before all modes finish.

## Per-run summary

| dataset | mode | seed | status | final epoch | best F | best F epoch | E at best F | best E | best E epoch |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| cheng_water | ictd_bridge_u_eager | 20260616 | complete | 299 | 0.0457 | 277 | 0.0038 | 0.0013 | 205 |
| cheng_water | ictd_bridge_u_eager | 20260617 | complete | 299 | 0.0453 | 290 | 0.003 | 0.0017 | 221 |
| cheng_water | ictd_bridge_u_eager | 20260618 | complete | 299 | 0.047 | 277 | 0.0099 | 0.0015 | 249 |
| cheng_water | ictd_bridge_u_makefx | 20260616 | complete | 299 | 0.0466 | 299 | 0.0018 | 0.0016 | 127 |
| cheng_water | ictd_bridge_u_makefx | 20260617 | complete | 299 | 0.0467 | 292 | 0.0044 | 0.0017 | 194 |
| cheng_water | ictd_bridge_u_makefx | 20260618 | complete | 299 | 0.0458 | 291 | 0.0033 | 0.0015 | 187 |
| cheng_water | ictd_cueq_makefx | 20260616 | complete | 299 | 0.0473 | 290 | 0.0038 | 0.0015 | 255 |
| cheng_water | ictd_cueq_makefx | 20260617 | complete | 299 | 0.0467 | 267 | 0.0048 | 0.0017 | 160 |
| cheng_water | ictd_cueq_makefx | 20260618 | complete | 299 | 0.0465 | 285 | 0.0074 | 0.0016 | 224 |
| cheng_water | mace_e3nn | 20260616 | complete | 299 | 0.06164 | 299 | 0.00514 | 0.00241 | 271 |
| cheng_water | mace_e3nn | 20260617 | complete | 299 | 0.06501 | 294 | 0.00294 | 0.00274 | 296 |
| cheng_water | mace_e3nn | 20260618 | complete | 299 | 0.06118 | 296 | 0.00315 | 0.00247 | 251 |
| cheng_water | mace_cueq | 20260616 | complete | 299 | 0.06592 | 298 | 0.00485 | 0.00277 | 282 |
| cheng_water | mace_cueq | 20260617 | complete | 299 | 0.0647 | 298 | 0.00941 | 0.00272 | 263 |
| cheng_water | mace_cueq | 20260618 | complete | 299 | 0.06193 | 298 | 0.00446 | 0.00246 | 293 |

## Aggregate by mode

| dataset | mode | runs | complete | best F mean | best F std | best E mean | best E std | mean log10 F |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cheng_water | ictd_bridge_u_eager | 3 | 3 | 0.046 | 0.000888819 | 0.0015 | 0.0002 | -1.19331 |
| cheng_water | ictd_bridge_u_makefx | 3 | 3 | 0.0463667 | 0.000493288 | 0.0016 | 0.0001 | -1.18946 |
| cheng_water | ictd_cueq_makefx | 3 | 3 | 0.0468333 | 0.000416333 | 0.0016 | 0.0001 | -1.18687 |
| cheng_water | mace_e3nn | 3 | 3 | 0.06261 | 0.00209115 | 0.00254 | 0.000175784 | -1.07772 |
| cheng_water | mace_cueq | 3 | 3 | 0.0641833 | 0.00204456 | 0.00265 | 0.000166433 | -1.06618 |

## Force convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.02 | 0.015 | 0.01 | 0.0075 | 0.005 | 0.0035 |
|---|---|---:|---:|---:|---:|---:|---:|
| cheng_water | ictd_bridge_u_eager | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| cheng_water | ictd_bridge_u_makefx | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| cheng_water | ictd_cueq_makefx | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| cheng_water | mace_e3nn | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| cheng_water | mace_cueq | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |

## Energy convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.005 | 0.002 | 0.001 | 0.0005 | 0.0002 |
|---|---|---:|---:|---:|---:|---:|
| cheng_water | ictd_bridge_u_eager | 16.6667 (3/3) | 77.6667 (3/3) | - (0/3) | - (0/3) | - (0/3) |
| cheng_water | ictd_bridge_u_makefx | 21 (3/3) | 100.333 (3/3) | - (0/3) | - (0/3) | - (0/3) |
| cheng_water | ictd_cueq_makefx | 21.6667 (3/3) | 108.333 (3/3) | - (0/3) | - (0/3) | - (0/3) |
| cheng_water | mace_e3nn | 64.3333 (3/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| cheng_water | mace_cueq | 67.3333 (3/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
