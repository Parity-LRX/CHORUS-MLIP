# MD17 convergence analysis

All force RMSE values are in eV/A and all energy RMSE values are in eV/atom.
Scalar validation losses are intentionally not compared because MACE-ICTC and mace-torch log different internal loss normalizations.
Partial rows are included so long-running jobs can be monitored before all modes finish.

## Per-run summary

| dataset | mode | seed | status | final epoch | best F | best F epoch | E at best F | best E | best E epoch |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| revised_ethanol | ictd_bridge_u_eager | 20260616 | complete | 299 | 0.0151 | 287 | 0.001 | 0.0003 | 278 |
| revised_ethanol | ictd_bridge_u_eager | 20260617 | complete | 299 | 0.0154 | 282 | 0.0004 | 0.0003 | 247 |
| revised_ethanol | ictd_bridge_u_eager | 20260618 | complete | 299 | 0.0146 | 295 | 0.0004 | 0.0003 | 270 |
| revised_ethanol | ictd_bridge_u_makefx | 20260616 | complete | 299 | 0.0152 | 296 | 0.0014 | 0.0004 | 188 |
| revised_ethanol | ictd_bridge_u_makefx | 20260617 | complete | 299 | 0.0153 | 289 | 0.0003 | 0.0003 | 244 |
| revised_ethanol | ictd_bridge_u_makefx | 20260618 | complete | 299 | 0.0147 | 297 | 0.0003 | 0.0003 | 224 |
| revised_ethanol | ictd_cueq_makefx | 20260616 | complete | 299 | 0.0153 | 296 | 0.0011 | 0.0004 | 188 |
| revised_ethanol | ictd_cueq_makefx | 20260617 | complete | 299 | 0.0153 | 289 | 0.0005 | 0.0003 | 249 |
| revised_ethanol | ictd_cueq_makefx | 20260618 | complete | 299 | 0.0145 | 294 | 0.0022 | 0.0003 | 272 |
| revised_ethanol | mace_e3nn | 20260616 | complete | 299 | 0.02937 | 299 | 0.00107 | 0.00085 | 288 |
| revised_ethanol | mace_e3nn | 20260617 | complete | 299 | 0.02761 | 299 | 0.00355 | 0.00213 | 292 |
| revised_ethanol | mace_e3nn | 20260618 | complete | 299 | 0.03001 | 299 | 0.0047 | 0.0027 | 239 |
| revised_ethanol | mace_cueq | 20260616 | complete | 299 | 0.02924 | 298 | 0.00221 | 0.00095 | 274 |
| revised_ethanol | mace_cueq | 20260617 | complete | 299 | 0.02946 | 295 | 0.00335 | 0.00226 | 296 |
| revised_ethanol | mace_cueq | 20260618 | complete | 299 | 0.0304 | 297 | 0.00253 | 0.00092 | 295 |

## Aggregate by mode

| dataset | mode | runs | complete | best F mean | best F std | best E mean | best E std | mean log10 F |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| revised_ethanol | ictd_bridge_u_eager | 3 | 3 | 0.0150333 | 0.000404145 | 0.0003 | 0 | -1.70125 |
| revised_ethanol | ictd_bridge_u_makefx | 3 | 3 | 0.0150667 | 0.000321455 | 0.000333333 | 5.7735e-05 | -1.7093 |
| revised_ethanol | ictd_cueq_makefx | 3 | 3 | 0.0150333 | 0.00046188 | 0.000333333 | 5.7735e-05 | -1.70956 |
| revised_ethanol | mace_e3nn | 3 | 3 | 0.0289967 | 0.00124279 | 0.00189333 | 0.000947435 | -1.45419 |
| revised_ethanol | mace_cueq | 3 | 3 | 0.0297 | 0.000616117 | 0.00137667 | 0.000765136 | -1.4485 |

## Force convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.02 | 0.015 | 0.01 | 0.0075 | 0.005 | 0.0035 |
|---|---|---:|---:|---:|---:|---:|---:|
| revised_ethanol | ictd_bridge_u_eager | 142 (3/3) | 289 (1/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | ictd_bridge_u_makefx | 152.667 (3/3) | 294 (1/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | ictd_cueq_makefx | 151 (3/3) | 269 (1/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | mace_e3nn | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | mace_cueq | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |

## Energy convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.005 | 0.002 | 0.001 | 0.0005 | 0.0002 |
|---|---|---:|---:|---:|---:|---:|
| revised_ethanol | ictd_bridge_u_eager | 95 (3/3) | 95 (3/3) | 98 (3/3) | 116 (3/3) | - (0/3) |
| revised_ethanol | ictd_bridge_u_makefx | 96.6667 (3/3) | 97 (3/3) | 99.3333 (3/3) | 121.333 (3/3) | - (0/3) |
| revised_ethanol | ictd_cueq_makefx | 95.6667 (3/3) | 96.3333 (3/3) | 98.6667 (3/3) | 137.667 (3/3) | - (0/3) |
| revised_ethanol | mace_e3nn | 185.667 (3/3) | 216 (1/3) | 276 (1/3) | - (0/3) | - (0/3) |
| revised_ethanol | mace_cueq | 160 (3/3) | 174 (2/3) | 269 (2/3) | - (0/3) | - (0/3) |
