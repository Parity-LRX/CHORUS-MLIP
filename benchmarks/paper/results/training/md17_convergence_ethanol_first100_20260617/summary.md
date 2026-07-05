# MD17 convergence analysis

All force RMSE values are in eV/A and all energy RMSE values are in eV/atom.
Scalar validation losses are intentionally not compared because MACE-ICTC and mace-torch log different internal loss normalizations.
Partial rows are included so long-running jobs can be monitored before all modes finish.

## Per-run summary

| dataset | mode | seed | status | final epoch | best F | best F epoch | E at best F | best E | best E epoch |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| revised_ethanol | ictd_bridge_u_eager | 20260616 | complete | 99 | 0.024 | 89 | 0.0021 | 0.0006 | 98 |
| revised_ethanol | ictd_bridge_u_eager | 20260617 | complete | 99 | 0.0253 | 96 | 0.0061 | 0.0007 | 66 |
| revised_ethanol | ictd_bridge_u_eager | 20260618 | complete | 99 | 0.0232 | 97 | 0.0014 | 0.0008 | 88 |
| revised_ethanol | ictd_bridge_u_makefx | 20260616 | complete | 99 | 0.0236 | 93 | 0.0027 | 0.0007 | 80 |
| revised_ethanol | ictd_bridge_u_makefx | 20260617 | complete | 99 | 0.0251 | 96 | 0.0006 | 0.0006 | 96 |
| revised_ethanol | ictd_bridge_u_makefx | 20260618 | complete | 99 | 0.0242 | 98 | 0.0021 | 0.0007 | 99 |
| revised_ethanol | ictd_cueq_makefx | 20260616 | complete | 99 | 0.0236 | 93 | 0.0022 | 0.0008 | 95 |
| revised_ethanol | ictd_cueq_makefx | 20260617 | complete | 99 | 0.0245 | 91 | 0.002 | 0.0007 | 79 |
| revised_ethanol | ictd_cueq_makefx | 20260618 | complete | 99 | 0.0253 | 99 | 0.0007 | 0.0007 | 86 |
| revised_ethanol | mace_e3nn | 20260616 | complete | 99 | 0.04443 | 98 | 0.02423 | 0.02158 | 99 |
| revised_ethanol | mace_e3nn | 20260617 | complete | 99 | 0.04381 | 95 | 0.02094 | 0.02005 | 96 |
| revised_ethanol | mace_e3nn | 20260618 | complete | 99 | 0.04431 | 98 | 0.02366 | 0.02239 | 97 |
| revised_ethanol | mace_cueq | 20260616 | complete | 99 | 0.04389 | 96 | 0.01072 | 0.00782 | 99 |
| revised_ethanol | mace_cueq | 20260617 | complete | 99 | 0.04436 | 99 | 0.04176 | 0.02011 | 0 |
| revised_ethanol | mace_cueq | 20260618 | complete | 99 | 0.04406 | 98 | 0.01209 | 0.01133 | 97 |

## Aggregate by mode

| dataset | mode | runs | complete | best F mean | best F std | best E mean | best E std | mean log10 F |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| revised_ethanol | ictd_bridge_u_eager | 3 | 3 | 0.0241667 | 0.00105987 | 0.0007 | 0.0001 | -1.3451 |
| revised_ethanol | ictd_bridge_u_makefx | 3 | 3 | 0.0243 | 0.000754983 | 0.000666667 | 5.7735e-05 | -1.34999 |
| revised_ethanol | ictd_cueq_makefx | 3 | 3 | 0.0244667 | 0.00085049 | 0.000733333 | 5.7735e-05 | -1.34579 |
| revised_ethanol | mace_e3nn | 3 | 3 | 0.0441833 | 0.000328836 | 0.02134 | 0.00118832 | -1.1521 |
| revised_ethanol | mace_cueq | 3 | 3 | 0.0441033 | 0.000237978 | 0.0130867 | 0.00633052 | -1.14677 |

## Force convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.02 | 0.015 | 0.01 | 0.0075 | 0.005 | 0.0035 |
|---|---|---:|---:|---:|---:|---:|---:|
| revised_ethanol | ictd_bridge_u_eager | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | ictd_bridge_u_makefx | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | ictd_cueq_makefx | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | mace_e3nn | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | mace_cueq | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |

## Energy convergence thresholds

Each cell is mean epoch over successful runs; `n/runs` reports how many seeds reached the threshold.

| dataset | mode | 0.005 | 0.002 | 0.001 | 0.0005 | 0.0002 |
|---|---|---:|---:|---:|---:|---:|
| revised_ethanol | ictd_bridge_u_eager | 27 (3/3) | 42 (3/3) | 59.6667 (3/3) | - (0/3) | - (0/3) |
| revised_ethanol | ictd_bridge_u_makefx | 29 (3/3) | 37.6667 (3/3) | 66.3333 (3/3) | - (0/3) | - (0/3) |
| revised_ethanol | ictd_cueq_makefx | 29.6667 (3/3) | 38 (3/3) | 62 (3/3) | - (0/3) | - (0/3) |
| revised_ethanol | mace_e3nn | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
| revised_ethanol | mace_cueq | - (0/3) | - (0/3) | - (0/3) | - (0/3) | - (0/3) |
