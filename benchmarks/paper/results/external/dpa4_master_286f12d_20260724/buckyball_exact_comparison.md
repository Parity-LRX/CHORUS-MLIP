# Exact Buckyball comparison: CHORUS and DPA-4

Single-seed results on the common MD22 Buckyball-catcher split: 600 training,
600 validation, and 4,902 test structures. All models use a 5 Å cutoff,
batch size 4, 45k optimizer steps, Energy:Force = 1:100, float32, no EMA,
and no TF32. DPA-4 and CHORUS retain their native learning-rate schedules.

## Full test set

Energy errors are in meV/atom; force errors are in meV/Å.

| Model | Parameters | E MAE | E RMSE | F MAE | F RMSE |
|---|---:|---:|---:|---:|---:|
| DPA-4 large, 43k | 650,348 | **0.219** | **0.274** | **5.582** | 7.843 |
| DPA-4 large, 45k | 650,348 | 0.251 | 0.298 | 5.581 | **7.841** |
| CHORUS small | 226,310 | 0.241 | 0.298 | 8.850 | 11.981 |
| CHORUS large | 672,134 | 0.233 | 0.288 | 6.724 | 9.089 |

## Full validation set

| Model | Parameters | E MAE | E RMSE | F MAE | F RMSE |
|---|---:|---:|---:|---:|---:|
| DPA-4 large, 43k | 650,348 | **0.217** | **0.270** | 5.671 | 7.970 |
| DPA-4 large, 45k | 650,348 | 0.249 | 0.297 | **5.668** | **7.966** |
| CHORUS small | 226,310 | 0.256 | 0.312 | 8.891 | 12.024 |
| CHORUS large | 672,134 | 0.241 | 0.293 | 6.740 | 9.152 |

## Interpretation

- Against the final 45k DPA-4 checkpoint, CHORUS large has 7.2% lower test
  energy MAE and 3.3% lower test energy RMSE, but 20.5% higher force MAE
  and 15.9% higher force RMSE.
- The 43k DPA-4 checkpoint is the stronger energy–force Pareto point: it
  improves both energy and force test errors over CHORUS large.
- CHORUS small reaches essentially the same test energy RMSE as final DPA-4
  with about 35% of its parameters, but its force error is substantially
  higher.
- These are single-seed results. They support competitiveness and energy
  parameter efficiency, not a claim that CHORUS uniformly outperforms DPA-4.
- Online training envelopes are not used in this table. All reported values
  come from full offline evaluation of one saved checkpoint at a time.

Source logs are under
`/home/ylzhang/chorus_runs/buckyball_fair_r5_noema_20260724/`.
