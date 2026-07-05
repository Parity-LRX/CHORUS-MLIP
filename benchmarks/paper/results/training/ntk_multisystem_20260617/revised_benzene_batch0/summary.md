# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 25 | 6.33974e-10 | 0.00136611 | 2.15484e+06 | 0.00181333 | 3.49618e-07 | 0.75337 | 1.36611e-06 |
| ictd_bridge_u | 26 | 146688 | 25 | 2.43108e-08 | 0.0572343 | 2.35428e+06 | 0.0765761 | 3.17472e-07 | 0.747417 | 5.72343e-05 |
| ictd_cueq | 26 | 146688 | 25 | 2.43108e-08 | 0.0572343 | 2.35428e+06 | 0.0765761 | 3.17472e-07 | 0.747417 | 5.72343e-05 |
