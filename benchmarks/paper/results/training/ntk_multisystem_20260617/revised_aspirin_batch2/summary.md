# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 1.79572e-10 | 0.00071552 | 3.98459e+06 | 0.00183992 | 9.75974e-08 | 0.388886 | 7.1552e-07 |
| ictd_bridge_u | 26 | 155648 | 26 | 8.49408e-09 | 0.0342304 | 4.02991e+06 | 0.0845819 | 1.00424e-07 | 0.404701 | 3.42304e-05 |
| ictd_cueq | 26 | 155648 | 26 | 8.49408e-09 | 0.0342304 | 4.02991e+06 | 0.0845819 | 1.00424e-07 | 0.404701 | 3.42304e-05 |
