# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 1.74315e-10 | 0.000646238 | 3.70729e+06 | 0.0019894 | 8.76221e-08 | 0.324841 | 6.46238e-07 |
| mace_cueq | 26 | 155008 | 26 | 1.79242e-10 | 0.000642545 | 3.5848e+06 | 0.00186754 | 9.59777e-08 | 0.34406 | 6.42545e-07 |
| ictd_bridge_u | 26 | 155648 | 26 | 7.69474e-09 | 0.0305471 | 3.96987e+06 | 0.0943028 | 8.15961e-08 | 0.323926 | 3.05471e-05 |
| ictd_cueq | 26 | 155648 | 26 | 7.69474e-09 | 0.0305471 | 3.96987e+06 | 0.0943028 | 8.15961e-08 | 0.323926 | 3.05471e-05 |
