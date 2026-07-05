# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 1.95563e-09 | 0.00491135 | 2.51139e+06 | 0.00826342 | 2.36661e-07 | 0.594348 | 4.91135e-06 |
| mace_cueq | 26 | 155008 | 26 | 1.80958e-09 | 0.00517071 | 2.85741e+06 | 0.00808124 | 2.23923e-07 | 0.639841 | 5.17071e-06 |
| ictd_bridge_u | 26 | 155648 | 26 | 8.30803e-08 | 0.231111 | 2.78178e+06 | 0.387412 | 2.1445e-07 | 0.596551 | 0.000231111 |
| ictd_cueq | 26 | 155648 | 26 | 8.30803e-08 | 0.231111 | 2.78178e+06 | 0.387412 | 2.1445e-07 | 0.596551 | 0.000231111 |
