# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 1.0287e-09 | 0.00384264 | 3.73545e+06 | 0.00591565 | 1.73894e-07 | 0.649573 | 3.84264e-06 |
| ictd_bridge_u | 26 | 155648 | 26 | 4.64005e-08 | 0.175969 | 3.79239e+06 | 0.273419 | 1.69705e-07 | 0.643587 | 0.000175969 |
| ictd_cueq | 26 | 155648 | 26 | 4.64005e-08 | 0.175969 | 3.79239e+06 | 0.273419 | 1.69705e-07 | 0.643587 | 0.000175969 |
