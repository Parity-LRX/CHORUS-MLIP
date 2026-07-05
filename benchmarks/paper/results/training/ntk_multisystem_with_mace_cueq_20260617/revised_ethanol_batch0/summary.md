# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 2.74482e-10 | 0.00353175 | 1.28669e+07 | 0.00692207 | 3.96532e-08 | 0.510215 | 3.53175e-06 |
| mace_cueq | 26 | 155008 | 26 | 2.80389e-10 | 0.00397313 | 1.41701e+07 | 0.00706895 | 3.96648e-08 | 0.562054 | 3.97313e-06 |
| ictd_bridge_u | 26 | 155648 | 26 | 1.19822e-08 | 0.163204 | 1.36206e+07 | 0.322425 | 3.71628e-08 | 0.506178 | 0.000163204 |
| ictd_cueq | 26 | 155648 | 26 | 1.19822e-08 | 0.163204 | 1.36206e+07 | 0.322425 | 3.71628e-08 | 0.506178 | 0.000163204 |
