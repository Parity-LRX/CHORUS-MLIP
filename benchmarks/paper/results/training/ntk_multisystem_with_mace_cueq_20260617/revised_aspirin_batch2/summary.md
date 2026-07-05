# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 2.76065e-10 | 0.000985604 | 3.57019e+06 | 0.00252886 | 1.09166e-07 | 0.389743 | 9.85604e-07 |
| mace_cueq | 26 | 155008 | 26 | 2.8783e-10 | 0.00098748 | 3.43077e+06 | 0.00238125 | 1.20873e-07 | 0.414689 | 9.8748e-07 |
| ictd_bridge_u | 26 | 155648 | 26 | 1.20177e-08 | 0.0471409 | 3.92262e+06 | 0.120149 | 1.00024e-07 | 0.392355 | 4.71409e-05 |
| ictd_cueq | 26 | 155648 | 26 | 1.20177e-08 | 0.0471409 | 3.92262e+06 | 0.120149 | 1.00024e-07 | 0.392355 | 4.71409e-05 |
