# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 2.05789e-10 | 0.00309527 | 1.5041e+07 | 0.00548757 | 3.75009e-08 | 0.56405 | 3.09527e-06 |
| ictd_bridge_u | 26 | 155648 | 26 | 9.27275e-09 | 0.140983 | 1.5204e+07 | 0.252912 | 3.66639e-08 | 0.557438 | 0.000140983 |
| ictd_cueq | 26 | 155648 | 26 | 9.27275e-09 | 0.140983 | 1.5204e+07 | 0.252912 | 3.66639e-08 | 0.557438 | 0.000140983 |
