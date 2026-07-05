# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 6.39279e-10 | 0.00186522 | 2.9177e+06 | 0.00361705 | 1.76741e-07 | 0.515675 | 1.86522e-06 |
| ictd_bridge_u | 26 | 155648 | 26 | 2.84364e-08 | 0.0854031 | 3.0033e+06 | 0.167894 | 1.69371e-07 | 0.508673 | 8.54031e-05 |
| ictd_cueq | 26 | 155648 | 26 | 2.84364e-08 | 0.0854031 | 3.0033e+06 | 0.167894 | 1.69371e-07 | 0.508673 | 8.54031e-05 |
