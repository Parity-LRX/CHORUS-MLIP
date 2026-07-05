# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 25 | 5.87733e-10 | 0.00126352 | 2.14982e+06 | 0.00175817 | 3.34287e-07 | 0.718658 | 1.26352e-06 |
| ictd_bridge_u | 26 | 146688 | 26 | 7.71331e-12 | 0.0538023 | 6.97525e+09 | 0.0748317 | 1.03075e-10 | 0.718977 | 5.38023e-05 |
| ictd_cueq | 26 | 146688 | 26 | 7.71331e-12 | 0.0538023 | 6.97525e+09 | 0.0748317 | 1.03075e-10 | 0.718977 | 5.38023e-05 |
