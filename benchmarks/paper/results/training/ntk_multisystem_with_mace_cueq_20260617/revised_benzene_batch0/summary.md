# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 25 | 6.40159e-10 | 0.00138925 | 2.17017e+06 | 0.00184808 | 3.46392e-07 | 0.75173 | 1.38925e-06 |
| mace_cueq | 26 | 146048 | 25 | 1.04055e-09 | 0.00171404 | 1.64725e+06 | 0.00231184 | 4.50094e-07 | 0.741416 | 1.71404e-06 |
| ictd_bridge_u | 26 | 146688 | 25 | 2.45531e-08 | 0.0593876 | 2.41874e+06 | 0.0807098 | 3.04215e-07 | 0.735816 | 5.93876e-05 |
| ictd_cueq | 26 | 146688 | 25 | 2.45531e-08 | 0.0593876 | 2.41874e+06 | 0.0807098 | 3.04215e-07 | 0.735816 | 5.93876e-05 |
