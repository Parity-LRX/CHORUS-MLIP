# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 26 | 8.58026e-10 | 0.00684212 | 7.97426e+06 | 0.0151114 | 5.67801e-08 | 0.452779 | 6.84212e-06 |
| mace_cueq | 26 | 146048 | 26 | 1.29509e-09 | 0.010007 | 7.72688e+06 | 0.0212718 | 6.08829e-08 | 0.470435 | 1.0007e-05 |
| ictd_bridge_u | 26 | 146688 | 26 | 3.61016e-08 | 0.294941 | 8.16976e+06 | 0.749949 | 4.81387e-08 | 0.393282 | 0.000294941 |
| ictd_cueq | 26 | 146688 | 26 | 3.61016e-08 | 0.294941 | 8.16976e+06 | 0.749949 | 4.81387e-08 | 0.393282 | 0.000294941 |
