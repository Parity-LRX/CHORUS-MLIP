# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 25 | 9.62408e-10 | 0.00133934 | 1.39166e+06 | 0.00183141 | 5.25502e-07 | 0.73132 | 1.33934e-06 |
| mace_cueq | 26 | 146048 | 25 | 1.39996e-09 | 0.00179764 | 1.28406e+06 | 0.00241022 | 5.80843e-07 | 0.745839 | 1.79764e-06 |
| ictd_bridge_u | 26 | 146688 | 26 | 7.4428e-12 | 0.0563865 | 7.57598e+09 | 0.0793938 | 9.37454e-11 | 0.710213 | 5.63865e-05 |
| ictd_cueq | 26 | 146688 | 26 | 7.4428e-12 | 0.0563865 | 7.57598e+09 | 0.0793938 | 9.37454e-11 | 0.710213 | 5.63865e-05 |
