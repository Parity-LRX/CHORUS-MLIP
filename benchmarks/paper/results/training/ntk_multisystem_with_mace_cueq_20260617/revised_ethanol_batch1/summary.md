# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 9.82536e-10 | 0.00231544 | 2.3566e+06 | 0.00506365 | 1.94037e-07 | 0.457267 | 2.31544e-06 |
| mace_cueq | 26 | 155008 | 26 | 8.45143e-10 | 0.00246053 | 2.91138e+06 | 0.00483656 | 1.74741e-07 | 0.508736 | 2.46053e-06 |
| ictd_bridge_u | 26 | 155648 | 26 | 4.39279e-08 | 0.107475 | 2.44662e+06 | 0.237321 | 1.85099e-07 | 0.452867 | 0.000107475 |
| ictd_cueq | 26 | 155648 | 26 | 4.39279e-08 | 0.107475 | 2.44662e+06 | 0.237321 | 1.85099e-07 | 0.452867 | 0.000107475 |
