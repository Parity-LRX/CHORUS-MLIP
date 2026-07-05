# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 26 | 1.77684e-10 | 0.0096043 | 5.40527e+07 | 0.0197521 | 8.99572e-09 | 0.486243 | 9.6043e-06 |
| mace_cueq | 26 | 146048 | 26 | 2.90382e-10 | 0.0145462 | 5.00933e+07 | 0.0285109 | 1.01849e-08 | 0.510197 | 1.45462e-05 |
| ictd_bridge_u | 26 | 146688 | 26 | 7.27833e-09 | 0.408667 | 5.61485e+07 | 0.944506 | 7.70596e-09 | 0.432678 | 0.000408667 |
| ictd_cueq | 26 | 146688 | 26 | 7.27833e-09 | 0.408667 | 5.61485e+07 | 0.944506 | 7.70596e-09 | 0.432678 | 0.000408667 |
