# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 25 | 3.68927e-10 | 0.00139142 | 3.77153e+06 | 0.00195765 | 1.88455e-07 | 0.710762 | 1.39142e-06 |
| ictd_bridge_u | 26 | 146688 | 26 | 1.2189e-11 | 0.0597498 | 4.90195e+09 | 0.0835881 | 1.45822e-10 | 0.714813 | 5.97498e-05 |
| ictd_cueq | 26 | 146688 | 26 | 1.2189e-11 | 0.0597498 | 4.90195e+09 | 0.0835881 | 1.45822e-10 | 0.714813 | 5.97498e-05 |
