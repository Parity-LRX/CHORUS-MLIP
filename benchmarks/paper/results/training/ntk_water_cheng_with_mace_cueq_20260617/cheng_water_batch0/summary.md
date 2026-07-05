# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 26 | 6.76042e-11 | 0.0108345 | 1.60264e+08 | 0.0186862 | 3.61787e-09 | 0.579815 | 1.08345e-05 |
| mace_cueq | 26 | 146048 | 26 | 9.01373e-11 | 0.0163207 | 1.81064e+08 | 0.0267214 | 3.37322e-09 | 0.610771 | 1.63207e-05 |
| ictd_bridge_u | 26 | 146688 | 26 | 2.7553e-09 | 0.458993 | 1.66585e+08 | 0.896892 | 3.07206e-09 | 0.51176 | 0.000458993 |
| ictd_cueq | 26 | 146688 | 26 | 2.7553e-09 | 0.458993 | 1.66585e+08 | 0.896892 | 3.07206e-09 | 0.51176 | 0.000458993 |
