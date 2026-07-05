# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 2.04811e-10 | 0.00050878 | 2.48415e+06 | 0.00139255 | 1.47076e-07 | 0.36536 | 5.0878e-07 |
| ictd_bridge_u | 26 | 155648 | 26 | 9.27519e-09 | 0.0236567 | 2.55053e+06 | 0.064522 | 1.43752e-07 | 0.366645 | 2.36567e-05 |
| ictd_cueq | 26 | 155648 | 26 | 9.27519e-09 | 0.0236567 | 2.55053e+06 | 0.064522 | 1.43752e-07 | 0.366645 | 2.36567e-05 |
