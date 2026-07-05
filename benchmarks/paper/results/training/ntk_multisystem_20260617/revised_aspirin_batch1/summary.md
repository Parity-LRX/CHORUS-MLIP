# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 9.94812e-11 | 0.00051177 | 5.14439e+06 | 0.00150367 | 6.61587e-08 | 0.340347 | 5.1177e-07 |
| ictd_bridge_u | 26 | 155648 | 26 | 4.42298e-09 | 0.0243007 | 5.49419e+06 | 0.0689648 | 6.41339e-08 | 0.352364 | 2.43007e-05 |
| ictd_cueq | 26 | 155648 | 26 | 4.42298e-09 | 0.0243007 | 5.49419e+06 | 0.0689648 | 6.41339e-08 | 0.352364 | 2.43007e-05 |
