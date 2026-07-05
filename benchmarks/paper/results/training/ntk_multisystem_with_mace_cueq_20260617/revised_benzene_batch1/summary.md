# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 146048 | 25 | 2.97433e-10 | 0.00155673 | 5.2339e+06 | 0.00214553 | 1.38629e-07 | 0.725571 | 1.55673e-06 |
| mace_cueq | 26 | 146048 | 25 | 4.64451e-10 | 0.00218545 | 4.70544e+06 | 0.00289547 | 1.60406e-07 | 0.754782 | 2.18545e-06 |
| ictd_bridge_u | 26 | 146688 | 26 | 1.29899e-11 | 0.0649759 | 5.00202e+09 | 0.092417 | 1.40558e-10 | 0.703074 | 6.49759e-05 |
| ictd_cueq | 26 | 146688 | 26 | 1.29899e-11 | 0.0649759 | 5.00202e+09 | 0.092417 | 1.40558e-10 | 0.703074 | 6.49759e-05 |
