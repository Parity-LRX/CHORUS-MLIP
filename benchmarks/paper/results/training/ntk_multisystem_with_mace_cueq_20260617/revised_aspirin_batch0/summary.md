# Empirical NTK Spectrum Diagnostic

Small-batch weighted-output kernel. Force components are deterministically sampled.

| mode | outputs | params | rank | lambda_min_pos | lambda_max | kappa_pos | trace | lambda_min/trace | lambda_max/trace | lr*lambda_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mace_e3nn | 26 | 155008 | 26 | 4.14247e-10 | 0.000725185 | 1.75061e+06 | 0.00194742 | 2.12716e-07 | 0.372383 | 7.25185e-07 |
| mace_cueq | 26 | 155008 | 26 | 3.64117e-10 | 0.00072669 | 1.99576e+06 | 0.00185031 | 1.96787e-07 | 0.39274 | 7.2669e-07 |
| ictd_bridge_u | 26 | 155648 | 26 | 1.83633e-08 | 0.0341765 | 1.86113e+06 | 0.0916697 | 2.0032e-07 | 0.372822 | 3.41765e-05 |
| ictd_cueq | 26 | 155648 | 26 | 1.83633e-08 | 0.0341765 | 1.86113e+06 | 0.0916697 | 2.0032e-07 | 0.372822 | 3.41765e-05 |
