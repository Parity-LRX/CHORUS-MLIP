# L=3 STI and 3BPA checkpoint diagnostics

These scans are diagnostic only. Checkpoints were sampled at predefined
optimizer steps, and the formally reported model remains the checkpoint chosen
by validation Force MAE. Test errors below must not be used to replace that
selection.

Configuration: C128, `lmax=max_ell=3`, two interactions, correlation 3,
CHORUS rank 16, strict FP32, no EMA or TF32.

## STI test trajectory

Energy MAE is in meV/atom; Force MAE and RMSE are in meV/Angstrom. `*` marks
the formal validation-selected checkpoint.

| Mode | Step | Energy MAE | Force MAE | Force RMSE |
|:--|--:|--:|--:|--:|
| Phase-off | 800 | 42.287 | 188.323 | 495.923 |
| Phase-off | 2,400 | 28.066 | 171.198 | 487.691 |
| Phase-off | 4,800 | 25.851 | 159.336 | 497.154 |
| Phase-off | 7,200 | 21.071 | 153.372 | 485.843 |
| Phase-off | 9,600 | 20.118 | 168.521 | 537.411 |
| Phase-off | 12,000 | 16.138 | 150.983 | 499.072 |
| Phase-off | 16,000 | 17.975 | **146.664** | 499.185 |
| Phase-off | 23,200* | 15.985 | 148.233 | 509.927 |
| Phase-off | 24,000 | 26.129 | 149.331 | 511.274 |
| CHORUS-Final | 800 | 31.379 | 187.147 | 501.517 |
| CHORUS-Final | 2,400 | 16.238 | 184.821 | 495.152 |
| CHORUS-Final | 4,800 | 26.547 | 188.814 | 812.951 |
| CHORUS-Final | 7,200 | 25.872 | 300.119 | 1,639.069 |
| CHORUS-Final | 9,600* | 22.800 | **169.149** | 575.266 |
| CHORUS-Final | 12,000 | 34.504 | 292.252 | 1,567.891 |
| CHORUS-Final | 16,000 | 25.174 | 463.979 | 2,578.491 |
| CHORUS-Final | 24,000 | 68.773 | 1,211.108 | 7,309.437 |
| CHORUS-Persistent | 800 | 21.698 | 179.749 | 480.834 |
| CHORUS-Persistent | 2,400 | 52.954 | 165.700 | 504.807 |
| CHORUS-Persistent | 4,800 | 27.766 | 165.457 | 509.329 |
| CHORUS-Persistent | 7,200* | 19.141 | **157.576** | 502.555 |
| CHORUS-Persistent | 9,600 | 31.063 | 176.838 | 666.691 |
| CHORUS-Persistent | 12,000 | 15.248 | 416.496 | 2,510.915 |
| CHORUS-Persistent | 16,000 | 15.150 | 247.791 | 1,169.189 |
| CHORUS-Persistent | 24,000 | 23.273 | 604.065 | 4,215.640 |

The Phase-off test minimum at step 16,000 is only 1.1% below its formal
validation-selected result. Final and Persistent both become unstable after
their selected checkpoints, but no sampled test checkpoint beats the formal
validation-selected checkpoint. The poor STI result is therefore not caused
by choosing the wrong saved checkpoint.

## 3BPA test Force-MAE trajectory

All values are Force MAE in meV/Angstrom. `*` marks the formal checkpoint
chosen by the 300 K validation Force MAE.

| Mode | Step | 300 K | 600 K | 1200 K |
|:--|--:|--:|--:|--:|
| Phase-off | 2,800 | 16.735 | 33.541 | 104.880 |
| Phase-off | 5,600 | 14.134 | 28.375 | 94.614 |
| Phase-off | 11,200 | 9.463 | 22.386 | 86.317 |
| Phase-off | 19,600 | 8.317 | 20.613 | 82.758 |
| Phase-off | 28,000 | 7.914 | 19.963 | 81.128 |
| Phase-off | 35,000 | 7.814 | 19.798 | 80.755 |
| Phase-off | 39,816* | **7.790** | **19.726** | **80.559** |
| Phase-off | 40,012 | 7.791 | 19.730 | 80.560 |
| CHORUS-Final | 2,800 | 16.866 | 32.609 | 136.925 |
| CHORUS-Final | 5,600 | 10.857 | 25.693 | 131.792 |
| CHORUS-Final | 11,200 | 9.087 | 23.647 | 132.935 |
| CHORUS-Final | 19,600 | 8.145 | 22.313 | 129.291 |
| CHORUS-Final | 28,000 | 7.935 | 21.990 | 128.224 |
| CHORUS-Final | 35,000 | 7.873 | 21.854 | 127.674 |
| CHORUS-Final | 39,144* | 7.860 | **21.792** | 127.425 |
| CHORUS-Final | 40,012 | **7.857** | 21.793 | **127.411** |
| CHORUS-Persistent | 2,800 | 16.483 | 34.105 | 166.572 |
| CHORUS-Persistent | 5,600 | 11.822 | 27.041 | 160.620 |
| CHORUS-Persistent | 11,200 | 8.701 | 24.117 | 161.627 |
| CHORUS-Persistent | 19,600 | 7.661 | 22.874 | 158.856 |
| CHORUS-Persistent | 28,000 | 7.349 | 22.349 | 156.479 |
| CHORUS-Persistent | 35,000 | 7.259 | 22.296 | 156.165 |
| CHORUS-Persistent | 40,012 | 7.235 | **22.229** | 155.861 |
| CHORUS-Persistent | 40,376* | **7.234** | 22.230 | **155.853** |

All three modes improve or plateau through the end of training. In particular,
the high-temperature error does not pass through an earlier, substantially
better test minimum. The 600 K and 1200 K gap is an extrapolation failure of
the fitted representation under a 300 K training/validation protocol, not a
checkpoint-selection failure.

## Interpretation

STI and 3BPA expose different failure modes. STI shows optimization and
generalisation instability for the L=3 CHORUS branches, while validation early
stopping already avoids the worst region. On 3BPA, the CHORUS branches fit the
300 K distribution progressively better but do not transfer that improvement
to 1200 K. A legitimate attempt to improve 3BPA extrapolation therefore needs
a training-only robustness intervention or a predeclared high-temperature
validation proxy; selecting a checkpoint on the reported test temperatures
would leak test information.
