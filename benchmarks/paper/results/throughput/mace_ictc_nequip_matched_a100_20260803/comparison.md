# Matched A100 throughput comparison

Protocol: NVIDIA A100-SXM4-40GB; strict FP32 with TF32 disabled; first 16
xxMD-MAL validation structures; 144 atoms and 1,152 directed edges per batch;
20 steady-state iterations; three repeats; median reported. Inference evaluates
energy and conservative forces. Training evaluates the energy-force loss and its
double backward, without an optimizer update. Compile time is excluded.

| Backbone and mode | Parameters | Inference (ms) | Inference (atom/s) | Training (ms) | Training (atom/s) |
|---|---:|---:|---:|---:|---:|
| MACE-ICTC, phase-off | 652,736 | 4.023 | 35,796 | 9.309 | 15,468 |
| MACE-ICTC, CHORUS-Final R16 | 792,198 | 5.031 | 28,624 | 12.739 | 11,304 |
| MACE-ICTC, CHORUS-Persistent R16 | 947,407 | 6.334 | 22,734 | 20.228 | 7,119 |
| NequIP-ICTC MakeFX, phase-off | 614,510 | 3.121 | 46,141 | 6.455 | 22,308 |
| NequIP-ICTC MakeFX, CHORUS-Final R16 | 811,091 | 5.188 | 27,758 | 11.035 | 13,050 |
| NequIP-ICTC MakeFX, CHORUS-Persistent R16 | 866,747 | 7.254 | 19,850 | 15.874 | 9,072 |
| NequIP-OEq, phase-off | 613,334 | 7.398 | 19,464 | 18.671 | 7,713 |
| NequIP-OEq, CHORUS-Final R16 | 809,915 | 10.442 | 13,791 | 26.353 | 5,464 |
| NequIP-OEq, CHORUS-Persistent R16 | 865,571 | 13.963 | 10,313 | 35.903 | 4,011 |

Relative to the matched NequIP-OEq mode, MACE-ICTC is 1.84x/2.01x faster for
phase-off inference/training, 2.08x/2.07x faster for CHORUS-Final, and
2.20x/1.77x faster for CHORUS-Persistent.

Within MACE-ICTC, CHORUS-Final adds 25.1% inference latency and 36.8% training
latency over phase-off. CHORUS-Persistent adds 57.5% and 117.3%, respectively.
The compile time and concurrent-run wall time are not included.

NequIP-ICTC eager contains no fused equivariant operator. Whole-model MakeFX
compilation accelerates eager NequIP-ICTC by 3.53x/4.17x for phase-off
inference/training, 3.99x/4.27x for CHORUS-Final, and 4.40x/4.55x for
CHORUS-Persistent. OpenEquivariance is an operator-level backend; the OEq rows
do not include whole-model MakeFX because e3nn's scripted spherical harmonics
currently block FakeTensor tracing. Therefore the OEq and ICTC rows should not
be used alone to attribute the throughput difference to the mathematical basis.
