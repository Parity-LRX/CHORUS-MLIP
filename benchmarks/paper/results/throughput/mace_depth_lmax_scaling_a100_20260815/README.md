# MACE-ICTC depth and angular-resolution cost controls

Strict-FP32 MakeFX benchmark on one NVIDIA A100 40 GB. All models use C128,
correlation order 3, a fixed directed degree of 32, and the same synthetic
graph protocol. Inference includes energy and conservative forces; training is
a complete energy--force optimiser update including force double backward.
Compilation and graph preparation are excluded from steady-state latency.

## Large-graph endpoints

| Configuration | Parameters | Inference, 2048 atoms (katom/s) | Inference memory (GiB) | Training, 2048 atoms (katom/s) | Training memory (GiB) | Inference, 4096 atoms (katom/s) |
|---|---:|---:|---:|---:|---:|---:|
| L2 Off | 652,736 | 31.890 | 7.743 | 12.267 | 14.766 | 31.880 |
| L2 CHORUS-Final R16 | 792,198 | 25.399 | 8.856 | 8.831 | 20.617 | 25.534 |
| L2 CHORUS-Persistent R16 | 1,029,967 | 21.576 | 9.363 | 8.229 | 22.755 | 21.708 |
| Lmax3 Off, two interactions | 1,029,696 | 10.870 | 22.685 | OOM | OOM | OOM |
| L2 Off, three interactions | 1,242,048 | 16.249 | 11.812 | 6.310 | 25.701 | 16.241 |

At essentially identical parameter count, L2 CHORUS-Persistent provides
1.98x the 2048-atom inference throughput of Lmax3 Off and uses 58.7% less
inference memory. Relative to the three-interaction Off control,
CHORUS-Persistent uses 17.1% fewer parameters, is 32.8% faster in inference
and 30.4% faster in training, and uses 20.7%/11.5% less inference/training
memory at 2048 atoms. CHORUS-Final is 56.3% faster in inference and 40.0%
faster in training than the three-interaction control while using 36.2% fewer
parameters.

The Lmax3 control reaches OOM at 4096-atom inference and 2048-atom training;
these outcomes are retained in the raw JSON files. Accuracy comparisons remain
separate: these measurements establish implementation cost and do not imply
that either CHORUS scope dominates the higher-angular or deeper controls on
every dataset.

Slurm job: `55784` (completed in 00:37:48, exit code 0).
