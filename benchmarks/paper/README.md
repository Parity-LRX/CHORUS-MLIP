# CHORUS benchmark artifacts

This directory is the curated computational record for CHORUS.  It contains
lightweight scripts, machine-readable summaries, logs, and vector figures.
Model checkpoints, datasets, and temporary build products are intentionally not
stored in Git.

The archive is organized by the scientific question being tested:

| Directory | Contents |
|---|---|
| `results/phase/` | CHORUS mechanism controls, rank/scope studies, MACE-ICTC experiments, and NequIP transfer summaries |
| `results/external/` | matched evaluations of DPA-4, TECE, native MACE, and other external model families |
| `results/throughput/` | strict-precision training and inference scaling records |
| `scripts/training/` | data preparation, training queues, checkpoint selection, calibration, and evaluation |
| `scripts/evaluation/` | shared metric and selected-checkpoint evaluation helpers |
| `scripts/throughput/` | cross-model timing and plotting tools |
| `external/` | external-model configuration files used by the benchmark |
| `figures/` | manuscript-facing vector figures generated from the archived data |

The previous MACE-ICTC operator, NTK, OFF23, long-range, and legacy MD17
benchmark archive is deliberately absent.  Those results belong to the
MACE-ICTC project and are not evidence for the independent CHORUS repository.

## Evidence policy

- Validation metrics select checkpoints; test data never select a checkpoint.
- Energy and force validation envelopes may be reported independently, but are
  never presented as one jointly attainable checkpoint.
- Dataset-dependent energy residual calibration is fit on training structures
  only and saved together with the uncalibrated metrics.
- Precision, backend, compilation mode, parameter count, and comparison budget
  are recorded with each external-model or throughput result.
- Logs may document execution, but manuscript tables should be generated from
  the corresponding JSON/CSV summaries.

See [`REPRODUCE.md`](REPRODUCE.md) for the supported entry points.  Historical
as-run queue scripts are retained when they are needed to audit an existing
result; they are not all intended as portable one-command examples.
