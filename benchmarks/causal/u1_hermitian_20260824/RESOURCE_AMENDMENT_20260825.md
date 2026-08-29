# Resource-only campaign amendment — 2026-08-25

The maximum concurrent A100 allocation was increased from two to four GPUs by
explicit user instruction after array submission. This changes wall-clock campaign
duration only. Dataset splits, interventions, seeds, model settings, optimisation,
checkpoint selection, test evaluation, and the causal estimands remain frozen.

Wall time measured during the concurrent campaign must not be used as a serial
training-throughput result.
