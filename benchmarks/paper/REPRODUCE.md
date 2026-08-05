# Reproducing CHORUS benchmarks

Run commands from the repository root in an environment containing the
dependencies required by the selected backbone.  GPU results in the paper use
the precision and compilation mode recorded with each result; changing either
can materially change throughput and sometimes optimization.

## Mechanism and backbone studies

The main MACE-ICTC training entry point is:

```bash
python -m chorus.cli.train --help
```

The maintained benchmark drivers are under
`benchmarks/paper/scripts/training/`.  They cover the four principal controls
(CHORUS, self-density, density-preserving attention, and phase-off), rank and
scope studies, Transition1x train-only energy calibration, MD22/3BPA/xxMD
evaluation, and selected-checkpoint audits.

NequIP and NequIP-ICTC use the integration captured in
`integrations/nequip/`.  Apply its patch to NequIP v0.6.2 or copy its overlay
onto a clean checkout.  The same `HermitianDensityResidual` implementation is
selected with:

```yaml
interaction_backend: e3nn  # or ictc
chorus_enabled: true
chorus_scope: final
chorus_rank: 16
```

## External-model comparisons

External configurations are stored under `benchmarks/paper/external/`.
Evaluation and energy-calibration helpers are under
`benchmarks/paper/scripts/evaluation/` and
`benchmarks/paper/scripts/training/`.  For every model:

1. train for the declared optimizer-step budget;
2. scan the complete validation history;
3. select one checkpoint using the declared validation rule;
4. evaluate that checkpoint once on the test split;
5. if required, fit an energy residual using training structures only.

Do not select test checkpoints or combine the independent best-energy and
best-force validation points into a fictitious model.

## Throughput

The cross-model scaling archive is:

```text
benchmarks/paper/results/throughput/cross_model_atom_scaling_20260728/
```

Regenerate its table and figure with:

```bash
python benchmarks/paper/scripts/throughput/summarize_cross_model_scaling.py
python benchmarks/paper/scripts/plot_tece_atom_scaling.py
```

Use the exact model widths, CHORUS ranks, strict-float32 policy, and compilation
settings recorded in the raw JSON.  Runs sharing a GPU with another process may
be used for accuracy but not for the formal wall-time or throughput table.

## Artifact boundaries

Git retains source, compact JSON/CSV summaries, selected text logs, and SVG
figures.  Datasets, checkpoints, full trajectories, compiled packages, and
temporary manuscript renders belong in external artifact storage.  A result is
ready for citation only when its machine-readable summary records the data
split, checkpoint rule, units, parameter count, precision, and provenance.
