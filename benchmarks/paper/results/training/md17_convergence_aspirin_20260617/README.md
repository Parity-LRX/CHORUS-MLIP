# rMD17 Aspirin Convergence Analysis

Source run: `/tmp/mace_ictc_train_multisystem_apple300_maceinit_average_e0_20260617_061128` on the RTX 4090 host.

This directory contains only `revised_aspirin` logs from that run. The same remote run also contained an earlier benzene attempt, but that benzene data is intentionally excluded because it predated the edge-shift sanitation fix. The aspirin H5 files were audited after the fix path and had zero nonperiodic edge shifts and maximum edge length below the 4.5 A cutoff.

Protocol:

- 300 epochs from random MACE-compatible initialization.
- 3 seeds: `20260616`, `20260617`, `20260618`.
- Modes: ICTC eager, ICTC compiled, ICTC+cuEq compiled, MACE e3nn, MACE cuEq.
- Metrics are validation RMSE values only: force RMSE in eV/A and energy RMSE in eV/atom.
- Scalar validation losses are not compared because MACE-ICTC and mace-torch log different internal loss normalizations.

Files:

- `curves.csv`: per-epoch validation RMSE curves.
- `runs.csv`: per-run best/final RMSE and threshold epochs.
- `aggregate_by_mode.csv`: mean/std aggregation by mode.
- `summary.md`: human-readable summary.
- `figures/revised_aspirin_convergence.{png,pdf}`: mean convergence curves by mode.
