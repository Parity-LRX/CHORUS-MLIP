# rMD17 Ethanol Convergence Analysis

Source run: `/tmp/mace_ictc_train_apple300_maceinit_average_e0_from100_20260617_014808` on the RTX 4090 host.

This is a continuation experiment: each mode starts from the corresponding 100-epoch checkpoint produced by `/tmp/mace_ictc_train_apple100_maceinit_average_e0_20260617_002357` and continues to epoch 300. It is useful convergence evidence, but it should be labeled separately from from-scratch 300-epoch runs.

The prepared ethanol H5 files were audited and had zero nonperiodic edge shifts and maximum edge length below the 4.5 A cutoff.

Protocol:

- Continue 100-epoch checkpoints to 300 epochs.
- 3 seeds: `20260616`, `20260617`, `20260618`.
- Modes: ICTC eager, ICTC compiled, ICTC+cuEq compiled, MACE e3nn, MACE cuEq.
- Metrics are validation RMSE values only: force RMSE in eV/A and energy RMSE in eV/atom.
- Scalar validation losses are not compared because MACE-ICTC and mace-torch log different internal loss normalizations.

Files:

- `curves.csv`: per-epoch validation RMSE curves.
- `runs.csv`: per-run best/final RMSE and threshold epochs.
- `aggregate_by_mode.csv`: mean/std aggregation by mode.
- `summary.md`: human-readable summary.
- `figures/revised_ethanol_convergence.{png,pdf}`: mean convergence curves by mode.
