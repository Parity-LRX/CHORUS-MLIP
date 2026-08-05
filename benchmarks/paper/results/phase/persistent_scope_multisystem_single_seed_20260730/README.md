# Final versus Persistent scope expansion

All values follow the manuscript reporting protocol: minimum validation total
loss selects one checkpoint, and energy and force metrics are read from that
same checkpoint. No independently optimized energy checkpoint is substituted.
The fixed-composition systems receive no post-hoc energy calibration.

| System | Split | Final E MAE | Persistent E MAE | Final F MAE | Persistent F MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| rMD17 aspirin | validation | **0.545** | 0.661 | 16.519 | **15.330** |
| rMD17 benzene | validation | **0.066** | 0.433 | 1.947 | **1.687** |
| MD22 Buckyball Catcher | held-out test | 0.249 | **0.229** | 11.501 | **10.082** |
| MD22 Ac-Ala3-NHMe | held-out test | 0.568 | **0.478** | 20.664 | **19.692** |
| MD22 DHA | held-out test | 0.616 | **0.589** | 16.860 | **15.922** |

Energy is in meV/atom and force in meV/Angstrom. These are single-seed
results (`20260616`). The MD22 test sets are never used for checkpoint
selection. The complete paired MAE/RMSE values and selected validation steps
are in `results.json`.
