# Pearl causal ladder used in the CHORUS mechanism analysis

| Rung | Question | Evidence | Identified claim |
|---|---|---|---|
| Seeing | Which learned phase and environment summaries co-vary with benefit? | Within-stratum MACE phase/environment diagnostics with dataset and seed retained as clusters | Descriptive association only; atom and frame rows are nested observations. |
| Doing | What changes when the operator policy is changed before optimisation? | Independently retrained Off, Diagonal, Aligned, and Full policies, paired by dataset and seed | Controlled algorithmic effects under the fixed benchmark protocol. |
| Counterfactual | Does this fitted checkpoint functionally depend on its learned relative phase? | Same-checkpoint alignment, phase dose, within-centre permutation, and common-shift interventions | Fitted-model phase reliance and the U(1) invariance negative control. |

The rungs answer different questions and are not pooled as exchangeable evidence.
Retraining measures the net effect after optimisation can compensate; frozen
interventions measure functional reliance of a particular fitted predictor.
Agreement strengthens the mechanism claim, while differences between the rungs
are expected and scientifically informative.
