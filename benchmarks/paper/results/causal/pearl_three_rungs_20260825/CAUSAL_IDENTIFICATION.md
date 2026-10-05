# Causal identification specification

## Structural system

For dataset `D`, seed `S`, architecture policy `A`, training data `X_train`, and residual GPU execution noise `U`, training produces weights `W=f(D,S,A,X_train,U)`. Validation selection produces `K=g(W,X_val)`, and the held-out outcome is `Y=h(K,X_test)`. The end-to-end potential outcome is therefore `Y_d,s(A)`, with optimisation and checkpoint selection retained as mediators.

At a fixed Full checkpoint `K`, the intervention `I` changes the local phase rule while messages, amplitudes, graph, weights, and examples stay fixed. The frozen potential outcome is `Y_K(I)`. The common-shift intervention is a null edge in this structural model under U(1) invariance.

## Estimands

For each dataset and seed, define the paired log benefit `δ_d,s = log(Y_d,s(control) / Y_d,s(treatment))`. Dataset effects average `δ` over five seeds. The equal-dataset macro averages the four dataset effects, and the reported percentage is `100[1-exp(-mean δ)]`. This keeps multiplicative error changes symmetric before conversion to an interpretable reduction.

The three principal estimands are:

1. **Self-density module:** Diagonal versus Off. This is a joint representation-and-capacity intervention because parameter counts differ.
2. **Cross-neighbour and phase decomposition:** Aligned versus Diagonal, followed by Full versus Aligned. The latter is nominally parameter matched; 32 phase-head weights are inactive in Aligned-Full.
3. **Functional phase use:** factual Full versus aligned, permuted, and dose-controlled phase at the same checkpoint.

Full versus Charge2 is retained as a same-parameter sensitivity control for the joint U(1)-neutral Hermitian closure. Its macro effect is -0.30% with an interval spanning zero, so the present panel does not establish an accuracy advantage over that non-neutral quadratic control.

## Identification assumptions

- **Consistency:** each mode implements the declared intervention, and the recorded outcome belongs to that implementation.
- **Positivity:** all declared modes exist for every one of the 4×5 dataset-seed blocks.
- **No cross-run interference:** one training run cannot change another run's model or dataset.
- **Paired exchangeability by design:** dataset, split, seed, optimiser, budget, and selection rule are fixed within each pair. GPU deterministic algorithms were not forced, leaving residual execution noise.
- **Selection integrity:** validation Force MAE selects one checkpoint; test outcomes remain downstream measurements only.
- **Frozen-intervention exclusion:** alignment, dose, and permutation alter only phase coefficients at inference; amplitudes, messages, graph, weights, and examples remain fixed.

## What is identified

The frozen interventions identify how the selected predictor function changes under phase manipulation. The trained contrasts identify effects of an algorithmic policy on the distribution of validation-selected predictors under the fixed protocol. They do not identify a molecular causal variable, since θ is learned and has no assigned physical observable.

The strongest supported statement is that CHORUS supplies a robust Hermitian second-moment update and learns phase-dependent fitted functions. The present retraining evidence for additional generalisation from relative phase is positive but inconclusive at the panel level. A widened-Off 4×5 paired control would be required to separate the self-density representation from active parameter capacity with the same strength as the Full-versus-Aligned contrast.
