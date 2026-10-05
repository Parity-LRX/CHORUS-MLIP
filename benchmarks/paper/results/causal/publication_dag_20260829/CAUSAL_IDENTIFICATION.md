# Causal identification and estimands

The DAG is an identification map for designed interventions, not a predictive
Bayesian network fitted to atom-level observations. Causal effects are
estimated from paired retraining and fixed-checkpoint interventions below; no
separate "DAG model" is trained.

For legibility, the rendered graph groups C, S, B, P, and H into the design-
context node X; the estimands and audits retain them separately.

## Variables in the DAG

- **C**: dataset, fixed train/validation/test split, and reference labels.
- **S**: paired random seed.
- **B**: complete backbone implementation. It is an effect modifier, not a randomised treatment.
- **P**: Final or Persistent propagation scope.
- **T**: independently retrained component policy: Off, Diagonal, Aligned, or Full.
- **H**: fixed training protocol, precision, optimiser-step budget, loss weights, cutoff, and checkpoint rule.
- **W**: learned weights produced by optimisation.
- **K**: checkpoint selected from validation Force MAE only.
- **I**: same-checkpoint phase intervention: alignment, dose, within-centre permutation, or common shift.
- **Φ**: learned relative-phase organisation.
- **ρ**: neutral Hermitian density returned to the O(3) feature stream.
- **Y**: held-out Energy/Force error.

## Primary retraining estimand

For treatment policy (t), control policy (c), backbone (b), and scope (p),

\[
\Delta_{b,p;t,c}
=\frac{1}{4}\sum_d\frac{1}{5}\sum_s
\log\frac{Y_{b,p,c,d,s}}{Y_{b,p,t,d,s}}.
\]

The reported percentage is (100[1-\exp(-\Delta)]). Equal dataset weight prevents the largest test set from defining the scientific target. Seeds are paired within dataset and intervention.

## Frozen-model estimand

For intervention (I),

\[
\Gamma_{b,p;I}
=\frac{1}{4}\sum_d\frac{1}{5}\sum_s
\log\frac{Y_{b,p,I,d,s}(K)}{Y_{b,p,\mathrm{native},d,s}(K)}.
\]

Here (K), all learned parameters, the neighbour graph, and held-out input are identical. This identifies functional reliance of the fitted predictor on the intervened phase structure. It does not identify a physical treatment effect on molecules.

## Scope and backbone interactions

The propagation interaction is (\Delta_{b,\mathrm{Persistent}}-\Delta_{b,\mathrm{Final}}), or the analogous frozen (\Gamma) contrast. A backbone contrast compares these paired effects between complete NequIP-SH and MACE-ICTC configurations. Because backbone identity bundles multiple implementation choices, it is evidence of transportability and effect modification, not an isolated causal effect of replacing one backbone.

## Identification assumptions and design checks

1. **Consistency**: each policy name maps to one audited implementation and protocol.
2. **Positivity**: all declared dataset-by-seed cells exist for every compared policy.
3. **Paired optimisation control**: dataset, split, seed, budget, loss, cutoff, precision, and checkpoint rule are matched within each retraining contrast.
4. **No test selection**: the test split never selects checkpoints or energy corrections.
5. **No cross-run interference**: one training run does not alter another run's data or optimiser state.
6. **Outcome aggregation**: frames and atoms are nested measurements; uncertainty is clustered at dataset and seed.
7. **Capacity boundary**: Diagonal versus Off includes active-capacity change and estimates the total Hermitian-module effect. Full versus Aligned retains a matched parameter schema.
8. **Population boundary**: the four datasets form a mechanism panel. Generalisation to all chemistry requires further datasets.

## What the DAG does not identify

- a causal effect of a physically observable phase, because θ is an internal learned variable;
- a universal chemistry-population average;
- mediation through any single scalar phase statistic;
- a pure backbone effect;
- independent evidence from millions of atom/frame rows.
