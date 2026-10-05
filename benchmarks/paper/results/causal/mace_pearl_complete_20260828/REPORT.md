# Complete MACE-only Pearl three-rung analysis

## Answer first

The MACE evidence separates architectural benefit from functional dependence. The Final intervention ladder assigns the most reproducible generalisation gain to the Hermitian self-density channel: 9.07% (95% hierarchical-bootstrap CI 4.89% to 11.92%; 20/20 wins). Cross-neighbour closure contributes 1.57% (95% hierarchical-bootstrap CI -2.39% to 5.15%; 14/20 wins), while learned relative phase adds 0.61% (95% hierarchical-bootstrap CI -0.12% to 1.30%; 16/20 wins) over a parameter-matched Aligned control. Under Persistent propagation, the independently retrained phase effect is 0.91% (95% hierarchical-bootstrap CI 0.08% to 1.81%; 11/20 wins). The Persistent-minus-Final change in phase benefit is +0.30% (95% CI -0.81% to +1.43%).

The same fitted functions rely much more strongly on their learned relative phase than the retraining contrast suggests. Frozen alignment raises the equal-dataset macro Force-MAE ratio to 1.866× in Final and 3.529× in Persistent. Equivalently, factual phase improves Force MAE by 46.41% (95% hierarchical-bootstrap CI 19.10% to 64.07%; 20/20 wins) in Final and 71.66% (95% hierarchical-bootstrap CI 26.63% to 90.70%; 20/20 wins) in Persistent relative to the aligned counterfactual. Breaking phase-to-edge assignment gives 69.40% (95% hierarchical-bootstrap CI 31.90% to 86.06%; 20/20 wins) and 84.66% (95% hierarchical-bootstrap CI 46.03% to 95.77%; 20/20 wins), respectively. Retraining therefore compensates for much of the removed phase mechanism; it does not imply that the fitted Full predictor ignores phase.

## Pearl rung 1: seeing

The observational rung is deliberately descriptive. In the Final checkpoints, fitted phase-organisation statistics were associated with trained and frozen effects using ranks within dataset. Across the combined Final/Persistent model-scope units, the association between end-to-end phase benefit and frozen alignment harm is r=-0.088 with a stratified permutation p=0.6483. This relationship is post hoc and is not used as a causal estimate.

## Pearl rung 2: doing

Each architecture policy was independently trained under the same split, seed, optimiser, precision, step budget, and validation-only checkpoint rule. Dataset and seed define the paired causal blocks. Final provides the full Off → Diagonal → Aligned → Full ladder. Persistent tests Full versus Aligned using the same parameter schema. The resulting scope interaction compares paired log benefits before averaging datasets equally. It identifies whether propagation scope modifies the algorithmic benefit of allowing learned relative phase under the frozen protocol.

The trained results support a clear hierarchy. Hermitian self-density has the largest stable end-to-end effect. Cross-neighbour closure and learned phase provide smaller, heterogeneous incremental effects on this four-system panel. Persistent propagation increases the mean learned-phase benefit only modestly, and its interval determines whether that increase is distinguishable from seed and dataset variation.

## Pearl rung 3: imagining

For every selected Full checkpoint, the factual prediction and its aligned, dose-controlled, permuted, and common-shift potential outcomes were evaluated on the same held-out structures. Messages, amplitudes, graph, examples, and all learned weights were fixed. The dose response is monotonic at the equal-dataset macro level in both scopes: as λ moves from zero alignment to the learned phase, the Force-MAE ratio falls from 1.866 to 1.000 in Final and from 3.529 to 1.000 in Persistent. The maximum common-shift change is 0.005722 meV Å⁻¹, confirming that predictions depend on relative phase rather than an arbitrary common phase origin.

The counterfactual tables contain 4,964,675 Final and 4,964,675 Persistent atomic force outcomes, plus 112,455 and 112,455 frame-energy outcomes. These rows localise effect heterogeneity but are not treated as independent replicates. Model-level paired inference remains the basis for confidence intervals.

## Causal interpretation

The three rungs answer different questions. Seeing describes how fitted phase organisation covaries with outcomes. Doing estimates the effect of permitting a representation rule in the training algorithm. Imagining quantifies how the selected predictor would change under controlled internal phase interventions. Together they show that the Hermitian second-moment channel drives the most reproducible generalisation gain, while learned relative phase is functionally embedded in the fitted solution and becomes more consequential when the charged state is propagated.

The analysis identifies causal effects inside this fixed MACE-ICTC/CHORUS training system. Theta is an internal learned variable, not a molecular observable. Diagonal versus Off is a joint representation-and-capacity intervention because active parameter counts differ. Full versus Aligned is the cleaner phase contrast because parameter counts and state schemas match. The four datasets form a mechanism panel rather than a random sample of chemical space.

## Validation status

- Input gate passed: `True`.
- Final cells: 80/80; Persistent cells: 40/40; frozen probes: 40/40.
- Test data never selected checkpoints.
- Full/Aligned parameter counts match within each scope and dataset/seed pair.
- Confidence intervals resample datasets and then seeds; atoms and frames are never promoted to causal replicates.
- Frozen scope interaction in alignment benefit: +89.11% (95% CI +7.44% to +317.68%).
