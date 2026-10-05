# Validation report: complete MACE Pearl analysis

## Overall assessment

**Ready to share with explicit inferential scope.** All declared MACE Final and Persistent model cells, frozen probes, and counterfactual summaries are complete. Key macro effects were independently reconstructed from the saved paired tables.

## Verified checks

- Trained cells: 120/120; frozen probes: 40/40.
- Full/Aligned parameter counts match within both Final and Persistent scopes.
- Validation Force MAE selects every checkpoint; test is evaluation-only.
- Common-shift Force-MAE replay differs by at most 0.000002 meV Å⁻¹; the maximum pointwise force-change audit is 0.005722 meV Å⁻¹.
- Final and Persistent macro dose curves are monotonic.
- Counterfactual coverage: 9,929,350 atomic force outcomes and 224,910 frame-energy outcomes. They are descriptive nested outcomes, not independent causal replicates.

## Inferential sensitivity

Persistent Full versus Aligned has a positive equal-dataset mean, and its hierarchical-bootstrap interval is above zero. The exact sign-flip p-value is 0.0305; Wilcoxon p=0.0637; the 11/20 win-count binomial p=0.8238. The magnitude-weighted mean is therefore positive, while rank- and sign-based evidence is weaker. The manuscript should report the effect and interval without calling it uniformly seed-stable.

## Claim boundary

The analysis identifies effects of representation policies inside the fixed MACE-ICTC/CHORUS training system. Frozen potential outcomes are model-internal counterfactuals, not causal properties of molecules. Diagonal versus Off remains a total module effect with active-capacity co-intervention.
