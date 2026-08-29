# Phase-specific causal DAG and identification plan

## Target questions

This secondary experiment separates two questions that the original ablation cannot answer alone.

1. At a fixed trained Full checkpoint, does the predictor actually use relative edge phase?
2. Under the same training protocol, does allowing relative phase improve the learned predictor compared with an otherwise identical phase-aligned model?

The first is a controlled functional intervention on a frozen model. The second is an end-to-end learning-algorithm intervention. Neither assigns a physical observable to the learned phase.

## DAG

```text
dataset/split D ───────────────────────────────────────────────────────────┐
seed S ───────────────────────────────────────────────────────────────────┤
frozen architecture and optimiser H ─────────────────────────────────────┤
                                                                           ▼
trained phase policy T ──► optimisation trajectory W ──► selected ckpt K ──► test error Y
                                  ▲                         ▲                 ▲
training data X_train ────────────┘       validation X_val ┘       test X_test

At fixed K:
phase intervention I ──► local relative phase Φ_I ──► Hermitian density R_I ──► prediction/error Y_I
amplitude, messages, graph, weights and checkpoint K ────────────────────────► R_I,Y_I
```

For the trained contrast, `T` is Full learned phase or Aligned-Full. The optimisation trajectory and validation selection are mediators and remain part of the estimand. For the frozen-checkpoint contrasts, `K` is held fixed and only the phase variable is intervened upon.

## Identification and controls

- Every contrast is paired by dataset and seed.
- Full and Aligned-Full have identical parameter names and counts. Both retain the positive-amplitude branch and the complete Hermitian \(j,k\) density. Only relative phase is disabled in Aligned-Full.
- The aligned intervention removes relative phase without changing amplitudes or model weights.
- The permutation intervention preserves the local phase multiset but breaks phase-to-edge assignment.
- Phase-dose interventions test whether error changes systematically as learned relative phase is restored.
- A common phase shift is a negative control. Predictions must remain invariant within numerical tolerance.
- Validation Force MAE selects checkpoints. The test split is never used for selection or training.

## Interpretation boundary

A favourable trained contrast supports the algorithmic value of allowing relative phase in this fixed CHORUS configuration. A favourable frozen-checkpoint contrast shows that the selected predictor functionally relies on its learned relative phases. These results do not identify a physical phase carried by atoms, and they do not turn the learned internal variable into a molecular causal quantity.

The four datasets were deliberately chosen as a mechanism panel. Dataset-level effects are therefore primary; the equal-weight macro effect summarises this panel rather than a population of all materials or all equivariant architectures.
