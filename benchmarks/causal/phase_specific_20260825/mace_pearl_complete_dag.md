# MACE-only Pearl three-rung DAG

## Causal units

- End-to-end effects use independently trained dataset-by-seed models.
- Frozen effects use validation-selected predictors paired with the same held-out input.
- Frames and atoms are nested outcomes. They are never counted as independent causal replicates.

## Structural graph

```text
dataset/split D -------------------------------+
seed S ----------------------------------------+|
fixed protocol H ------------------------------+||
propagation scope P ---------------------------+|||
training policy T -----------------------------+||||
training examples X_train ---------------------+|||||
                                                vvvvvv
                                      optimisation W
                                                |
validation examples X_val -------------------->| checkpoint K
                                                |
held-out examples X_test ----------------------+------> factual error Y

At fixed K and X_test:

phase intervention I -> relative phase Phi(I) -> Hermitian density R(I)
                    -> prediction f_K(I, X_test) -> potential outcome Y_K(I)

messages, amplitudes, graph, weights, and held-out examples are fixed.
```

`T` is Off, Diagonal, Aligned, or Full in Final scope. Persistent scope uses the parameter-matched Aligned and Full policies. `P` is Final or Persistent. Optimisation and validation selection are downstream mediators of an architecture policy, so they stay inside the end-to-end estimand.

## Pearl ladder

1. **Seeing.** Fitted phase-organisation statistics and the association between frozen phase reliance and trained generalisation benefit. These are post-hoc associations.
2. **Doing.** Independently retrained paired policies identify Diagonal versus Off, Aligned versus Diagonal, Full versus Aligned, and the Persistent-minus-Final interaction in the Full-versus-Aligned effect.
3. **Imagining.** At one selected Full checkpoint, alignment, phase dose, within-centre phase permutation, and a common-shift negative control produce model-internal potential outcomes for the same held-out input.

## Identified effects

- Diagonal versus Off is the total self-density module effect. Active capacity changes with the module.
- Aligned versus Diagonal adds phase-free cross-neighbour closure.
- Full versus Aligned adds learned relative phase with matched nominal parameter count and state schema.
- The propagation interaction is

  ```text
  log(Y_Persistent,Aligned / Y_Persistent,Full)
  - log(Y_Final,Aligned / Y_Final,Full).
  ```

- Frozen `Y_K(I)` contrasts identify functional reliance of a fitted predictor on its internal relative-phase rule.

## Claim boundary

The effects belong to the declared MACE-ICTC/CHORUS training system and four-dataset mechanism panel. The learned phase is not assigned a molecular observable. The common-shift test validates the intended U(1) redundancy; it does not by itself prove that U(1) is the only possible implementation of an effective Hermitian density rule.
