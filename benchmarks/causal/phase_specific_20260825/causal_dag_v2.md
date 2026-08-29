# CHORUS causal graph v2

## Scope

The causal analysis concerns a learned representation rule inside a fixed MLIP
training system.  It does not treat the learned phase as a molecular observable.
The causal units are independently trained models for end-to-end contrasts and
frozen validation-selected predictors for inference-time interventions.  Frames
and atoms are repeated outcomes nested inside those units; they are not counted
as independent experimental replicates.

## Structural graph

```text
dataset and split D ---------------------------------------------------+
seed S ---------------------------------------------------------------+|
backbone/configuration B --------------------------------------------+||
propagation scope P ------------------------------------------------+|||
phase policy T ----------------------------------------------------+||||
training protocol H ----------------------------------------------+|||||
                                                                  vvvvvv
training data X_train ---> optimisation trajectory W ---> selected checkpoint K ---> held-out error Y
                                                   ^             ^
validation data X_val -----------------------------+             |
test examples X_test ---------------------------------------------+

At fixed K:
phase intervention I ---> relative phase Phi_I ---> Hermitian density R_I ---> prediction/error Y_I
messages, amplitudes, graph, examples, and all weights except the intervened phase rule -----------^
```

Here `T` distinguishes Off, Diagonal, Aligned, and Full.  `P` distinguishes
Final and Persistent propagation.  `B` distinguishes the complete MACE-ICTC
and NequIP configurations used in the mechanism experiments.  Optimisation and
validation checkpoint selection are mediators of an end-to-end architecture
policy, so they remain inside that estimand.

## Identified effects

For outcome error `Y`, the paired benefit of treatment over control in dataset
`d` and seed `s` is

```text
delta[d,s] = log(Y[d,s,control] / Y[d,s,treatment]).
```

Effects are averaged over seeds within dataset and then equally over the four
datasets.  Reporting `100 * (1 - exp(-mean(delta)))` gives the corresponding
relative error reduction.  Test outcomes are evaluated only after validation
Force MAE has selected the checkpoint.

1. **Hermitian self-density total effect:** Diagonal versus Off.  This adds the
   module and its active parameters.  Separate capacity controls already test
   whether comparable widening reproduces the gain, but the causal label here
   remains the total module effect rather than a parameter-free mathematical
   effect.
2. **Cross-neighbour closure:** Aligned versus Diagonal.
3. **Learned relative phase:** Full versus Aligned.  These modes share the same
   state schema and nominal parameter count; the aligned policy disables the
   relative phase contrast.
4. **Propagation-scope interaction:**

   ```text
   [log(Y_Persistent,Aligned / Y_Persistent,Full)]
   - [log(Y_Final,Aligned / Y_Final,Full)].
   ```

   A positive value means learned relative phase has a larger benefit under
   Persistent propagation.
5. **Backbone replication contrast:** the Full-versus-Aligned effect is estimated
   independently in MACE-ICTC and NequIP.  Their difference describes effect
   modification across the two bundled backbone configurations; it is not a
   pure causal effect of changing only the backbone.
6. **Frozen functional interventions:** at a selected Full checkpoint, align,
   permute, or continuously attenuate relative phase while holding messages,
   amplitudes, graph, examples, and weights fixed.

## Negative controls and heterogeneity

- A common phase shift must leave predictions invariant within numerical
  tolerance.  This tests the intended U(1) redundancy.
- Phase permutation preserves the local phase multiset while breaking its edge
  assignment.
- Frozen intervention effects are stratified by reference-force magnitude,
  neighbour count, and element.  Inference resamples trained predictors (the
  dataset-by-seed clusters), never individual atoms as if they were independent.
- Absolute intervention harm and proportional error inflation are both reported.
  A larger absolute effect in difficult environments is not interpreted as
  stronger mechanistic dependence unless the proportional contrast agrees.

## Completion gate

No scope or backbone interaction is reported from a partial campaign.  A cell
enters the synthesis only when all declared dataset-by-seed modes are complete,
the validation selection rule is auditable, metrics are finite, and Full/Aligned
parameter counts agree within that configuration.

## Claim boundary

The trained contrasts identify the effect of permitting a representation rule
under the declared training protocol.  Frozen interventions identify functional
reliance of the selected predictor on relative phase.  The analysis does not
assign physical meaning to `theta`, estimate a causal property of a molecule,
or establish one representation as uniquely necessary for every equivariant
architecture.
