# Classical DAG extension plan

## Status and separation from the confirmatory campaign

The frozen 80-run campaign remains the confirmatory analysis. Its treatments,
seeds, datasets, checkpoints, estimands, and summary code must not change.

This document specifies a second-stage, environment-level analysis motivated
after partial aggregate results were visible. It must therefore be reported as
a secondary mechanism and treatment-effect-heterogeneity analysis, not as part
of the original preregistration. It uses only checkpoints selected by the
frozen validation Force-MAE rule.

## Causal questions

1. **Primary total effect:** for a fixed dataset, seed, test environment, and
   training protocol, how does Full change force error relative to Charge2?
   This identifies the joint effect of Hermitian conjugation and U(1)-neutral
   closure; it does not identify separate coefficients for those properties.
2. **Secondary total effects:** Full versus Diagonal measures the contribution
   of cross-neighbour terms within the Hermitian construction, and Diagonal
   versus Off measures the contribution of stable self-density.
3. **Effect heterogeneity:** which pre-treatment chemical and geometric
   environments are associated with larger or smaller paired treatment effects?
4. **Mechanism probes:** do controlled interventions on the learned phase alter
   predictions in the manner expected from cross-neighbour interference?

## Units, treatments, and outcomes

- The evaluation unit is an atom in a held-out frame, indexed by dataset,
  training seed, frame, and atom.
- Every unit is evaluated under every mode, so both potential outcomes are
  observed for each trained seed. Pairing must be preserved.
- The primary atomic outcome is the mean absolute Cartesian force-component
  error. Its mean over atoms exactly reproduces the reported Force MAE.
- Per-frame Energy MAE and vector-force error are secondary outcomes.
- Test labels never select checkpoints or tune the effect model.

## Pre-treatment covariates

Covariates may be computed only from the reference structure, reference label,
and dataset metadata. The initial set is:

- dataset/system and temperature where applicable;
- element, composition, atom count, and reference force magnitude;
- neighbour count within the frozen cutoff;
- radial-distance moments and local density;
- angular-distribution entropy and low-order rotational invariants;
- reference energy per atom where it is scientifically meaningful.

Atoms from the same frame must never be split across fitting and validation
folds. Trajectory identifiers must be used as the outer grouping variable when
available.

## DAG and adjustment rules

```text
dataset/geometry X -------------------------------> outcome Y
        |                                               ^
        +---------------> mechanism state M -----------+

treatment T -> representation R -> optimisation W -> checkpoint K -> Y
        |                                      |
        +-----------------> mechanism state M -+

seed S --------------------> optimisation W -> checkpoint K -> Y
```

Treatment is crossed by design and has no pre-treatment parent. Consequently,
the total-effect backdoor adjustment set is empty. Dataset and seed are paired
blocks that improve precision rather than variables needed to repair treatment
confounding. Pre-treatment covariates are used to estimate effect modification.

Do not adjust the total-effect model for learned phase, cancellation statistics,
representation norms, optimisation trajectory, selected step, or checkpoint.
They are downstream mediators. Parameter count and step budget are design
constraints and must be audited rather than statistically adjusted.

## Estimation

1. Export paired predictions from the validation-selected checkpoint for every
   mode, dataset, seed, test frame, and atom.
2. Compute direct paired atomic and frame-level treatment differences before
   fitting any machine-learning model.
3. Report per-dataset effects first. Report the four-dataset macro effect only
   as a summary of the declared panel.
4. Fit a cross-validated effect-modifier model to the paired difference as a
   function of pre-treatment covariates. A direct paired-difference learner is
   primary; a causal forest or orthogonal learner is a sensitivity analysis,
   not required for identification.
5. Use grouped cross-validation and a hierarchical bootstrap over dataset,
   seed, and trajectory/frame clusters. Do not treat atoms from one frame as
   independent replicates.
6. Include seed-wise sign stability, leave-one-dataset-out sensitivity, and
   covariate-label permutation controls.

Applying DML or TMLE directly to the 80 aggregate runs is prohibited: the
treatment is fully crossed, the propensity is fixed by design, and the aggregate
sample is too small for a useful nuisance-model fit.

## Mechanism interventions

The following are post-training mechanism probes and must not replace the
retrained treatment contrasts:

- add a common phase shift to all incident edges and verify numerical U(1)
  invariance;
- set learned phases to zero while retaining the Hermitian cross-density;
- permute phases among neighbours of the same centre, preserving the marginal
  phase distribution while breaking edge-phase alignment;
- relate paired error changes to phase concentration, off-diagonal density
  fraction, and a predeclared constructive/destructive interference index.

Associations between a learned mediator and error do not identify a natural
indirect effect. Only explicit phase interventions may be described as causal
mechanism probes.

## Required artefacts

- `dag_environment.dot` and a rendered DAG;
- `paired_predictions.parquet` with schema and provenance;
- `pre_treatment_covariates.parquet`;
- `paired_ate.json`, `effect_heterogeneity.json`, and bootstrap draws;
- invariance and phase-intervention audit files;
- a concise report separating confirmatory total effects, secondary CATE
  analysis, and exploratory mediation evidence.
