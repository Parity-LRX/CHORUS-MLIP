# Causal DAG and identification plan

## Target question

The primary question is deliberately narrow: within the frozen small MACE-ICTC/CHORUS-Final configuration, what is the effect on held-out Force MAE of replacing the non-Hermitian charge-2 closure \(z\otimes z\) with the U(1)-neutral Hermitian closure \(z\otimes z^*\)?

This is an algorithmic intervention, not an observational association. Every mode is run for every dataset and seed. The test split is evaluated only after the validation Force-MAE rule has selected one checkpoint.

## DAG

The machine-readable graph is [`causal_dag.dot`](causal_dag.dot). Its paths are:

```text
                            ┌──────── parameterisation / compute P ────────┐
intervention T ─────────────┤                                               ▼
                            └──────── quadratic representation R ──► optimisation W
seed S ─────────────────────────────────────────────────────────────►       │
training split X_train ─────────────────────────────────────────────►       │
frozen protocol H ──────────────────────────────────────────────────►       │
                                                                            ▼
validation split X_val ───────────────────────────────────────────► validation path V
                                                                            │
                                                                            ▼
                                                               selected checkpoint K
                                                                            │
held-out test split X_test ──────────────────────────────────────────────────┴──► test error Y
```

The optimisation trajectory and validation-based selection are mediators of the intervention. They are not adjusted away because the target is the end-to-end learning-algorithm effect under the declared training protocol.

## Identification

- **Blocking and pairing.** Dataset, split, seed, architecture, optimiser, precision, training steps, and checkpoint rule are held fixed within a paired contrast. The analysis first computes a treatment/control log-error ratio within each dataset-seed pair.
- **Primary contrast.** Full versus Charge2 has the same latent rank, phase network, projection maps, writeback maps, parameter count, optimiser, and step budget. Parameter equality is checked from every checkpoint before analysis.
- **Selection.** Validation Force MAE determines the checkpoint. Test metrics have no path into the selected checkpoint.
- **Aggregation.** Effects are reported per dataset, followed by an equal-weight macro average over the four predeclared systems. A deterministic hierarchical paired bootstrap resamples datasets and paired seeds.
- **Concurrent execution.** Full and Charge2 jobs for the same dataset-seed block are adjacent array tasks and run under the same A100 partition. Accuracy is admissible; concurrent wall time and throughput are not.

## What the DAG can and cannot establish

The Full/Charge2 intervention identifies the joint effect of Hermitian conjugation and U(1)-neutral closure. Those two properties change together, so this experiment cannot assign separate causal coefficients to “Hermiticity” and “neutrality”. The diagonal and Off modes answer different secondary questions: whether cross-neighbour terms add value beyond self-density, and whether self-density adds value beyond the unaugmented backbone.

The panel was chosen using prior mechanistic evidence rather than sampled randomly from chemical space. The dataset-level effects are therefore the main result. Their macro average describes this panel and is not an estimator of a universal effect over all materials or all equivariant architectures.

## Falsification and audit checks

1. Full and Charge2 must have exactly equal trainable parameter counts for every dataset-seed pair.
2. All four modes must use identical split files, source revision, precision flags, optimiser-step budget, and checkpoint rule.
3. Any test access before checkpoint selection invalidates the run.
4. Report Full/Charge2 losses even when the result is negative; do not stop after favourable seeds.
5. Treat a systematic Charge2 win as evidence against the proposed joint closure mechanism under this protocol.
