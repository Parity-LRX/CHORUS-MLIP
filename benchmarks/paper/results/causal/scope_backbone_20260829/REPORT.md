# CHORUS scope and backbone causal synthesis

All entries use one validation-Force-MAE-selected checkpoint per trained model. Positive reductions favour Full over the named control. Frames and atoms are not treated as independent causal units.

## Within-configuration effects

| Contrast | Error reduction | 95% hierarchical-bootstrap interval | Wins |
|---|---:|---:|---:|
| MACE-ICTC Final full vs aligned | 0.61% | [-0.12, 1.30]% | 16/20 |
| MACE-ICTC Persistent full vs aligned | 0.91% | [0.08, 1.81]% | 11/20 |
| NequIP Final diagonal vs off | 18.33% | [9.35, 26.51]% | 20/20 |
| NequIP Final aligned vs diagonal | 20.44% | [6.13, 30.32]% | 17/20 |
| NequIP Final full vs aligned | 0.42% | [-0.41, 1.26]% | 13/20 |
| NequIP Persistent full vs aligned | 2.43% | [1.14, 3.82]% | 18/20 |

## Effect modification

| Interaction | Benefit-ratio change | 95% hierarchical-bootstrap interval | Positive pairs |
|---|---:|---:|---:|
| MACE-ICTC Persistent-minus-Final phase interaction | 0.30% | [-0.82, 1.43]% | 10/20 |
| NequIP Persistent-minus-Final phase interaction | 2.06% | [0.45, 3.86]% | 15/20 |
| NequIP-minus-MACE Final phase-effect replication contrast | -0.19% | [-1.08, 0.80]% | 6/20 |
| NequIP-minus-MACE Persistent phase-effect replication contrast | 1.56% | [-0.26, 3.61]% | 15/20 |

The propagation-scope interaction compares Persistent and Final inside one backbone. The backbone interaction compares two complete bundled configurations; it is evidence of transportability or effect modification, not a pure intervention on backbone identity alone.
