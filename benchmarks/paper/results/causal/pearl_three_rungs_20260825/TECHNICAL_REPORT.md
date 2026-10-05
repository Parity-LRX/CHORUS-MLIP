# CHORUS causal-mechanism audit

## Technical summary

The complete Pearl three-rung analysis separates functional phase use from end-to-end generalisation. Stable Hermitian self-density is the most robust component: Diagonal reduces test Force MAE against Off by 9.07% (95% hierarchical-bootstrap CI 4.87% to 11.90%; 20/20 paired wins). Adding phase-free cross-neighbour closure contributes 1.57% (95% hierarchical-bootstrap CI -2.40% to 5.14%; 14/20 paired wins); its sign varies across datasets. Learned relative phase adds 0.61% (95% hierarchical-bootstrap CI -0.11% to 1.30%; 16/20 paired wins) over an independently trained Aligned-Full control. The mean is positive, while its interval includes zero.

The fitted Full predictors nevertheless use their learned phases strongly. Aligning relative phases at fixed weights worsens Force MAE by 46.41% (95% hierarchical-bootstrap CI 19.33% to 64.04%; 20/20 paired wins); permuting phase-to-edge assignments worsens it by 69.40% (95% hierarchical-bootstrap CI 32.04% to 86.11%; 20/20 paired wins). Restoring relative phase continuously lowers the equal-dataset Force-MAE ratio from 1.866 at λ=0 to 1.000 at λ=1. A shared phase shift changes forces by at most 0.0043 meV Å⁻¹, confirming the intended U(1) invariance.

Taken together, CHORUS learns phase-dependent solutions, but current evidence attributes most reproducible generalisation gain to the stable Hermitian second-moment channel. Cross-neighbour closure is useful but heterogeneous, and the incremental learned-phase effect is modest on this four-dataset panel. This is stronger than an ablation-only claim because the same conclusion survives trained interventions, frozen interventions, a dose response, an assignment intervention, and a symmetry negative control.

## Pearl rung 1: seeing

The observational extension measures relative-phase organisation in the validation-selected Full checkpoints. Associations are computed from ranks within each dataset, preventing between-dataset scale differences from generating the signal. The largest absolute association is `relative_phase_entropy_normalized` versus `frozen_alignment_log_harm` (r=0.475, within-dataset permutation p=0.0628); none of the twelve precomputed associations reaches p<0.05. Negative-cosine mass is rare in MAL (0.113%), Buckyball (0.000%), and 3BPA (0.000%), while ZrO is more variable (5.94% on average). The learned phase therefore acts mainly through graded pair reweighting on this panel, with explicit sign-reversing cancellation concentrated in ZrO. These twenty fitted-model observations are exploratory and do not identify a causal effect.

## Pearl rung 2: doing

The frozen interventions hold the checkpoint, messages, amplitudes, graph, and test examples fixed. Alignment removes relative phase; permutation preserves each centre's phase multiset while breaking its edge assignment; λ restores the learned deviations about the local circular mean. Their large, monotonic effects show that the selected Full functions rely on both relative phase and its edge assignment. The common-shift negative control passes, so the effect comes from relative phase rather than an arbitrary global phase origin.

## Pearl rung 3: imagining

Full and Aligned-Full use the same data, seeds, optimisation protocol, nominal parameter counts, Hermitian j,k closure, and validation selection rule. Aligned-Full fixes the imaginary coefficient to zero and makes only a 32-element phase head inactive (0.0178% of the MAL parameter count). The paired effect therefore estimates the learning-system consequence of permitting relative phase, including optimisation and checkpoint selection as mediators. CUDA deterministic algorithms were not forced, so this is a paired potential-outcome estimate rather than bitwise twin-world replay.

## Component decomposition

| Treatment vs control | Macro Force-MAE reduction | 95% CI | Wins | Exact sign-flip p | Interpretation |
|---|---:|---:|---:|---:|---|
| Diagonal vs Off | 9.07% | [4.87, 11.90] | 20/20 | 1.907e-06 | Stable self-density |
| Aligned vs Diagonal | 1.57% | [-2.40, 5.14] | 14/20 | 0.1613 | Phase-free cross-neighbour closure |
| Full vs Aligned | 0.61% | [-0.11, 1.30] | 16/20 | 0.06556 | Incremental learned phase |
| Full vs Off | 11.05% | [5.85, 15.78] | 20/20 | 1.907e-06 | Complete CHORUS path |
| Full vs Charge2 | -0.30% | [-1.65, 1.05] | 8/20 | 0.5524 | Sensitivity control; no panel-wide advantage |

## Identification boundary

The causal units are training runs or frozen predictors, not molecules. Results identify effects inside the fixed two-layer MACE-ICTC/CHORUS configuration and the four-dataset mechanism panel. Diagonal and Full contain more trainable parameters than Off, so Diagonal-versus-Off identifies the total effect of adding the self-density module; it does not isolate representation from active capacity. Full versus Aligned is much tighter: nominal counts and state schemas match, and only 32 phase-head elements become functionally inactive. The analysis does not assign physical meaning to θ, prove universal benefit across architectures, or establish relative phase as a uniquely necessary representation. The observational phase statistics were added after the intervention protocol and remain explicitly post hoc.

## Data and audit status

- Primary modes: 80/80 complete; aligned retraining: 20/20; frozen probes: 20/20; observational phase summaries: 20/20.
- Selection: validation Force MAE; test never selects a checkpoint.
- Atomic outcomes: 4,964,675; frame outcomes: 112,455; PyArrow 24.0.0.
- Input validation passed: **True**.
- Full and Aligned state schemas match: **True** (73 tensors; 181,131 stored elements each).
