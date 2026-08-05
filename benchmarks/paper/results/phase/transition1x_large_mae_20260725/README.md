# Transition1x large-model MAE-selected comparison

Reaction-disjoint 50k/10k/10k split, seed 20260616, 100k optimizer
steps, batch size 16, 5 Å cutoff, Energy:Force = 1:100, MakeFX, no EMA,
and TF32 disabled. All 32 validation checkpoints were retained. Each model
was selected by minimum validation Force MAE. A four-element constant energy
residual was then fitted on the training split only and frozen before the
validation and test evaluations.

## Train-only-calibrated results

| Split | Model | Energy MAE (meV/atom) | Energy RMSE (meV/atom) | Force MAE (meV/Å) | Force RMSE (meV/Å) |
|---|---|---:|---:|---:|---:|
| Validation | Phase-off MACE-ICTC | 24.560 | 39.400 | 92.232 | 203.104 |
| Validation | CHORUS | **19.904** | **31.070** | **85.542** | **196.440** |
| Test | Phase-off MACE-ICTC | 24.066 | 32.355 | 91.888 | 171.595 |
| Test | CHORUS | **20.540** | **28.498** | **82.513** | **164.296** |

Relative to phase-off, CHORUS reduces test Energy MAE by 14.7%, Energy
RMSE by 11.9%, Force MAE by 10.2%, and Force RMSE by 4.3%.

## Raw selected-checkpoint results

| Split | Model | Energy MAE (meV/atom) | Energy RMSE (meV/atom) | Force MAE (meV/Å) | Force RMSE (meV/Å) |
|---|---|---:|---:|---:|---:|
| Validation | Phase-off MACE-ICTC | 35.747 | 51.642 | 92.232 | 203.104 |
| Validation | CHORUS | **26.589** | **37.752** | **85.542** | **196.441** |
| Test | Phase-off MACE-ICTC | 34.855 | 46.130 | 91.888 | 171.595 |
| Test | CHORUS | **26.206** | **34.523** | **82.513** | **164.296** |

The force metrics are unchanged by calibration. State-dictionary comparison
confirmed that `element_energy_correction` was the only model tensor changed.
No validation or test structure was used to fit the correction.

## Evidence boundary

This is a single-seed comparison. CHORUS has 792,198 raw parameters versus
652,736 for phase-off (+21.4%), so it is a matched-backbone configuration
comparison rather than a strict parameter-matched comparison. The Force-MAE
checkpoint rule was fixed after inspecting an earlier validation trajectory
and then applied symmetrically to both reruns before any test evaluation.
