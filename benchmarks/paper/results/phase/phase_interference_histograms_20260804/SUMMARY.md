# CHORUS phase-interference diagnostics

Relative-phase histograms use the complete validation split. Intervention metrics use the leading validation batches of the same split and the same trained checkpoint.

| Dataset | Relative pairs / channel | Destructive mass range | Force MAE change, zero phase | Force MAE change, local permutation | Common-shift mean |ΔF| |
|:--|--:|--:|--:|--:|--:|
| MD22 Buckyball | 43,170,961 | 0.0–4.9% | +522.2% | +58.1% | 0.0002 meV Å⁻¹ |
| xxMD MAL | 1,738,182 | 0.0–0.0% | +12.7% | +0.3% | 0.0002 meV Å⁻¹ |
| xxMD STI | 24,373,325 | 0.0–0.0% | +228.5% | +0.0% | 0.0002 meV Å⁻¹ |
| 3BPA 300 K | 189,197 | 0.0–47.9% | +5610.4% | +6140.6% | 0.0002 meV Å⁻¹ |
| Transition1x subset | 6,433,698 | 0.0–0.0% | +10.3% | +17.0% | 0.0002 meV Å⁻¹ |

Absolute-phase histograms are descriptive because a common U(1) shift changes θ but leaves the Hermitian density invariant. The relative-phase histograms and the intervention tests carry the mechanistic interpretation.
