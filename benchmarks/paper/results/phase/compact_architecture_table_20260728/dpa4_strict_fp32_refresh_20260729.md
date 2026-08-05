# Strict-FP32 DPA-4 refresh

All DPA-4 entries below use `torch.compile` with strict FP32 training and
inference; TF32, AMP, and EMA are disabled. Checkpoints are selected by minimum
full-validation Force MAE, and test data never participate in selection.
Energy and Force MAE are reported in meV/atom and meV/Angstrom, respectively.

| Dataset | CHORUS R16 E/F | DPA-4 C32 FP32 E/F | DPA-4 C48 FP32 E/F | TECE-C36 E/F | TECE-C48 E/F |
|---|---:|---:|---:|---:|---:|
| Transition1x | 20.540 / 82.513 | 15.502* / 88.696 | running | 19.917 / 94.744 | 17.297 / 88.289 |
| xxMD MAL | 12.766 / 159.062 | 18.676 / 225.660 | running | 18.408 / 216.259 | 23.809 / 210.597 |
| MD22 Buckyball | 0.237 / 6.724 | 0.201 / 5.915 | running | 0.233 / 12.079 | 0.240 / 11.315 |
| 3BPA 300 K | 0.154 / 9.096 | 0.189 / 15.963 | 0.148 / 14.032 | 0.220 / 12.799 | 0.229 / 11.868 |
| 3BPA 600 K | 0.360 / 24.216 | 0.592 / 39.236 | 0.644 / 38.678 | 0.500 / 26.515 | 0.465 / 24.487 |
| 3BPA 1200 K | 3.348 / 119.618 | 3.830 / 130.431 | 5.291 / 150.605 | 1.584 / 68.464 | 1.271 / 62.040 |

\* The refreshed DPA-4 C32 Transition1x energy is the raw test MAE. The
train-only element-energy residual calibration required by the common
Transition1x protocol has not yet been applied; Force MAE is unaffected.

The completed strict-FP32 C32 result for xxMD STI is E/F =
13.822 / 144.155. It is kept outside this compact table until the matching
cross-model STI row is regenerated from the same result manifest.

Sources:

- `/home/ylzhang/chorus_runs/dpa4_fp32_compiled_all_20260728/c32_mix3`
- `/home/ylzhang/chorus_runs/dpa4_fp32_compiled_all_20260728/c48_mix3`
- `results.json` in this directory for the unchanged CHORUS and TECE entries
