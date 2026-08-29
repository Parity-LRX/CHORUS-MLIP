# MS25 solid-state external-model benchmark

All values use the official Set1 split. Checkpoints are selected by full-validation Force MAE; test does not select. Energy is meV/atom and force is meV/Å.

For the audited interpretation and model-family comparisons, see `SUMMARY.md`.

## hea

| Model | Parameters | E MAE | E RMSE | F MAE | F RMSE | Selected step |
|:--|--:|--:|--:|--:|--:|--:|
| DPA-4 C32 | 650,818 | 9.34 | 12.29 | 7.71 | 10.94 | 41,250 |
| DPA-4 C48 | 1,297,210 | 9.30 | 12.27 | 7.76 | 10.82 | 30,500 |
| MACE-CHORUS-Final C128 R16 | 852,230 | 10.11 | 12.77 | 6.75 | 9.19 | 42,750 |
| MACE-CHORUS-Persistent C128 R16 | 1,112,527 | 10.33 | 13.06 | 6.90 | 9.48 | 40,500 |
| MACE-ICTC C128 | 690,240 | 10.40 | 13.07 | 6.74 | 9.60 | 43,250 |
| MACE-SH C128 | 688,448 | 18.12 | 22.58 | 21.75 | 28.57 | 45,000 |
| NequIP-CHORUS-Final C84 R16 | 1,033,187 | 9.31 | 12.23 | 9.61 | 12.72 | 44,500 |
| NequIP-CHORUS-Persistent C84 R16 | 1,118,411 | 9.62 | 12.49 | 9.18 | 12.13 | 44,000 |
| NequIP-SH C84 | 726,398 | 9.55 | 12.37 | 12.20 | 16.32 | 44,750 |
| TECE C36 | 916,989 | 9.47 | 12.37 | 7.98 | 10.54 | 42,500 |
| TECE C48 | 1,599,225 | 9.23 | 12.18 | 7.73 | 10.26 | 44,000 |

## zr-o

| Model | Parameters | E MAE | E RMSE | F MAE | F RMSE | Selected step |
|:--|--:|--:|--:|--:|--:|--:|
| DPA-4 C32 | 650,689 | 24.40 | 50.80 | 81.27 | 149.81 | 41,500 |
| DPA-4 C48 | 1,297,021 | 21.40 | 46.04 | 73.91 | 142.07 | 44,750 |
| MACE-CHORUS-Final C128 R16 | 672,134 | 33.86 | 60.76 | 64.80 | 122.93 | 37,848 |
| MACE-CHORUS-Persistent C128 R16 | 864,847 | 33.79 | 60.61 | 64.39 | 122.63 | 38,346 |
| MACE-ICTC C128 | 577,728 | 34.00 | 63.26 | 68.12 | 129.18 | 39,342 |
| MACE-SH C128 | 575,936 | 161.09 | 262.01 | 131.78 | 182.01 | 45,000 |
| NequIP-CHORUS-Final C84 R16 | 698,279 | 42.92 | 70.30 | 76.40 | 128.49 | 43,500 |
| NequIP-CHORUS-Persistent C84 R16 | 739,151 | 45.00 | 75.49 | 75.81 | 132.15 | 32,000 |
| NequIP-SH C84 | 556,802 | 38.23 | 66.51 | 84.03 | 134.57 | 44,250 |
| TECE C36 | 639,645 | 35.33 | 74.68 | 73.11 | 129.15 | 35,500 |
| TECE C48 | 1,106,745 | 35.66 | 78.20 | 69.38 | 129.45 | 41,750 |

## mgo-2x2

| Model | Parameters | E MAE | E RMSE | F MAE | F RMSE | Selected step |
|:--|--:|--:|--:|--:|--:|--:|
| DPA-4 C32 | 650,689 | 0.05 | 0.06 | 3.03 | 4.59 | 45,000 |
| DPA-4 C48 | 1,297,021 | 0.04 | 0.05 | 2.56 | 3.91 | 44,750 |
| MACE-CHORUS-Final C128 R16 | 672,134 | 0.09 | 0.10 | 3.68 | 4.83 | 41,250 |
| MACE-CHORUS-Persistent C128 R16 | 864,847 | 0.08 | 0.10 | 3.69 | 4.83 | 41,250 |
| MACE-ICTC C128 | 577,728 | 0.19 | 0.20 | 3.95 | 5.18 | 42,250 |
| MACE-SH C128 | 575,936 | 0.24 | 0.28 | 9.79 | 15.18 | 45,000 |
| NequIP-CHORUS-Final C84 R16 | 698,279 | 0.06 | 0.07 | 4.54 | 6.17 | 44,750 |
| NequIP-CHORUS-Persistent C84 R16 | 739,151 | 0.06 | 0.07 | 4.63 | 6.31 | 41,750 |
| NequIP-SH C84 | 556,802 | 1.40 | 1.40 | 5.18 | 7.28 | 43,250 |
| TECE C36 | 639,645 | 0.04 | 0.06 | 4.79 | 6.81 | 44,250 |
| TECE C48 | 1,106,745 | 0.08 | 0.10 | 4.73 | 6.62 | 42,500 |

## MgO 2×2 → 4×4 size transfer

| Model | E MAE | E RMSE | F MAE | F RMSE |
|:--|--:|--:|--:|--:|
| DPA-4 C32 | 0.12 | 0.15 | 29.18 | 37.33 |
| DPA-4 C48 | 0.13 | 0.17 | 34.44 | 44.51 |
| MACE-CHORUS-Final C128 R16 | 0.17 | 0.21 | 34.25 | 44.23 |
| MACE-CHORUS-Persistent C128 R16 | 0.21 | 0.25 | 32.01 | 41.15 |
| MACE-ICTC C128 | 0.19 | 0.23 | 33.92 | 43.59 |
| MACE-SH C128 | 0.32 | 0.36 | 37.43 | 48.50 |
| NequIP-CHORUS-Final C84 R16 | 0.34 | 0.39 | 37.84 | 48.42 |
| NequIP-CHORUS-Persistent C84 R16 | 0.16 | 0.20 | 37.01 | 47.56 |
| NequIP-SH C84 | 1.41 | 1.41 | 16.89 | 23.33 |
| TECE C36 | 0.23 | 0.28 | 37.73 | 48.65 |
| TECE C48 | 0.16 | 0.20 | 38.50 | 49.33 |
