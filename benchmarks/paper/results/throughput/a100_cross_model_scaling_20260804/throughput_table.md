# A100 atom-count scaling

Strict FP32 with TF32 disabled; fixed 32 directed neighbors per atom. Inference computes energy and conservative forces. Training includes the energy-force loss, backward pass, and optimizer update. Compilation and graph preparation are excluded from steady-state throughput.
DPA-4 constructs its neighbor representation inside the timed model call; the MACE, NequIP, and TECE inputs use prebuilt fixed-degree graphs.

## Configurations

| Model | Parameters | Backend |
|:--|--:|:--|
| MACE-SH | 650,944 | CuEq-only (native MACE has no compatible MakeFX force path) |
| MACE-ICTC | 652,736 | MakeFX/Inductor |
| CHORUS-Final R8 | 744,070 | MakeFX/Inductor |
| CHORUS-Final R16 | 792,198 | MakeFX/Inductor |
| CHORUS-Final R32 | 888,454 | MakeFX/Inductor |
| CHORUS-Persistent R16 | 1,029,967 | MakeFX/Inductor |
| NequIP-SH | 669,866 | OpenEquivariance 0.6.8 precompiled AOTI |
| NequIP-CHORUS-Final | 921,551 | OpenEquivariance 0.6.8 precompiled AOTI |
| NequIP-CHORUS-Persistent | 991,991 | OpenEquivariance 0.6.8 precompiled AOTI |
| DPA-4 C32 | 650,428 | DPA-4 internal torch.compile (model.use_compile=true) |
| DPA-4 C48 | 1,296,800 | DPA-4 internal torch.compile (model.use_compile=true) |
| TECE C36 | 824,541 | TECE OpenEquivariance |
| TECE C48 | 1,435,065 | TECE OpenEquivariance |

## Inference throughput (atoms/s)

| Model | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|:--|--:|--:|--:|--:|--:|--:|
| MACE-SH | 6,969 | 14,053 | 28,053 | 55,953 | 110,791 | 149,849 |
| MACE-ICTC | 21,451 | 25,105 | 29,119 | 31,308 | 32,199 | 32,478 |
| CHORUS-Final R8 | 17,285 | 19,775 | 22,673 | 24,111 | 24,383 | 24,414 |
| CHORUS-Final R16 | 17,446 | 19,512 | 22,258 | 23,553 | 24,012 | 23,840 |
| CHORUS-Final R32 | 17,117 | 19,179 | 21,922 | 23,257 | 23,729 | 23,554 |
| CHORUS-Persistent R16 | 14,226 | 17,514 | 20,593 | 22,090 | 22,633 | 22,669 |
| NequIP-SH | 17,084 | 34,104 | 66,679 | 92,882 | 104,130 | 111,681 |
| NequIP-CHORUS-Final | 12,587 | 25,035 | 39,257 | 46,596 | 50,508 | 52,633 |
| NequIP-CHORUS-Persistent | 9,259 | 18,389 | 30,915 | 38,534 | 42,175 | 44,088 |
| DPA-4 C32 | 4,773 | 9,537 | 19,004 | 32,529 | 38,626 | 42,360 |
| DPA-4 C48 | 4,635 | 9,175 | 16,033 | 22,234 | 24,788 | 26,206 |
| TECE C36 | 3,836 | 7,585 | 11,580 | 14,151 | 15,985 | 16,750 |
| TECE C48 | 3,819 | 7,357 | 9,743 | 11,567 | 12,627 | 13,078 |

## Training throughput (atoms/s)

| Model | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|:--|--:|--:|--:|--:|--:|--:|
| MACE-SH | 3,076 | 6,183 | 12,414 | 24,711 | 43,636 | 54,336 |
| MACE-ICTC | 8,317 | 10,269 | 11,738 | 12,196 | 12,386 | 12,377 |
| CHORUS-Final R8 | 6,256 | 7,766 | 8,794 | 9,007 | 9,082 | OOM |
| CHORUS-Final R16 | 6,171 | 7,648 | 8,606 | 8,864 | 8,896 | OOM |
| CHORUS-Final R32 | 6,080 | 7,558 | 8,481 | 8,737 | 8,765 | OOM |
| CHORUS-Persistent R16 | 5,376 | 6,788 | 7,635 | 8,103 | 8,262 | OOM |
| NequIP-SH | 6,531 | 13,067 | 23,933 | 32,106 | 35,423 | 37,539 |
| NequIP-CHORUS-Final | 4,725 | 9,437 | 14,631 | 17,053 | 18,470 | 19,233 |
| NequIP-CHORUS-Persistent | 3,481 | 6,944 | 11,653 | 13,957 | 15,519 | 16,279 |
| DPA-4 C32 | 6,483 | 11,297 | 15,232 | 18,093 | 19,440 | 20,630 |
| DPA-4 C48 | 5,920 | 8,454 | 10,706 | 12,389 | 13,193 | 13,747 |
| TECE C36 | 1,408 | 2,821 | 4,773 | 5,672 | 6,303 | 6,634 |
| TECE C48 | 1,396 | 2,794 | 4,030 | 4,612 | 4,993 | 5,172 |

## Inference peak memory (GiB)

| Model | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|:--|--:|--:|--:|--:|--:|--:|
| MACE-SH | 0.08 | 0.14 | 0.27 | 0.52 | 1.01 | 1.99 |
| MACE-ICTC | 0.56 | 1.32 | 2.62 | 5.22 | 10.41 | 20.78 |
| CHORUS-Final R8 | 0.57 | 1.93 | 3.86 | 7.67 | 15.32 | 30.59 |
| CHORUS-Final R16 | 0.57 | 1.95 | 3.88 | 7.73 | 15.94 | 30.81 |
| CHORUS-Final R32 | 0.57 | 1.97 | 3.94 | 7.84 | 15.64 | 32.25 |
| CHORUS-Persistent R16 | 0.61 | 2.09 | 4.17 | 8.30 | 16.56 | 33.07 |
| NequIP-SH | 0.16 | 0.30 | 0.58 | 1.15 | 2.27 | 4.51 |
| NequIP-CHORUS-Final | 0.23 | 0.44 | 0.85 | 1.67 | 3.32 | 6.59 |
| NequIP-CHORUS-Persistent | 0.24 | 0.46 | 0.89 | 1.76 | 3.48 | 6.92 |
| DPA-4 C32 | 0.21 | 0.41 | 0.77 | 1.50 | 2.94 | 5.84 |
| DPA-4 C48 | 0.30 | 0.57 | 1.08 | 2.09 | 4.13 | 8.21 |
| TECE C36 | 0.27 | 0.50 | 0.96 | 1.88 | 3.72 | 7.36 |
| TECE C48 | 0.37 | 0.70 | 1.33 | 2.58 | 5.11 | 10.17 |

## Training peak memory (GiB)

| Model | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|:--|--:|--:|--:|--:|--:|--:|
| MACE-SH | 0.15 | 0.27 | 0.52 | 1.01 | 1.99 | 3.93 |
| MACE-ICTC | 0.95 | 1.87 | 3.77 | 7.43 | 14.80 | 29.52 |
| CHORUS-Final R8 | 1.34 | 2.62 | 5.19 | 10.32 | 20.51 | OOM |
| CHORUS-Final R16 | 1.39 | 2.64 | 5.24 | 10.37 | 20.62 | OOM |
| CHORUS-Final R32 | 1.37 | 2.66 | 5.28 | 10.49 | 20.84 | OOM |
| CHORUS-Persistent R16 | 1.52 | 2.92 | 5.78 | 11.44 | 22.75 | OOM |
| NequIP-SH | 0.29 | 0.55 | 1.08 | 2.14 | 4.23 | 8.42 |
| NequIP-CHORUS-Final | 0.45 | 0.87 | 1.72 | 3.40 | 6.76 | 13.47 |
| NequIP-CHORUS-Persistent | 0.50 | 0.97 | 1.91 | 3.78 | 7.51 | 14.96 |
| DPA-4 C32 | 0.44 | 0.82 | 1.59 | 3.15 | 6.16 | 12.31 |
| DPA-4 C48 | 0.61 | 1.17 | 2.30 | 4.48 | 8.88 | 17.68 |
| TECE C36 | 0.62 | 1.21 | 2.35 | 4.64 | 9.24 | 18.34 |
| TECE C48 | 0.86 | 1.66 | 3.24 | 6.36 | 12.67 | 25.25 |
