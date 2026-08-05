# Cross-model atom-scaling benchmark

One RTX 4090; every model uses strict FP32, with TF32, AMP, and EMA disabled. Inference computes energy and conservative forces. Training includes the energy-force loss, backward pass, and optimizer update. Compile/preparation time is excluded from steady-state throughput.

Native MACE uses CuEq-only tensor products; TECE uses OpenEquivariance. MACE-family and TECE graph construction is excluded. DPA-4 uses its standard compiled model interface, whose internal neighbor-list work remains inside the timed call; this interface difference must be retained as a caveat.

## Configurations

| Model | Trainable parameters | Backend |
|---|---:|---|
| Native MACE · CuEq | 650,944 | CuEq-only (native MACE has no compatible MakeFX force path) |
| MACE-ICTC | 652,736 | MakeFX/Inductor |
| CHORUS R8 | 744,070 | MakeFX/Inductor |
| CHORUS R16 | 792,198 | MakeFX/Inductor |
| CHORUS R32 | 888,454 | MakeFX/Inductor |
| DPA-4 C32 · FP32 | 650,428 | DPA-4 internal torch.compile (model.use_compile=true) |
| DPA-4 C48 · FP32 | 1,296,800 | DPA-4 internal torch.compile (model.use_compile=true) |
| TECE C36 · OpenEq | 824,541 | TECE OpenEquivariance |
| TECE C48 · OpenEq | 1,435,065 | TECE OpenEquivariance |

## Compilation and peak memory

| Model | Compile at 32 atoms, infer (s) | Compile at 32 atoms, train (s) | Peak memory at 2048, infer (GiB) | Peak memory at 2048, train (GiB) |
|---|---:|---:|---:|---:|
| Native MACE · CuEq | 0.000 | 0.000 | 1.009 | 1.985 |
| MACE-ICTC | 25.537 | 47.475 | 10.312 | 14.432 |
| CHORUS R8 | 32.418 | 61.569 | 15.225 | 20.185 |
| CHORUS R16 | 15.270 | 23.329 | 15.329 | 20.166 |
| CHORUS R32 | 34.628 | 58.738 | 15.543 | 20.521 |
| DPA-4 C32 · FP32 | 3.081 | 124.535 | 2.935 | 6.299 |
| DPA-4 C48 · FP32 | 3.073 | 123.448 | 4.135 | 9.046 |
| TECE C36 · OpenEq | 0.000 | 0.000 | 3.715 | 9.237 |
| TECE C48 · OpenEq | 0.000 | 0.000 | 5.114 | 12.668 |

## Inference atoms/s

| Model | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native MACE · CuEq | 1,438 | 2,921 | 5,858 | 11,717 | 23,470 | 46,841 | 93,397 |
| MACE-ICTC | 7,681 | 12,046 | 22,721 | 27,799 | 30,616 | 35,263 | 33,345 |
| CHORUS R16 | 6,066 | 6,397 | 14,850 | 22,733 | 24,668 | 30,516 | 26,291 |
| DPA-4 C32 · FP32 | 1,004 | 1,980 | 3,970 | 7,949 | 15,932 | 29,763 | 37,523 |
| DPA-4 C48 · FP32 | 1,003 | 1,981 | 3,973 | 7,923 | 15,912 | 25,500 | 27,507 |
| TECE C36 · OpenEq | 760 | 1,508 | 3,010 | 6,012 | 11,998 | 15,357 | 16,214 |
| TECE C48 · OpenEq | 760 | 1,511 | 3,019 | 6,017 | 10,911 | 12,255 | 12,500 |

## Training atoms/s

| Model | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native MACE · CuEq | 652 | 1,311 | 2,630 | 5,239 | 10,506 | 20,876 | 41,842 |
| MACE-ICTC | 2,691 | 5,374 | 10,076 | 11,885 | 12,636 | 12,855 | 12,521 |
| CHORUS R16 | 1,611 | 3,176 | 6,793 | 8,593 | 9,048 | 9,026 | 8,758 |
| DPA-4 C32 · FP32 | 1,482 | 2,952 | 5,924 | 11,902 | 19,312 | 21,716 | 21,729 |
| DPA-4 C48 · FP32 | 1,499 | 2,978 | 5,993 | 10,943 | 15,139 | 15,949 | 16,071 |
| TECE C36 · OpenEq | 282 | 564 | 1,126 | 2,253 | 4,486 | 6,117 | 6,228 |
| TECE C48 · OpenEq | 283 | 566 | 1,129 | 2,250 | 4,210 | 4,752 | 4,711 |
