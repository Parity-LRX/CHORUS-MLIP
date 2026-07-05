# Operator benchmark: MACE-ICTC ICTC product vs cartnn Cartesian tensor product

**RTX 4090 (D), torch 2.7.1+cu128, e3nn 0.5.9, cartnn 0.5.8 @ 4d0dc38, MACE-ICTC local git 414aa25. TF32 disabled.**

## What is being compared (and what is NOT)

The matched operator is the **equivariant tensor product** that couples a hidden node feature (degrees `0..hidden_lmax`, `C` channels) with the edge angular embedding (degrees `0..max_ell`), per-edge weighted, over a batch of `E` directed edges — i.e. the MACE convolution tensor product. All backends run the **identical `(l1,l2,l3)` natural-parity path set** and the same per-edge weight count (`num_paths*C`):

| backend | operator | basis / storage | fusion |
|---|---|---|---|
| `e3nn` (reference) | `e3nn.o3.TensorProduct` (wigner_3j) | spherical, `2l+1` | opt_einsum_fx codegen |
| `cartnn` | `cartnn.o3.TensorProduct` (cartesian_3j) | **full Cartesian, `3**l`** | opt_einsum_fx codegen |
| `ictd` | `EdgeWeightedPathPreservingTensorProduct` | irreducible-Cartesian (ICTC), `2l+1` | **eager** (Python per-path) |
| `ictd_compiled` | same ICTC op under `torch.compile` | ICTC `2l+1` | torch.compile (deployed form) |

**Caveats (do not over-read):**
- This is an *operator-level comparable workload*, **not** an exact apples-to-apples comparison: cartnn stores a degree-`l` tensor in `3**l` components (vs `2l+1`), and the per-path normalizations differ. Numerical outputs are **not** expected to match across backends.
- cartnn ships **no symmetric-contraction operator** (the authors declined to implement ICTC), so the MACE symmetric contraction is **out of scope** here; only the binary tensor product is compared.
- `e3nn`/`cartnn` `TensorProduct` are **codegen-fused**; the bare `ictd` operator is timed in **eager** mode (Python per-path overhead, dominant at small sizes). The deployed MACE-ICTC model removes this via AOTI/`torch.compile` — see `ictd_compiled` below and the existing model-level throughput benchmarks. Read `ictd` eager numbers as a lower bound on the deployed ICTC speed.
- No chemical-accuracy or model-level superiority is claimed or measured here.

## Headline (channels=64, edges=100000)

`total_ms` = forward (+backward). Speedup `>1` ⇒ row backend faster than cartnn.

### float32, forward_only

| config (hid/ell) | e3nn total_ms | cartnn total_ms | ictd total_ms | ictd_compiled total_ms | ictd/cartnn | ictd_comp/cartnn |
|---|---|---|---|---|---|---|
| 1/1 | 1.341 | 1.342 | 6.153 | 3.575 | 0.22 | 0.38 |
| 1/2 | 3.798 | 3.802 | 9.427 | 9.522 | 0.40 | 0.40 |
| 2/2 | 13.512 | 16.407 | 28.991 | 29.161 | 0.57 | 0.56 |
| 2/3 | 18.727 | 22.502 | 37.962 | 38.158 | 0.59 | 0.59 |
| 3/3 | 34.785 | 97.317 | 81.310 | 81.512 | 1.20 | 1.19 |

### float32, forward_backward

| config (hid/ell) | e3nn total_ms | cartnn total_ms | ictd total_ms | ictd_compiled total_ms | ictd/cartnn | ictd_comp/cartnn |
|---|---|---|---|---|---|---|
| 1/1 | 7.681 | 7.679 | 16.538 | 12.606 | 0.46 | 0.61 |
| 1/2 | 13.512 | 13.508 | 25.133 | 25.227 | 0.54 | 0.54 |
| 2/2 | 50.803 | 62.615 | 84.975 | 84.911 | 0.74 | 0.74 |
| 2/3 | 67.537 | 81.622 | 113.384 | 113.448 | 0.72 | 0.72 |
| 3/3 | 174.743 | — | 267.859 | 267.766 | n/a | n/a |

### float64, forward_only

| config (hid/ell) | e3nn total_ms | cartnn total_ms | ictd total_ms | ictd_compiled total_ms | ictd/cartnn | ictd_comp/cartnn |
|---|---|---|---|---|---|---|
| 1/1 | 2.645 | 2.641 | 30.145 | 30.199 | 0.09 | 0.09 |
| 1/2 | 10.532 | 10.520 | 41.623 | 41.616 | 0.25 | 0.25 |
| 2/2 | 38.798 | 43.294 | 105.351 | 105.483 | 0.41 | 0.41 |
| 2/3 | 55.434 | 60.983 | 150.474 | 150.451 | 0.41 | 0.41 |
| 3/3 | 173.793 | — | — | 287.017 | n/a | n/a |

### float64, forward_backward

| config (hid/ell) | e3nn total_ms | cartnn total_ms | ictd total_ms | ictd_compiled total_ms | ictd/cartnn | ictd_comp/cartnn |
|---|---|---|---|---|---|---|
| 1/1 | 15.875 | 15.880 | 63.518 | 63.561 | 0.25 | 0.25 |
| 1/2 | 43.517 | 43.663 | 92.055 | 92.452 | 0.47 | 0.47 |
| 2/2 | 179.758 | 229.929 | 264.068 | 265.266 | 0.87 | 0.87 |
| 2/3 | 251.243 | 319.996 | 357.313 | 356.939 | 0.90 | 0.90 |
| 3/3 | — | — | — | — | n/a | n/a |

## Throughput vs directed edges (channels=64, edges/s)

### float32, forward_only

| config | backend | E=10000 | E=50000 | E=100000 | E=500000 |
|---|---|---|---|---|---|
| 1/1 | e3nn | 42.46M | 74.00M | 74.55M | 72.29M |
| 1/1 | cartnn | 42.59M | 77.40M | 74.50M | 72.31M |
| 1/1 | ictd | 18.82M | 18.03M | 16.25M | 11.26M |
| 1/1 | ictd_compiled | 26.68M | 28.69M | 27.97M | 28.01M |
| 1/2 | e3nn | 19.53M | 27.14M | 26.33M | 25.83M |
| 1/2 | cartnn | 19.57M | 27.15M | 26.30M | 25.81M |
| 1/2 | ictd | 14.25M | 12.04M | 10.61M | 8.01M |
| 1/2 | ictd_compiled | 12.74M | 11.83M | 10.50M | 7.97M |
| 2/2 | e3nn | 6.82M | 7.63M | 7.40M | 7.38M |
| 2/2 | cartnn | 5.77M | 6.27M | 6.09M | 6.07M |
| 2/2 | ictd | 4.52M | 3.70M | 3.45M | — |
| 2/2 | ictd_compiled | 4.35M | 3.68M | 3.43M | 2.75M |
| 2/3 | e3nn | 5.08M | 5.45M | 5.34M | 5.31M |
| 2/3 | cartnn | 4.29M | 4.47M | 4.44M | — |
| 2/3 | ictd | 3.33M | 2.79M | 2.63M | — |
| 2/3 | ictd_compiled | 3.23M | 2.79M | 2.62M | — |
| 3/3 | e3nn | 2.88M | 2.89M | 2.87M | — |
| 3/3 | cartnn | 1.04M | 1.03M | 1.03M | — |
| 3/3 | ictd | 1.53M | 1.29M | 1.23M | — |
| 3/3 | ictd_compiled | 1.51M | 1.29M | 1.23M | — |

### float32, forward_backward

| config | backend | E=10000 | E=50000 | E=100000 | E=500000 |
|---|---|---|---|---|---|
| 1/1 | e3nn | 10.24M | 14.51M | 13.02M | 12.49M |
| 1/1 | cartnn | 10.29M | 14.47M | 13.02M | 12.49M |
| 1/1 | ictd | 6.28M | 6.80M | 6.05M | 4.61M |
| 1/1 | ictd_compiled | 8.96M | 8.76M | 7.93M | 7.18M |
| 1/2 | e3nn | 7.76M | 7.89M | 7.40M | 7.20M |
| 1/2 | cartnn | 7.74M | 7.87M | 7.40M | 7.20M |
| 1/2 | ictd | 5.21M | 4.38M | 3.98M | 3.19M |
| 1/2 | ictd_compiled | 4.93M | 4.35M | 3.96M | 3.18M |
| 2/2 | e3nn | 2.38M | 2.00M | 1.97M | — |
| 2/2 | cartnn | 1.90M | 1.62M | 1.60M | — |
| 2/2 | ictd | 1.61M | 1.24M | 1.18M | — |
| 2/2 | ictd_compiled | 1.58M | 1.24M | 1.18M | — |
| 2/3 | e3nn | 1.78M | 1.50M | 1.48M | — |
| 2/3 | cartnn | 1.44M | 1.23M | 1.23M | — |
| 2/3 | ictd | 1.16M | 0.92M | 0.88M | — |
| 2/3 | ictd_compiled | 1.15M | 0.92M | 0.88M | — |
| 3/3 | e3nn | 0.63M | 0.57M | 0.57M | — |
| 3/3 | cartnn | 0.25M | 0.12M | — | — |
| 3/3 | ictd | 0.46M | 0.39M | 0.37M | — |
| 3/3 | ictd_compiled | 0.46M | 0.39M | 0.37M | — |

### float64, forward_only

| config | backend | E=10000 | E=50000 | E=100000 | E=500000 |
|---|---|---|---|---|---|
| 1/1 | e3nn | 36.27M | 39.25M | 37.81M | 37.47M |
| 1/1 | cartnn | 36.44M | 39.27M | 37.87M | 37.52M |
| 1/1 | ictd | 3.74M | 3.40M | 3.32M | 3.10M |
| 1/1 | ictd_compiled | 3.64M | 3.38M | 3.31M | 3.09M |
| 1/2 | e3nn | 9.31M | 9.60M | 9.50M | 9.50M |
| 1/2 | cartnn | 9.17M | 9.59M | 9.51M | 9.50M |
| 1/2 | ictd | 2.65M | 2.45M | 2.40M | 2.25M |
| 1/2 | ictd_compiled | 2.61M | 2.45M | 2.40M | 2.25M |
| 2/2 | e3nn | 2.68M | 2.58M | 2.58M | — |
| 2/2 | cartnn | 2.39M | 2.32M | 2.31M | — |
| 2/2 | ictd | 1.06M | 0.96M | 0.95M | — |
| 2/2 | ictd_compiled | 1.04M | 0.96M | 0.95M | — |
| 2/3 | e3nn | 1.89M | 1.80M | 1.80M | — |
| 2/3 | cartnn | 1.71M | 1.64M | 1.64M | — |
| 2/3 | ictd | 0.72M | 0.67M | 0.66M | — |
| 2/3 | ictd_compiled | 0.72M | 0.67M | 0.66M | — |
| 3/3 | e3nn | 0.58M | 0.57M | 0.58M | — |
| 3/3 | cartnn | 0.17M | 0.17M | — | — |
| 3/3 | ictd | 0.37M | 0.35M | — | — |
| 3/3 | ictd_compiled | 0.38M | 0.35M | 0.35M | — |

### float64, forward_backward

| config | backend | E=10000 | E=50000 | E=100000 | E=500000 |
|---|---|---|---|---|---|
| 1/1 | e3nn | 9.01M | 6.62M | 6.30M | 6.31M |
| 1/1 | cartnn | 9.17M | 6.62M | 6.30M | 6.33M |
| 1/1 | ictd | 1.89M | 1.62M | 1.57M | 1.47M |
| 1/1 | ictd_compiled | 1.87M | 1.62M | 1.57M | 1.47M |
| 1/2 | e3nn | 2.66M | 2.32M | 2.30M | 2.29M |
| 1/2 | cartnn | 2.61M | 2.32M | 2.29M | 2.29M |
| 1/2 | ictd | 1.26M | 1.11M | 1.09M | — |
| 1/2 | ictd_compiled | 1.26M | 1.11M | 1.08M | — |
| 2/2 | e3nn | 0.59M | 0.56M | 0.56M | — |
| 2/2 | cartnn | 0.45M | 0.44M | 0.43M | — |
| 2/2 | ictd | 0.42M | 0.38M | 0.38M | — |
| 2/2 | ictd_compiled | 0.42M | 0.38M | 0.38M | — |
| 2/3 | e3nn | 0.42M | 0.40M | 0.40M | — |
| 2/3 | cartnn | 0.32M | 0.31M | 0.31M | — |
| 2/3 | ictd | 0.31M | 0.28M | 0.28M | — |
| 2/3 | ictd_compiled | 0.31M | 0.28M | 0.28M | — |
| 3/3 | e3nn | 0.17M | 0.17M | — | — |
| 3/3 | cartnn | 0.07M | — | — | — |
| 3/3 | ictd | 0.14M | 0.14M | — | — |
| 3/3 | ictd_compiled | 0.15M | 0.14M | — | — |

## Channel scaling (edges=100000, forward+backward, total_ms)

### float32

| config | backend | C=32 | C=64 | C=128 |
|---|---|---|---|---|
| 1/1 | e3nn | 3.47 | 7.68 | 16.06 |
| 1/1 | cartnn | 3.48 | 7.68 | 16.04 |
| 1/1 | ictd | 7.38 | 16.54 | 35.44 |
| 1/1 | ictd_compiled | — | 12.61 | — |
| 1/2 | e3nn | 7.33 | 13.51 | 25.68 |
| 1/2 | cartnn | 7.35 | 13.51 | 25.69 |
| 1/2 | ictd | 11.54 | 25.13 | 53.31 |
| 1/2 | ictd_compiled | — | 25.23 | — |
| 2/2 | e3nn | 28.92 | 50.80 | 94.20 |
| 2/2 | cartnn | 35.27 | 62.62 | 117.03 |
| 2/2 | ictd | 40.58 | 84.97 | 175.10 |
| 2/2 | ictd_compiled | — | 84.91 | — |
| 2/3 | e3nn | 39.19 | 67.54 | 124.05 |
| 2/3 | cartnn | 47.03 | 81.62 | 150.74 |
| 2/3 | ictd | 54.33 | 113.38 | 233.31 |
| 2/3 | ictd_compiled | — | 113.45 | — |
| 3/3 | e3nn | 92.43 | 174.74 | — |
| 3/3 | cartnn | 346.46 | — | — |
| 3/3 | ictd | 130.88 | 267.86 | — |
| 3/3 | ictd_compiled | — | 267.77 | — |

### float64

| config | backend | C=32 | C=64 | C=128 |
|---|---|---|---|---|
| 1/1 | e3nn | 7.57 | 15.87 | 31.67 |
| 1/1 | cartnn | 7.57 | 15.88 | 31.67 |
| 1/1 | ictd | 30.93 | 63.52 | 129.41 |
| 1/1 | ictd_compiled | — | 63.56 | — |
| 1/2 | e3nn | 21.94 | 43.52 | 86.95 |
| 1/2 | cartnn | 21.96 | 43.66 | 86.97 |
| 1/2 | ictd | 45.22 | 92.05 | 187.50 |
| 1/2 | ictd_compiled | — | 92.45 | — |
| 2/2 | e3nn | 90.72 | 179.76 | 358.43 |
| 2/2 | cartnn | 116.04 | 229.93 | — |
| 2/2 | ictd | 130.89 | 264.07 | — |
| 2/2 | ictd_compiled | — | 265.27 | — |
| 2/3 | e3nn | 127.81 | 251.24 | — |
| 2/3 | cartnn | 162.13 | 320.00 | — |
| 2/3 | ictd | 176.80 | 357.31 | — |
| 2/3 | ictd_compiled | — | 356.94 | — |
| 3/3 | e3nn | 298.06 | — | — |
| 3/3 | cartnn | — | — | — |
| 3/3 | ictd | 368.56 | — | — |
| 3/3 | ictd_compiled | — | — | — |

## OOM / error cells

| backend | config | channels | edges | dtype | mode | status | error |
|---|---|---|---|---|---|---|---|
| ictd | 1/1 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.86 GiB. GPU 0 has a tot |
| e3nn | 1/1 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 3.82 GiB. GPU 0 has a tot |
| cartnn | 1/1 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 3.82 GiB. GPU 0 has a tot |
| ictd | 1/1 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.86 GiB. GPU 0 has a tot |
| ictd | 1/2 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.15 GiB. GPU 0 has a tot |
| ictd | 1/2 | 128 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.15 GiB. GPU 0 has a tot |
| ictd | 1/2 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 7.15 GiB. GPU 0 has a tot |
| e3nn | 1/2 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.25 GiB. GPU 0 has a tot |
| cartnn | 1/2 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.25 GiB. GPU 0 has a tot |
| ictd | 1/2 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 7.15 GiB. GPU 0 has a tot |
| ictd | 2/2 | 32 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| e3nn | 2/2 | 32 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.17 GiB. GPU 0 has a tot |
| cartnn | 2/2 | 32 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 6.08 GiB. GPU 0 has a tot |
| ictd | 2/2 | 32 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| ictd | 2/2 | 64 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| e3nn | 2/2 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.17 GiB. GPU 0 has a tot |
| cartnn | 2/2 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 6.08 GiB. GPU 0 has a tot |
| ictd | 2/2 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| e3nn | 2/2 | 64 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/2 | 64 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/2 | 64 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| e3nn | 2/2 | 64 | 500000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/2 | 64 | 500000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/2 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| e3nn | 2/2 | 128 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/2 | 128 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/2 | 128 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| e3nn | 2/2 | 128 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/2 | 128 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/2 | 128 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| cartnn | 2/2 | 128 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.87 GiB. GPU 0 has a tot |
| ictd | 2/2 | 128 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.91 GiB. GPU 0 has a tot |
| e3nn | 2/2 | 128 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/2 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.25 GiB. GPU 0 has a tot |
| ictd | 2/2 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| e3nn | 2/2 | 128 | 500000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/2 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.25 GiB. GPU 0 has a tot |
| ictd | 2/2 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| ictd | 2/3 | 32 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.49 GiB. GPU 0 has a tot |
| cartnn | 2/3 | 32 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/3 | 32 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.79 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 32 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.13 GiB. GPU 0 has a tot |
| cartnn | 2/3 | 32 | 500000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/3 | 32 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.79 GiB. GPU 0 has a tot |
| cartnn | 2/3 | 64 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/3 | 64 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.79 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.13 GiB. GPU 0 has a tot |
| cartnn | 2/3 | 64 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/3 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.79 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 64 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/3 | 64 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/3 | 64 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 64 | 500000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/3 | 64 | 500000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/3 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 128 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/3 | 128 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/3 | 128 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 128 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/3 | 128 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 2/3 | 128 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| ictd | 2/3 | 128 | 100000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 128 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.10 GiB. GPU 0 has a tot |
| cartnn | 2/3 | 128 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 6.01 GiB. GPU 0 has a tot |
| ictd | 2/3 | 128 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 128 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 2/3 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 6.20 GiB. GPU 0 has a tot |
| ictd | 2/3 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 6.20 GiB. GPU 0 has a tot |
| e3nn | 2/3 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 6.20 GiB. GPU 0 has a tot |
| cartnn | 2/3 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 6.20 GiB. GPU 0 has a tot |
| ictd | 2/3 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 6.20 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 32 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 32 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.09 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 32 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 32 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 32 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.09 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 32 | 100000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| e3nn | 3/3 | 32 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 32 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 32 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.17 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 32 | 500000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 32 | 500000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 32 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.17 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 64 | 100000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| e3nn | 3/3 | 64 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 64 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 64 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.17 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 64 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 64 | 500000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.17 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 64 | 50000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 64 | 100000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 64 | 100000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 64 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.72 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 64 | 100000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 64 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.34 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 64 | 500000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 64 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 9.54 GiB. GPU 0 has a tot |
| ictd | 3/3 | 64 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.48 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.48 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 9.54 GiB. GPU 0 has a tot |
| ictd | 3/3 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.48 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 128 | 50000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.89 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 128 | 100000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 128 | 100000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 128 | 100000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.72 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 128 | 100000 | float32 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 128 | 100000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.34 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 128 | 500000 | float32 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 128 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 9.54 GiB. GPU 0 has a tot |
| ictd | 3/3 | 128 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.48 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 128 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.48 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 128 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 9.54 GiB. GPU 0 has a tot |
| ictd | 3/3 | 128 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.48 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 128 | 50000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 128 | 50000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 128 | 50000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.72 GiB. GPU 0 has a tot |
| cartnn | 3/3 | 128 | 50000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 128 | 50000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.34 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 128 | 100000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 128 | 100000 | float64 | forward_only | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 128 | 100000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.67 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 128 | 100000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| cartnn | 3/3 | 128 | 100000 | float64 | forward_backward | oom | RuntimeError:The following operation failed in the TorchScript interpreter.
Trac |
| ictd | 3/3 | 128 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.67 GiB. GPU 0 has a tot |
| e3nn | 3/3 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 10.97 GiB. GPU 0 has a to |
| cartnn | 3/3 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 19.07 GiB. GPU 0 has a to |
| ictd | 3/3 | 128 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 10.97 GiB. GPU 0 has a to |
| e3nn | 3/3 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 10.97 GiB. GPU 0 has a to |
| cartnn | 3/3 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 19.07 GiB. GPU 0 has a to |
| ictd | 3/3 | 128 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 10.97 GiB. GPU 0 has a to |
| ictd_compiled | 1/2 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.15 GiB. GPU 0 has a tot |
| ictd_compiled | 2/2 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.38 GiB. GPU 0 has a tot |
| ictd_compiled | 2/2 | 64 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.86 GiB. GPU 0 has a tot |
| ictd_compiled | 2/2 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 3.58 GiB. GPU 0 has a tot |
| ictd_compiled | 2/3 | 64 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 2.98 GiB. GPU 0 has a tot |
| ictd_compiled | 2/3 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 4.17 GiB. GPU 0 has a tot |
| ictd_compiled | 2/3 | 64 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.96 GiB. GPU 0 has a tot |
| ictd_compiled | 2/3 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 3.58 GiB. GPU 0 has a tot |
| ictd_compiled | 3/3 | 64 | 500000 | float32 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 5.84 GiB. GPU 0 has a tot |
| ictd_compiled | 3/3 | 64 | 500000 | float32 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.79 GiB. GPU 0 has a tot |
| ictd_compiled | 3/3 | 64 | 100000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.67 GiB. GPU 0 has a tot |
| ictd_compiled | 3/3 | 64 | 500000 | float64 | forward_only | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 1.67 GiB. GPU 0 has a tot |
| ictd_compiled | 3/3 | 64 | 500000 | float64 | forward_backward | oom | OutOfMemoryError:CUDA out of memory. Tried to allocate 734.00 MiB. GPU 0 has a t |

## Observations (measured, scoped to the tested workloads)

- **float32 forward_only**: across all tested (config,channels,edges), eager `ictd` vs `cartnn` total-time ratio (cartnn/ictd) median **0.53×** (min 0.16×, max 1.78×); >1 ⇒ ICTC faster.
  - `ictd_compiled` vs `cartnn`: median **0.59×** (min 0.31×, max 1.45×).
- **float32 forward_backward**: across all tested (config,channels,edges), eager `ictd` vs `cartnn` total-time ratio (cartnn/ictd) median **0.67×** (min 0.37×, max 3.34×); >1 ⇒ ICTC faster.
  - `ictd_compiled` vs `cartnn`: median **0.73×** (min 0.44×, max 3.35×).
- **float64 forward_only**: across all tested (config,channels,edges), eager `ictd` vs `cartnn` total-time ratio (cartnn/ictd) median **0.40×** (min 0.08×, max 2.30×); >1 ⇒ ICTC faster.
  - `ictd_compiled` vs `cartnn`: median **0.34×** (min 0.08×, max 2.23×).
- **float64 forward_backward**: across all tested (config,channels,edges), eager `ictd` vs `cartnn` total-time ratio (cartnn/ictd) median **0.51×** (min 0.21×, max 3.15×); >1 ⇒ ICTC faster.
  - `ictd_compiled` vs `cartnn`: median **0.67×** (min 0.20×, max 2.19×).

Interpretation guidance: cartnn's full `3**l` storage makes its per-edge work grow faster with `max_ell` than the `2l+1` ICTC/e3nn layouts, which is the main structural difference these numbers probe. Where eager ICTC trails, it is the eager per-path launch overhead, not the ICTC algebra — compare the `ictd_compiled` row. e3nn is included only as the spherical MACE-native reference. None of this speaks to accuracy or to full-model performance.

## Key findings (added)

1. **cartnn `3**l` penalty grows with angular order.** At channels=64, edges=100000, fp32 forward-only,
   `cartnn/e3nn` total-time ratio is 1.00× (l1/1, l1/2) → 1.21× (l2/2) → 1.20× (l2/3) → **2.80× (l3/3)**.
   The full Cartesian layout (degree-l = `3**l` numbers vs `2l+1`) only costs once l=3 tensors (27 vs 7)
   enter the product.
2. **ICTC vs cartnn crosses over at l=3.** Eager ICTC is slower than cartnn at low order
   (`cartnn/ictd` 0.22–0.59×) but **faster at l3/3** (1.20× fp32 fwd-only). The `2l+1` packing keeps
   ICTC competitive exactly where cartnn's storage blows up.
3. **e3nn (spherical 2l+1) is fastest of the three at every tested point.** Both Cartesian-basis
   operators (ICTC intrinsic, cartnn full) trail the well-optimized spherical TP at the operator level.
4. **`torch.compile` ≈ no-op for the standalone ICTC operator** (graph-breaks on dict I/O):
   `ictd_compiled ≈ ictd` except the tiny l1/1 case (0.58×) and fitting l3/3 fp64 (287 ms) where eager
   OOMs. Deployed competitiveness is from **model-level AOTI** (existing `benchmark_results/`), not this
   op in isolation.
5. **Memory / OOM (24 GB):** OOM cells by backend e3nn 35 < cartnn 46 < ictd 50 (of 720); cartnn OOMs
   more than e3nn from the `3**l` memory; ICTC's eager path-preserving layout is the most memory-hungry.

**Warmup validity:** the ICTC operator's first forward is 2–22× slower (it populates the
`(device,dtype)` CG/projector cache), but the plateau is reached by call #2–3 and is flat to ≤0.1%;
the harness discards `warmup=20` calls, so all numbers above are warm. Details: `warmup_validation.md`,
`warmup_curve.log`.

**Scope:** operator-level *comparable workload* (same `(l1,l2,l3)` path set, edge batch, per-edge weight
count), **not** exact apples-to-apples (cartnn `3**l` storage; different per-path normalization). cartnn
ships no symmetric-contraction operator, so the MACE contraction is not benchmarked. No model-level or
chemical-accuracy claim.

## Reconciliation with model-level results (read this before quoting "e3nn fastest")

This bench isolates the **operator** and times ICTC **eager** (torch.compile graph-breaks on its dict
I/O) vs e3nn's **codegen-fused** TP — so "ICTC slower than e3nn" is the **un-fused operator at large
scale**, NOT the deployed model. The model-level `benchmark_results/` (inference, C64, l2/2) shows the
same eager regime and the real deployment win:

| atoms | mace-e3nn eager | ICTC eager | ICTC AOTI | ICTC-eager/e3nn | ICTC-AOTI/e3nn |
|---|---|---|---|---|---|
| 512 | 34.2 | 14.6 | 3.76 | 2.35× | 9.10× |
| 4096 | 65.0 | 68.0 | 27.7 | 0.96× | 2.35× |
| 8192 | 114.0 | 141.1 | 55.3 | 0.81× | 2.06× |

ICTC **eager** whole-model is itself slower than mace-e3nn at ≥4096 atoms (matches this op bench at large
edges). The "MACE-ICTC ≫ mace-e3nn" the model sees is **ICTC AOTI** = whole-graph fusion killing launch
overhead, not the isolated operator. (mace's `MACE cuEq` is a flat ~20 ms floor; ICTC-AOTI beats it
≤2048 atoms, loses ≥4096.) End-to-end speed here is governed by **fusion (AOTI)**, not per-op FLOPs;
this bench does not AOTI the ICTC op, so it under-represents the deployed ICTC.
