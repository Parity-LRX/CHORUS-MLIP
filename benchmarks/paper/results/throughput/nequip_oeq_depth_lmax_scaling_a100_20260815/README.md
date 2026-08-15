# NequIP OpenEquivariance cost controls on A100

This benchmark measures strict-FP32 energy/force throughput on one NVIDIA A100.
All models use 64 channels, a fixed directed degree of 32, and the same
synthetic periodic graphs. Inference includes conservative forces; training
includes the force double backward and one AdamW update. Inference is measured
from 128 to 4096 atoms and training from 128 to 2048 atoms.

OpenEquivariance is enabled for every configuration. Each result records a
loaded precompiled OpenEquivariance extension and audits the corresponding
tensor-product modules. The CHORUS configurations contain both
`OpenEquivarianceTensorProduct` backbone modules and
`OpenEquivarianceHermitianTensorProduct` modules, so the reported numbers
measure the complete OpenEquivariance-accelerated model rather than an e3nn
fallback.

## Representative 2048-atom point

| Configuration | Parameters | Inference (katom/s) | Training (katom/s) | Inference memory (GiB) | Training memory (GiB) |
|:--|--:|--:|--:|--:|--:|
| NequIP-OEQ, L2 Off | 389,416 | 147.18 | 51.12 | 1.71 | 3.19 |
| NequIP-OEQ, L2 CHORUS-Final R16 | 581,261 | 66.83 | 24.72 | 2.51 | 5.14 |
| NequIP-OEQ, L2 CHORUS-Persistent R16 | 635,381 | 54.88 | 20.30 | 2.65 | 5.74 |
| NequIP-OEQ, Lmax3 Off, 2 layers | 631,080 | 76.31 | 27.82 | 4.08 | 7.11 |
| NequIP-OEQ, L2 Off, 3 layers | 660,776 | 88.00 | 30.64 | 2.16 | 4.87 |

At 4096 atoms, inference throughput is 162.83, 70.96, 58.17, 80.42, and
94.68 katom/s in the same row order.

## Cost interpretation

Relative to the two-layer Off model, Final adds 49.3% parameters and reduces
2048-atom inference/training throughput by 54.6%/51.7%. Persistent adds 63.2%
parameters and reduces throughput by 62.7%/60.3%. These are complete-model
costs with OpenEquivariance active in both the backbone and Hermitian tensor
products.

Persistent and the two-layer Lmax3 control have nearly identical parameter
counts (within 0.7%). Persistent uses 35.1% less inference memory and 19.2%
less training memory, but is 28.1%/27.0% slower in inference/training. The
three-layer Off control is 3.8% larger than Persistent and is 60.3%/51.0%
faster in inference/training, while Persistent uses 22.7%/17.8% more memory.
Consequently, the NequIP result supports a memory advantage over raising
angular order, but not a raw-throughput advantage over adding a layer. Any
accuracy--cost claim should combine these measurements with the matched test
errors rather than treating parameter count as a proxy for compute.

Raw machine-readable records are stored in `raw/`.
