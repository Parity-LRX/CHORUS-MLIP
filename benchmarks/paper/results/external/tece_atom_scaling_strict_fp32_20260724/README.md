# CHORUS–TECE atom-count scaling, strict FP32

## Protocol

- Hardware: one NVIDIA GeForce RTX 4090 D; all modes run serially.
- Arithmetic: fp32 with `float32_matmul_precision=highest`,
  `matmul.allow_tf32=False`, and `cudnn.allow_tf32=False`.
- Structures: deterministic jittered periodic simple-cubic carbon cells.
- Density: 3.0 Å lattice spacing and approximately 18 directed neighbours per
  atom at a 5.0 Å cutoff.
- Timed quantity: model execution for energy plus conservative forces on a
  preconstructed graph. Neighbour-list construction, checkpoint loading, and
  compilation are excluded from the steady latency.
- Statistic: median of 15 calls through 512 atoms, 8 calls through 1728 atoms,
  and 5 calls thereafter, following three warm-up calls per size.
- Models: deployed CHORUS C48/L2 rank-8 full-Hermitian residual (168,634
  parameters) and TECE C24/L2 two-layer `[CGTP, SO2]` (263,361 parameters).

## Results

The speedup uses the faster TECE mode at each size; TECE eager was faster than
TECE-CUE at every point.

| Atoms | Edges | CHORUS (ms) | TECE eager (ms) | TECE CUE (ms) | CHORUS speedup | CHORUS peak GiB | TECE eager peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1,152 | **4.960** | 32.922 | 33.652 | **6.64×** | 0.087 | 0.058 |
| 125 | 2,250 | **5.034** | 33.087 | 33.729 | **6.57×** | 0.152 | 0.096 |
| 216 | 3,888 | **5.100** | 33.130 | 33.751 | **6.50×** | 0.267 | 0.156 |
| 343 | 6,178 | **4.987** | 33.251 | 33.863 | **6.67×** | 0.416 | 0.233 |
| 512 | 9,222 | **5.810** | 33.290 | 33.981 | **5.73×** | 0.643 | 0.340 |
| 729 | 13,126 | **7.817** | 33.456 | 33.884 | **4.28×** | 0.923 | 0.479 |
| 1,000 | 18,006 | **9.982** | 33.524 | 33.888 | **3.36×** | 1.233 | 0.642 |
| 1,728 | 31,120 | **15.910** | 40.322 | 41.160 | **2.53×** | 2.052 | 1.096 |
| 2,744 | 49,404 | **25.533** | 61.468 | 62.103 | **2.41×** | 3.182 | 1.722 |
| 4,096 | 73,768 | **37.327** | 90.465 | 91.046 | **2.42×** | 4.731 | 2.560 |

At 4,096 atoms, CHORUS reaches 109.7k atoms/s versus 45.3k atoms/s for TECE
eager. A diagnostic affine fit over the four points from 1,000 to 4,096 atoms
gives 8.90 μs/atom for CHORUS and 18.93 μs/atom for TECE eager, with
respectively 0.9995 and 0.9847 R². The large-system per-atom cost is therefore
about 2.13× lower for CHORUS.

## Validation and evidence boundary

**Assessment: ready to share with caveats.**

- All 30 measurements completed without OOM. Atom and edge counts match exactly
  across all three modes. Neighbour density stays within 0.012 of 18 per atom.
- The maximum within-size min-to-max timing spread is 6.7% for CHORUS, 1.2% for
  TECE eager, and 2.7% for TECE-CUE. Medians, rather than minima, are reported.
- CHORUS is faster at every measured size, but its advantage narrows from about
  6.6× on small graphs to 2.4× on large graphs because TECE's large fixed
  overhead is progressively amortized.
- CUE does not accelerate this TECE configuration, including at 4,096 atoms.
  It replaces only the first-layer O(3) scatter tensor product; the second
  SO(2)/attention layer remains unchanged.
- CHORUS uses more memory: 4.73 GiB versus 2.56 GiB at 4,096 atoms. The result is
  a speed advantage, not simultaneous dominance in speed and memory.
- CHORUS currently requires an atom-count-specific MakeFX bucket because a
  model-internal view specializes the traced node count. Compilation is
  excluded from steady latency and must be amortized in fixed-size MD.
- These synthetic carbon cells test computational scaling, not accuracy,
  chemical realism, neighbour-list construction, ASE overhead, or complete MD
  throughput. Those claims require separate benchmarks.

Raw machine-readable records are in [`raw/`](raw/).
