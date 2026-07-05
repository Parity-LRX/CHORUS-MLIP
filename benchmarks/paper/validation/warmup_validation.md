# Warmup validity of the operator benchmark (ICTC cache)

**Concern:** the ICTC operator uses lazily-populated caches, so speed must be measured **after**
warmup, or ICTC looks artificially slow. Verified here.

**Mechanism (code):** `EdgeWeightedPathPreservingTensorProduct` (and its parents) hold instance-level
`_cg_cache_by_dev_dtype` / `_proj_group_cache_by_dev_dtype`, keyed by `(device, dtype)`. The first
forward calls `_cg_for(device, dtype)`, which moves the CPU-float64 CG/projector tensors to the GPU
and casts them, then caches; every later forward hits the cache. (The fixed CG/U/projector *values*
are also lru-cached + on-disk cached at build time.) `internal_compute_dtype` defaults to
`torch.get_default_dtype()`, which the harness sets per dtype-block, so fp32 ICTC computes in fp32.

**Harness:** `operator_bench.py::time_config` runs `warmup=20` full forward (and forward+backward)
iterations, then `torch.cuda.synchronize()`, **then** CUDA-event-times `measured` iterations
(calls #21–#120). So the cold first call is discarded.

**Per-call probe** (`warmup_curve.py` → `warmup_curve.log`, channels=64, edges=100000), forward ms:

| op · config · dtype | call #1 (cold) | call #2 | warm median[21:] | call1/warm | drift median[6:]/[21:] |
|---|---|---|---|---|---|
| ictd l1/1 fp32 | 137.79 | 6.32 | 6.26 | 22.0× | 1.001 |
| ictd l1/1 fp64 | 134.81 | 31.70 | 30.10 | 4.5× | 1.001 |
| ictd l2/2 fp32 | 166.04 | 29.57 | 29.05 | 5.7× | 1.000 |
| ictd l2/2 fp64 | 223.35 | 105.15 | 105.06 | 2.1× | 1.000 |
| ictd l2/3 fp32 | 175.53 | 38.72 | 38.16 | 4.6× | 1.000 |
| ictd l3/3 fp32 | 220.30 | 81.91 | 81.44 | 2.7× | 1.000 |
| cartnn l2/2 fp32 | 54.66 | 24.14 | 16.38 | 3.3× | 0.999 |
| e3nn l2/2 fp32 | 19.63 | 20.98 | 13.47 | 1.5× | 1.000 |

**Conclusions:**
1. ICTC's first call is genuinely expensive (up to **22× warm**) — the cache population is real, exactly
   as flagged. **But it amortizes completely by call #2–#3.**
2. The warm plateau is flat: `median(calls 6–40) == median(calls 21–40)` to ≤0.1% for every op. So
   `warmup=20` lands well inside the plateau; the measured numbers are **not** cold-cache-contaminated.
3. Cross-check: probe warm medians (ictd 6.26 / 29.05 / 81.44 ms at l1/1, l2/2, l3/3 fp32) match the
   benchmark CSV (6.15 / 28.99 / 81.31 ms) to <2%. The recorded ICTC speeds are warm and valid.
4. cartnn/e3nn (codegen-fused) also warm within 2–3 calls; same `warmup=20` covers them.

(The probe itself has no OOM guard and aborted on the very last config, l3/3 **fp64** cartnn, with a
CUDA OOM on an 11.6 GB einsum — itself a data point on cartnn's `3**l` memory cost; the main sweep
records that cell as `status=oom`.)
