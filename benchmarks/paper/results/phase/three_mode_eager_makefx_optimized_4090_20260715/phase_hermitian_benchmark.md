# PEMP Hermitian phase matched benchmark

- python: `/home/ylzhang/micromamba/envs/FSCETP/bin/python`
- torch: `2.7.1+cu128`
- cuda: `12.8`
- gpu: `NVIDIA GeForce RTX 4090 D`
- dtype: `float32`
- tf32: `False`
- product_backend: `ictd-bridge-u`
- elapsed_s: `364.9620690345764`
- command: `/home/ylzhang/MACE-ICTC-Phase/mace_ictc/bench/bench_phase_hermitian.py --device cuda --dtype float32 --product-backend ictd-bridge-u --atoms-list 128,512 --avg-degree 20 --modes baseline,phase_full_l_rank8,phase_full_l_persistent_rank8 --channels 64 --hidden-lmax 1 --max-ell 2 --num-interactions 2 --correlation 2 --phase-density-rank 8 --train-warmup 10 --train-iters 100 --infer-warmup 20 --infer-iters 200 --include-makefx --out-dir benchmarks/paper/results/phase/three_mode_eager_makefx_optimized_4090_20260715`

| task | mode | atoms | parameters | ms | steps/s | atoms/s | overhead | peak MiB | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| train_eager | baseline | 128 | 194256 | 31.1184 | 32.135 | 4113.3 | 0.0% | 194.0361328125 | ok |
| train_eager | phase_full_l_rank8 | 128 | 223956 | 43.6201 | 22.925 | 2934.4 | 40.2% | 260.095703125 | ok |
| train_eager | phase_full_l_persistent_rank8 | 128 | 278235 | 58.3285 | 17.144 | 2194.5 | 87.4% | 297.11669921875 | ok |
| inference_eager | baseline | 128 | 194256 | 12.6444 | 79.086 | 10123.0 | 0.0% | 129.5341796875 | ok |
| inference_eager | phase_full_l_rank8 | 128 | 223956 | 17.5939 | 56.838 | 7275.2 | 39.1% | 166.548828125 | ok |
| inference_eager | phase_full_l_persistent_rank8 | 128 | 278235 | 23.6573 | 42.270 | 5410.6 | 87.1% | 180.71142578125 | ok |
| train_makefx | baseline | 128 | 194256 | 10.0809 | 99.198 | 12697.3 | 0.0% | 205.97998046875 | ok |
| train_makefx | phase_full_l_rank8 | 128 | 223956 | 13.6942 | 73.024 | 9347.0 | 35.8% | 347.56494140625 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 128 | 278235 | 17.4250 | 57.389 | 7345.8 | 72.9% | 389.8525390625 | ok |
| train_eager | baseline | 512 | 194256 | 31.1586 | 32.094 | 16432.1 | 0.0% | 719.42919921875 | ok |
| train_eager | phase_full_l_rank8 | 512 | 223956 | 44.0097 | 22.722 | 11633.8 | 41.2% | 994.81103515625 | ok |
| train_eager | phase_full_l_persistent_rank8 | 512 | 278235 | 58.3864 | 17.127 | 8769.2 | 87.4% | 1145.0380859375 | ok |
| inference_eager | baseline | 512 | 194256 | 12.4992 | 80.005 | 40962.6 | 0.0% | 473.82861328125 | ok |
| inference_eager | phase_full_l_rank8 | 512 | 223956 | 17.9494 | 55.712 | 28524.6 | 43.6% | 625.71142578125 | ok |
| inference_eager | phase_full_l_persistent_rank8 | 512 | 278235 | 23.8034 | 42.011 | 21509.6 | 90.4% | 683.138671875 | ok |
| train_makefx | baseline | 512 | 194256 | 10.4445 | 95.744 | 49021.2 | 0.0% | 744.99609375 | ok |
| train_makefx | phase_full_l_rank8 | 512 | 223956 | 13.9054 | 71.915 | 36820.3 | 33.1% | 1104.53125 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 512 | 278235 | 18.0063 | 55.536 | 28434.5 | 72.4% | 1284.10400390625 | ok |
