# PEMP Hermitian phase matched benchmark

- python: `/home/ylzhang/micromamba/envs/FSCETP/bin/python`
- torch: `2.7.1+cu128`
- cuda: `12.8`
- gpu: `NVIDIA GeForce RTX 4090 D`
- dtype: `float32`
- tf32: `False`
- product_backend: `ictd-bridge-u`
- elapsed_s: `437.8459851741791`
- command: `/home/ylzhang/MACE-ICTC-Phase/mace_ictc/bench/bench_phase_hermitian.py --device cuda --dtype float32 --product-backend ictd-bridge-u --atoms-list 128,512 --avg-degree 20 --channels 64 --hidden-lmax 1 --max-ell 2 --num-interactions 2 --correlation 2 --phase-density-rank 8 --modes baseline,phase_full_l_rank8,phase_full_l_persistent_rank8 --train-warmup 10 --train-iters 100 --infer-warmup 20 --infer-iters 200 --include-makefx --out-dir benchmarks/paper/results/phase/three_mode_eager_makefx_4090_20260715`

| task | mode | atoms | parameters | ms | steps/s | atoms/s | overhead | peak MiB | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| train_eager | baseline | 128 | 194256 | 30.6653 | 32.610 | 4174.1 | 0.0% | 194.0361328125 | ok |
| train_eager | phase_full_l_rank8 | 128 | 223956 | 45.3125 | 22.069 | 2824.8 | 47.8% | 248.91796875 | ok |
| train_eager | phase_full_l_persistent_rank8 | 128 | 278235 | 63.3704 | 15.780 | 2019.9 | 106.7% | 281.19140625 | ok |
| inference_eager | baseline | 128 | 194256 | 12.4675 | 80.208 | 10266.7 | 0.0% | 129.5341796875 | ok |
| inference_eager | phase_full_l_rank8 | 128 | 223956 | 18.2645 | 54.751 | 7008.1 | 46.5% | 149.2578125 | ok |
| inference_eager | phase_full_l_persistent_rank8 | 128 | 278235 | 25.6407 | 39.001 | 4992.1 | 105.7% | 162.7216796875 | ok |
| train_makefx | baseline | 128 | 194256 | 9.9560 | 100.442 | 12856.6 | 0.0% | 273.6669921875 | ok |
| train_makefx | phase_full_l_rank8 | 128 | 223956 | 14.0462 | 71.194 | 9112.8 | 41.1% | 334.578125 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 128 | 278235 | 19.3240 | 51.749 | 6623.9 | 94.1% | 384.47900390625 | ok |
| train_eager | baseline | 512 | 194256 | 30.9306 | 32.330 | 16553.2 | 0.0% | 718.80419921875 | ok |
| train_eager | phase_full_l_rank8 | 512 | 223956 | 45.7714 | 21.848 | 11186.0 | 48.0% | 943.07470703125 | ok |
| train_eager | phase_full_l_persistent_rank8 | 512 | 278235 | 63.8961 | 15.650 | 8013.0 | 106.6% | 1069.01416015625 | ok |
| inference_eager | baseline | 512 | 194256 | 12.5004 | 79.997 | 40958.7 | 0.0% | 473.82861328125 | ok |
| inference_eager | phase_full_l_rank8 | 512 | 223956 | 18.5983 | 53.768 | 27529.4 | 48.8% | 550.10986328125 | ok |
| inference_eager | phase_full_l_persistent_rank8 | 512 | 278235 | 25.8843 | 38.633 | 19780.3 | 107.1% | 605.43408203125 | ok |
| train_makefx | baseline | 512 | 194256 | 10.4331 | 95.848 | 49074.4 | 0.0% | 812.63671875 | ok |
| train_makefx | phase_full_l_rank8 | 512 | 223956 | 14.2012 | 70.417 | 36053.4 | 36.1% | 1051.1015625 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 512 | 278235 | 22.1383 | 45.171 | 23127.3 | 112.2% | 1256.64306640625 | ok |
