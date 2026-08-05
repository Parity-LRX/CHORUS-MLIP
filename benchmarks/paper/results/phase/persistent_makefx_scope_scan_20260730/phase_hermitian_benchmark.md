# U1-CHORUS Hermitian phase matched benchmark

- python: `/home/ylzhang/micromamba/envs/FSCETP/bin/python`
- torch: `2.7.1+cu128`
- cuda: `12.8`
- gpu: `NVIDIA GeForce RTX 4090 D`
- dtype: `float32`
- tf32: `False`
- product_backend: `ictd-bridge-u`
- elapsed_s: `1181.2379138469696`
- command: `mace_ictc/bench/bench_phase_hermitian.py --device cuda --dtype float32 --product-backend ictd-bridge-u --atoms-list 128,256,512,1024,2048 --avg-degree 20 --modes phase_full_l_rank8,phase_full_l_persistent_rank8 --channels 64 --hidden-lmax 1 --max-ell 2 --num-interactions 2 --correlation 2 --phase-density-rank 8 --tasks train_makefx,inference_makefx --train-warmup 10 --train-iters 100 --infer-warmup 20 --infer-iters 200 --out-dir /home/ylzhang/chorus_runs/persistent_makefx_scope_scan_20260730`

| task | mode | atoms | parameters | ms | steps/s | atoms/s | overhead | peak MiB | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| train_makefx | phase_full_l_rank8 | 128 | 223956 | 13.5999 | 73.530 | 9411.8 |  | 336.275390625 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 128 | 278235 | 21.4898 | 46.534 | 5956.3 |  | 379.17236328125 | ok |
| inference_makefx | phase_full_l_rank8 | 128 | 223956 | 5.1327 | 194.828 | 24938.0 |  | 225.95947265625 | ok |
| inference_makefx | phase_full_l_persistent_rank8 | 128 | 278235 | 6.5847 | 151.867 | 19439.0 |  | 259.2763671875 | ok |
| train_makefx | phase_full_l_rank8 | 256 | 223956 | 13.8281 | 72.316 | 18513.0 |  | 589.19091796875 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 256 | 278235 | 17.6726 | 56.585 | 14485.7 |  | 686.36767578125 | ok |
| inference_makefx | phase_full_l_rank8 | 256 | 223956 | 5.2829 | 189.289 | 48457.9 |  | 477.26220703125 | ok |
| inference_makefx | phase_full_l_persistent_rank8 | 256 | 278235 | 6.6026 | 151.455 | 38772.6 |  | 540.365234375 | ok |
| train_makefx | phase_full_l_rank8 | 512 | 223956 | 13.9564 | 71.652 | 36685.8 |  | 1059.404296875 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 512 | 278235 | 17.7953 | 56.195 | 28771.7 |  | 1238.3359375 | ok |
| inference_makefx | phase_full_l_rank8 | 512 | 223956 | 5.1603 | 193.786 | 99218.6 |  | 856.81103515625 | ok |
| inference_makefx | phase_full_l_persistent_rank8 | 512 | 278235 | 6.5074 | 153.672 | 78679.9 |  | 991.142578125 | ok |
| train_makefx | phase_full_l_rank8 | 1024 | 223956 | 22.1865 | 45.072 | 46154.2 |  | 2114.26123046875 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 1024 | 278235 | 26.1914 | 38.181 | 39096.9 |  | 2461.322265625 | ok |
| inference_makefx | phase_full_l_rank8 | 1024 | 223956 | 7.6474 | 130.763 | 133901.1 |  | 1688.22119140625 | ok |
| inference_makefx | phase_full_l_persistent_rank8 | 1024 | 278235 | 8.9145 | 112.177 | 114868.7 |  | 1939.447265625 | ok |
| train_makefx | phase_full_l_rank8 | 2048 | 223956 | 40.6788 | 24.583 | 50345.7 |  | 4097.01806640625 | ok |
| train_makefx | phase_full_l_persistent_rank8 | 2048 | 278235 | 47.5261 | 21.041 | 43092.1 |  | 4804.83642578125 | ok |
| inference_makefx | phase_full_l_rank8 | 2048 | 223956 | 15.2198 | 65.704 | 134561.7 |  | 3333.3544921875 | ok |
| inference_makefx | phase_full_l_persistent_rank8 | 2048 | 278235 | 17.6619 | 56.619 | 115955.9 |  | 3841.3701171875 | ok |
