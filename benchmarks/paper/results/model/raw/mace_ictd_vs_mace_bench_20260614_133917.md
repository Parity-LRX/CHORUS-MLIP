# MACE-ICTC vs mace-torch benchmark

## Metadata

- python: `/home/ylzhang/micromamba/envs/FSCETP/bin/python`
- torch: `2.7.1+cu128`
- cuda_available: `True`
- cuda_device: `NVIDIA GeForce RTX 4090 D`
- mace_version: `0.3.16`
- mace_file: `/tmp/mace_torch_0_3_16/mace/__init__.py`
- device: `cuda`
- dtype: `float32`
- matmul_precision: `high`
- atoms: `1024`
- avg_degree: `16`
- channels: `64`
- num_interactions: `2`
- correlation: `2`
- configs: `1:2,2:3`
- train_iters: `2`
- infer_iters: `5`
- aoti: `True`
- include_pure_u: `False`

## OK rows

| task | mode | atoms | lmax | max_ell | ms | compile_s | speedup_vs_e3nn |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | mace_torch_e3nn | 1024 | 1 | 2 | 297.183 |  | 1 |
| inference | mace_torch_e3nn | 1024 | 1 | 2 | 20.3126 |  | 1 |
| train | mace_torch_cueq | 1024 | 1 | 2 | 40.936 |  | 7.2597 |
| inference | mace_torch_cueq | 1024 | 1 | 2 | 18.3131 |  | 1.10918 |
| train | mace_ictc_bridge_u_eager | 1024 | 1 | 2 | 35.0105 |  | 8.4884 |
| train | mace_ictc_bridge_u_makefx_train | 1024 | 1 | 2 | 14.8282 | 43.1189 | 20.0417 |
| inference | mace_ictc_bridge_u_eager | 1024 | 1 | 2 | 13.0186 |  | 1.56028 |
| inference | mace_ictc_bridge_u_aoti | 1024 | 1 | 2 | 4.39656 | 28.1894 | 4.62011 |
| train | mace_ictc_cueq_product_eager | 1024 | 1 | 2 | 41.6927 |  | 7.12794 |
| train | mace_ictc_cueq_product_makefx_train | 1024 | 1 | 2 | 16.1851 | 43.4363 | 18.3615 |
| inference | mace_ictc_cueq_product_eager | 1024 | 1 | 2 | 15.9964 |  | 1.26982 |
| inference | mace_ictc_cueq_product_aoti | 1024 | 1 | 2 | 4.35861 | 22.5435 | 4.66034 |
| train | mace_torch_e3nn | 1024 | 2 | 3 | 391.975 |  | 1 |
| inference | mace_torch_e3nn | 1024 | 2 | 3 | 47.9856 |  | 1 |
| train | mace_torch_cueq | 1024 | 2 | 3 | 53.7548 |  | 7.29191 |
| inference | mace_torch_cueq | 1024 | 2 | 3 | 23.5861 |  | 2.03449 |
| train | mace_ictc_bridge_u_eager | 1024 | 2 | 3 | 92.9726 |  | 4.21603 |
| train | mace_ictc_bridge_u_makefx_train | 1024 | 2 | 3 | 35.0039 | 77.6783 | 11.198 |
| inference | mace_ictc_bridge_u_eager | 1024 | 2 | 3 | 32.2828 |  | 1.48641 |
| inference | mace_ictc_bridge_u_aoti | 1024 | 2 | 3 | 11.7716 | 38.3021 | 4.07639 |
| train | mace_ictc_cueq_product_eager | 1024 | 2 | 3 | 96.484 |  | 4.06259 |
| train | mace_ictc_cueq_product_makefx_train | 1024 | 2 | 3 | 35.9611 | 74.5269 | 10.9 |
| inference | mace_ictc_cueq_product_eager | 1024 | 2 | 3 | 33.9575 |  | 1.41311 |
| inference | mace_ictc_cueq_product_aoti | 1024 | 2 | 3 | 11.7675 | 34.7689 | 4.07781 |
