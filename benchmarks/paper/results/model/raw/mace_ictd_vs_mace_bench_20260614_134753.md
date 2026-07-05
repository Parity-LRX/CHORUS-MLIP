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
- atoms: `2048`
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
| train | mace_torch_e3nn | 2048 | 1 | 2 | 300.382 |  | 1 |
| inference | mace_torch_e3nn | 2048 | 1 | 2 | 21.0724 |  | 1 |
| train | mace_torch_cueq | 2048 | 1 | 2 | 40.7398 |  | 7.37318 |
| inference | mace_torch_cueq | 2048 | 1 | 2 | 17.9436 |  | 1.17437 |
| train | mace_ictc_bridge_u_eager | 2048 | 1 | 2 | 58.9068 |  | 5.09928 |
| train | mace_ictc_bridge_u_makefx_train | 2048 | 1 | 2 | 26.1559 | 45.8602 | 11.4843 |
| inference | mace_ictc_bridge_u_eager | 2048 | 1 | 2 | 20.1544 |  | 1.04555 |
| inference | mace_ictc_bridge_u_aoti | 2048 | 1 | 2 | 8.56171 | 28.8534 | 2.46124 |
| train | mace_ictc_cueq_product_eager | 2048 | 1 | 2 | 60.5725 |  | 4.95905 |
| train | mace_ictc_cueq_product_makefx_train | 2048 | 1 | 2 | 24.7152 | 45.6179 | 12.1537 |
| inference | mace_ictc_cueq_product_eager | 2048 | 1 | 2 | 20.9245 |  | 1.00707 |
| inference | mace_ictc_cueq_product_aoti | 2048 | 1 | 2 | 7.78037 | 23.229 | 2.70841 |
| train | mace_torch_e3nn | 2048 | 2 | 3 | 443.357 |  | 1 |
| inference | mace_torch_e3nn | 2048 | 2 | 3 | 69.8027 |  | 1 |
| train | mace_torch_cueq | 2048 | 2 | 3 | 54.345 |  | 8.15819 |
| inference | mace_torch_cueq | 2048 | 2 | 3 | 23.4238 |  | 2.97999 |
| train | mace_ictc_bridge_u_eager | 2048 | 2 | 3 | 181.298 |  | 2.44546 |
| train | mace_ictc_bridge_u_makefx_train | 2048 | 2 | 3 | 69.0492 | 85.3339 | 6.42089 |
| inference | mace_ictc_bridge_u_eager | 2048 | 2 | 3 | 64.2576 |  | 1.08629 |
| inference | mace_ictc_bridge_u_aoti | 2048 | 2 | 3 | 23.3489 | 40.1149 | 2.98955 |
| train | mace_ictc_cueq_product_eager | 2048 | 2 | 3 | 180.495 |  | 2.45634 |
| train | mace_ictc_cueq_product_makefx_train | 2048 | 2 | 3 | 66.0196 | 79.828 | 6.71554 |
| inference | mace_ictc_cueq_product_eager | 2048 | 2 | 3 | 64.6449 |  | 1.07979 |
| inference | mace_ictc_cueq_product_aoti | 2048 | 2 | 3 | 21.7533 | 36.2565 | 3.20883 |
