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
- atoms: `4096`
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
| train | mace_torch_e3nn | 4096 | 1 | 2 | 312.148 |  | 1 |
| inference | mace_torch_e3nn | 4096 | 1 | 2 | 31.7407 |  | 1 |
| train | mace_torch_cueq | 4096 | 1 | 2 | 41.3771 |  | 7.54398 |
| inference | mace_torch_cueq | 4096 | 1 | 2 | 18.1711 |  | 1.74677 |
| train | mace_ictc_bridge_u_eager | 4096 | 1 | 2 | 115.692 |  | 2.69809 |
| train | mace_ictc_bridge_u_makefx_train | 4096 | 1 | 2 | 50.9039 | 47.3873 | 6.1321 |
| inference | mace_ictc_bridge_u_eager | 4096 | 1 | 2 | 39.2443 |  | 0.808798 |
| inference | mace_ictc_bridge_u_aoti | 4096 | 1 | 2 | 17.1367 | 29.635 | 1.85221 |
| train | mace_ictc_cueq_product_eager | 4096 | 1 | 2 | 111.55 |  | 2.79828 |
| train | mace_ictc_cueq_product_makefx_train | 4096 | 1 | 2 | 45.2399 | 44.962 | 6.89984 |
| inference | mace_ictc_cueq_product_eager | 4096 | 1 | 2 | 38.8906 |  | 0.816154 |
| inference | mace_ictc_cueq_product_aoti | 4096 | 1 | 2 | 15.1479 | 20.8934 | 2.09539 |
| train | mace_torch_e3nn | 4096 | 2 | 3 | 576.093 |  | 1 |
| inference | mace_torch_e3nn | 4096 | 2 | 3 | 118.513 |  | 1 |
| train | mace_torch_cueq | 4096 | 2 | 3 | 55.5171 |  | 10.3769 |
| inference | mace_torch_cueq | 4096 | 2 | 3 | 24.6651 |  | 4.80489 |
| train | mace_ictc_bridge_u_eager | 4096 | 2 | 3 | 368.236 |  | 1.56447 |
| train | mace_ictc_bridge_u_makefx_train | 4096 | 2 | 3 | 142.21 | 89.3567 | 4.051 |
| inference | mace_ictc_bridge_u_eager | 4096 | 2 | 3 | 132.582 |  | 0.893885 |
| inference | mace_ictc_bridge_u_aoti | 4096 | 2 | 3 | 46.9276 | 40.554 | 2.52544 |
| train | mace_ictc_cueq_product_eager | 4096 | 2 | 3 | 357.238 |  | 1.61263 |
| train | mace_ictc_cueq_product_makefx_train | 4096 | 2 | 3 | 131.494 | 80.6442 | 4.38114 |
| inference | mace_ictc_cueq_product_eager | 4096 | 2 | 3 | 128.695 |  | 0.920883 |
| inference | mace_ictc_cueq_product_aoti | 4096 | 2 | 3 | 42.655 | 36.0617 | 2.77841 |
