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
- atoms: `512`
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
| train | mace_torch_e3nn | 512 | 1 | 2 | 300.467 |  | 1 |
| inference | mace_torch_e3nn | 512 | 1 | 2 | 20.4563 |  | 1 |
| train | mace_torch_cueq | 512 | 1 | 2 | 40.4438 |  | 7.42925 |
| inference | mace_torch_cueq | 512 | 1 | 2 | 18.3542 |  | 1.11453 |
| train | mace_ictc_bridge_u_eager | 512 | 1 | 2 | 33.5516 |  | 8.95537 |
| train | mace_ictc_bridge_u_makefx_train | 512 | 1 | 2 | 11.6893 | 43.1745 | 25.7044 |
| inference | mace_ictc_bridge_u_eager | 512 | 1 | 2 | 12.7908 |  | 1.5993 |
| inference | mace_ictc_bridge_u_aoti | 512 | 1 | 2 | 2.42543 | 29.7587 | 8.43409 |
| train | mace_ictc_cueq_product_eager | 512 | 1 | 2 | 41.5559 |  | 7.23043 |
| train | mace_ictc_cueq_product_makefx_train | 512 | 1 | 2 | 15.8426 | 42.3995 | 18.9658 |
| inference | mace_ictc_cueq_product_eager | 512 | 1 | 2 | 15.7004 |  | 1.30292 |
| inference | mace_ictc_cueq_product_aoti | 512 | 1 | 2 | 2.73452 | 22.4388 | 7.48076 |
| train | mace_torch_e3nn | 512 | 2 | 3 | 378.995 |  | 1 |
| inference | mace_torch_e3nn | 512 | 2 | 3 | 43.9097 |  | 1 |
| train | mace_torch_cueq | 512 | 2 | 3 | 54.9283 |  | 6.89981 |
| inference | mace_torch_cueq | 512 | 2 | 3 | 23.765 |  | 1.84766 |
| train | mace_ictc_bridge_u_eager | 512 | 2 | 3 | 52.0692 |  | 7.27868 |
| train | mace_ictc_bridge_u_makefx_train | 512 | 2 | 3 | 20.11 | 77.5214 | 18.8461 |
| inference | mace_ictc_bridge_u_eager | 512 | 2 | 3 | 17.72 |  | 2.47797 |
| inference | mace_ictc_bridge_u_aoti | 512 | 2 | 3 | 6.06463 | 36.8882 | 7.24029 |
| train | mace_ictc_cueq_product_eager | 512 | 2 | 3 | 59.3788 |  | 6.38267 |
| train | mace_ictc_cueq_product_makefx_train | 512 | 2 | 3 | 22.845 | 74.7572 | 16.5898 |
| inference | mace_ictc_cueq_product_eager | 512 | 2 | 3 | 22.3056 |  | 1.96855 |
| inference | mace_ictc_cueq_product_aoti | 512 | 2 | 3 | 6.68581 | 32.7709 | 6.5676 |
