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
- atoms: `256`
- avg_degree: `16`
- channels: `16`
- num_interactions: `2`
- correlation: `2`
- configs: `1:1`
- train_iters: `2`
- infer_iters: `5`
- aoti: `True`
- include_pure_u: `False`

## OK rows

| task | mode | atoms | lmax | max_ell | ms | compile_s | speedup_vs_e3nn |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | mace_torch_e3nn | 256 | 1 | 1 | 153.397 |  | 1 |
| inference | mace_torch_e3nn | 256 | 1 | 1 | 11.8562 |  | 1 |
| train | mace_torch_cueq | 256 | 1 | 1 | 36.209 |  | 4.23643 |
| inference | mace_torch_cueq | 256 | 1 | 1 | 16.3641 |  | 0.724525 |
| train | mace_ictc_bridge_u_eager | 256 | 1 | 1 | 27.716 |  | 5.5346 |
| train | mace_ictc_bridge_u_makefx_train | 256 | 1 | 1 | 9.70897 | 31.8136 | 15.7995 |
| inference | mace_ictc_bridge_u_eager | 256 | 1 | 1 | 10.6046 |  | 1.11802 |
| inference | mace_ictc_bridge_u_aoti | 256 | 1 | 1 | 1.25363 | 21.9815 | 9.4575 |
| train | mace_ictc_cueq_product_eager | 256 | 1 | 1 | 34.5298 |  | 4.44245 |
| train | mace_ictc_cueq_product_makefx_train | 256 | 1 | 1 | 13.0493 | 29.2287 | 11.7552 |
| inference | mace_ictc_cueq_product_eager | 256 | 1 | 1 | 12.8711 |  | 0.921149 |
| inference | mace_ictc_cueq_product_aoti | 256 | 1 | 1 | 2.33421 | 17.556 | 5.07932 |
