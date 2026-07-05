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
- atoms: `512,1024,2048,4096,8192`
- avg_degree: `16`
- channels: `64`
- num_interactions: `2`
- correlation: `2`
- configs: `1:1,2:2`
- train_iters: `2`
- infer_iters: `5`
- aoti: `True`
- include_pure_u: `False`

## OK rows

| task | mode | atoms | lmax | max_ell | ms | compile_s | speedup_vs_e3nn |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | mace_torch_e3nn | 512 | 1 | 1 | 152.101 |  | 1 |
| inference | mace_torch_e3nn | 512 | 1 | 1 | 11.9106 |  | 1 |
| train | mace_torch_cueq | 512 | 1 | 1 | 36.7683 |  | 4.13674 |
| inference | mace_torch_cueq | 512 | 1 | 1 | 16.5686 |  | 0.718866 |
| train | mace_ictc_bridge_u_eager | 512 | 1 | 1 | 27.6823 |  | 5.49452 |
| train | mace_ictc_bridge_u_makefx_train | 512 | 1 | 1 | 9.72621 | 32.1413 | 15.6383 |
| inference | mace_ictc_bridge_u_eager | 512 | 1 | 1 | 10.5308 |  | 1.13103 |
| inference | mace_ictc_bridge_u_aoti | 512 | 1 | 1 | 1.59233 | 22.8183 | 7.47998 |
| train | mace_ictc_cueq_product_eager | 512 | 1 | 1 | 34.5954 |  | 4.39657 |
| train | mace_ictc_cueq_product_makefx_train | 512 | 1 | 1 | 13.1187 | 29.2805 | 11.5942 |
| inference | mace_ictc_cueq_product_eager | 512 | 1 | 1 | 12.7818 |  | 0.931841 |
| inference | mace_ictc_cueq_product_aoti | 512 | 1 | 1 | 2.34281 | 18.2177 | 5.08389 |
| train | mace_torch_e3nn | 512 | 2 | 2 | 113.063 |  | 1 |
| inference | mace_torch_e3nn | 512 | 2 | 2 | 34.2013 |  | 1 |
| train | mace_torch_cueq | 512 | 2 | 2 | 45.2216 |  | 2.5002 |
| inference | mace_torch_cueq | 512 | 2 | 2 | 20.2678 |  | 1.68747 |
| train | mace_ictc_bridge_u_eager | 512 | 2 | 2 | 39.4633 |  | 2.86502 |
| train | mace_ictc_bridge_u_makefx_train | 512 | 2 | 2 | 13.7447 | 57.7028 | 8.22593 |
| inference | mace_ictc_bridge_u_eager | 512 | 2 | 2 | 14.5717 |  | 2.3471 |
| inference | mace_ictc_bridge_u_aoti | 512 | 2 | 2 | 3.76024 | 29.6334 | 9.09551 |
| train | mace_ictc_cueq_product_eager | 512 | 2 | 2 | 49.6787 |  | 2.27588 |
| train | mace_ictc_cueq_product_makefx_train | 512 | 2 | 2 | 18.6716 | 57.965 | 6.05535 |
| inference | mace_ictc_cueq_product_eager | 512 | 2 | 2 | 18.5544 |  | 1.8433 |
| inference | mace_ictc_cueq_product_aoti | 512 | 2 | 2 | 4.17324 | 27.578 | 8.19538 |
| train | mace_torch_e3nn | 1024 | 1 | 1 | 45.757 |  | 1 |
| inference | mace_torch_e3nn | 1024 | 1 | 1 | 12.0292 |  | 1 |
| train | mace_torch_cueq | 1024 | 1 | 1 | 36.4653 |  | 1.25481 |
| inference | mace_torch_cueq | 1024 | 1 | 1 | 16.7274 |  | 0.719131 |
| train | mace_ictc_bridge_u_eager | 1024 | 1 | 1 | 28.1427 |  | 1.62589 |
| train | mace_ictc_bridge_u_makefx_train | 1024 | 1 | 1 | 10.2053 | 29.8895 | 4.48365 |
| inference | mace_ictc_bridge_u_eager | 1024 | 1 | 1 | 10.7251 |  | 1.12159 |
| inference | mace_ictc_bridge_u_aoti | 1024 | 1 | 1 | 2.58346 | 20.79 | 4.65624 |
| train | mace_ictc_cueq_product_eager | 1024 | 1 | 1 | 34.4241 |  | 1.32921 |
| train | mace_ictc_cueq_product_makefx_train | 1024 | 1 | 1 | 13.4779 | 27.3128 | 3.39497 |
| inference | mace_ictc_cueq_product_eager | 1024 | 1 | 1 | 13.124 |  | 0.91658 |
| inference | mace_ictc_cueq_product_aoti | 1024 | 1 | 1 | 2.48748 | 17.475 | 4.8359 |
| train | mace_torch_e3nn | 1024 | 2 | 2 | 114.693 |  | 1 |
| inference | mace_torch_e3nn | 1024 | 2 | 2 | 34.3576 |  | 1 |
| train | mace_torch_cueq | 1024 | 2 | 2 | 45.2641 |  | 2.53386 |
| inference | mace_torch_cueq | 1024 | 2 | 2 | 20.3166 |  | 1.69111 |
| train | mace_ictc_bridge_u_eager | 1024 | 2 | 2 | 52.8306 |  | 2.17096 |
| train | mace_ictc_bridge_u_makefx_train | 1024 | 2 | 2 | 22.068 | 55.7353 | 5.19725 |
| inference | mace_ictc_bridge_u_eager | 1024 | 2 | 2 | 17.8733 |  | 1.92229 |
| inference | mace_ictc_bridge_u_aoti | 1024 | 2 | 2 | 7.12203 | 30.7214 | 4.82413 |
| train | mace_ictc_cueq_product_eager | 1024 | 2 | 2 | 57.289 |  | 2.00201 |
| train | mace_ictc_cueq_product_makefx_train | 1024 | 2 | 2 | 22.487 | 54.5705 | 5.10041 |
| inference | mace_ictc_cueq_product_eager | 1024 | 2 | 2 | 19.0691 |  | 1.80174 |
| inference | mace_ictc_cueq_product_aoti | 1024 | 2 | 2 | 7.08466 | 28.8627 | 4.84958 |
| train | mace_torch_e3nn | 2048 | 1 | 1 | 45.7581 |  | 1 |
| inference | mace_torch_e3nn | 2048 | 1 | 1 | 12.1227 |  | 1 |
| train | mace_torch_cueq | 2048 | 1 | 1 | 37.2877 |  | 1.22716 |
| inference | mace_torch_cueq | 2048 | 1 | 1 | 16.8779 |  | 0.718259 |
| train | mace_ictc_bridge_u_eager | 2048 | 1 | 1 | 31.565 |  | 1.44965 |
| train | mace_ictc_bridge_u_makefx_train | 2048 | 1 | 1 | 15.323 | 30.0562 | 2.98624 |
| inference | mace_ictc_bridge_u_eager | 2048 | 1 | 1 | 10.7434 |  | 1.12839 |
| inference | mace_ictc_bridge_u_aoti | 2048 | 1 | 1 | 4.64029 | 22.569 | 2.61249 |
| train | mace_ictc_cueq_product_eager | 2048 | 1 | 1 | 35.3621 |  | 1.29399 |
| train | mace_ictc_cueq_product_makefx_train | 2048 | 1 | 1 | 14.3953 | 29.3585 | 3.17868 |
| inference | mace_ictc_cueq_product_eager | 2048 | 1 | 1 | 12.9794 |  | 0.933995 |
| inference | mace_ictc_cueq_product_aoti | 2048 | 1 | 1 | 3.89276 | 20.7156 | 3.11417 |
| train | mace_torch_e3nn | 2048 | 2 | 2 | 144.893 |  | 1 |
| inference | mace_torch_e3nn | 2048 | 2 | 2 | 41.588 |  | 1 |
| train | mace_torch_cueq | 2048 | 2 | 2 | 45.2005 |  | 3.20556 |
| inference | mace_torch_cueq | 2048 | 2 | 2 | 20.0843 |  | 2.07067 |
| train | mace_ictc_bridge_u_eager | 2048 | 2 | 2 | 96.7562 |  | 1.49751 |
| train | mace_ictc_bridge_u_makefx_train | 2048 | 2 | 2 | 40.4694 | 59.7983 | 3.58031 |
| inference | mace_ictc_bridge_u_eager | 2048 | 2 | 2 | 34.1028 |  | 1.21949 |
| inference | mace_ictc_bridge_u_aoti | 2048 | 2 | 2 | 13.8342 | 30.0728 | 3.00617 |
| train | mace_ictc_cueq_product_eager | 2048 | 2 | 2 | 104.009 |  | 1.39308 |
| train | mace_ictc_cueq_product_makefx_train | 2048 | 2 | 2 | 38.2009 | 57.232 | 3.79292 |
| inference | mace_ictc_cueq_product_eager | 2048 | 2 | 2 | 34.5012 |  | 1.20541 |
| inference | mace_ictc_cueq_product_aoti | 2048 | 2 | 2 | 12.6855 | 25.8844 | 3.27839 |
| train | mace_torch_e3nn | 4096 | 1 | 1 | 56.6858 |  | 1 |
| inference | mace_torch_e3nn | 4096 | 1 | 1 | 12.697 |  | 1 |
| train | mace_torch_cueq | 4096 | 1 | 1 | 36.4977 |  | 1.55313 |
| inference | mace_torch_cueq | 4096 | 1 | 1 | 16.5854 |  | 0.765553 |
| train | mace_ictc_bridge_u_eager | 4096 | 1 | 1 | 56.2066 |  | 1.00853 |
| train | mace_ictc_bridge_u_makefx_train | 4096 | 1 | 1 | 28.3508 | 34.1529 | 1.99944 |
| inference | mace_ictc_bridge_u_eager | 4096 | 1 | 1 | 18.1998 |  | 0.697645 |
| inference | mace_ictc_bridge_u_aoti | 4096 | 1 | 1 | 9.58734 | 20.6372 | 1.32435 |
| train | mace_ictc_cueq_product_eager | 4096 | 1 | 1 | 54.0679 |  | 1.04842 |
| train | mace_ictc_cueq_product_makefx_train | 4096 | 1 | 1 | 23.6666 | 27.2258 | 2.39518 |
| inference | mace_ictc_cueq_product_eager | 4096 | 1 | 1 | 17.7112 |  | 0.716891 |
| inference | mace_ictc_cueq_product_aoti | 4096 | 1 | 1 | 7.49289 | 18.0578 | 1.69454 |
| train | mace_torch_e3nn | 4096 | 2 | 2 | 218.862 |  | 1 |
| inference | mace_torch_e3nn | 4096 | 2 | 2 | 65.0103 |  | 1 |
| train | mace_torch_cueq | 4096 | 2 | 2 | 45.2526 |  | 4.83645 |
| inference | mace_torch_cueq | 4096 | 2 | 2 | 20.1817 |  | 3.22125 |
| train | mace_ictc_bridge_u_eager | 4096 | 2 | 2 | 193.964 |  | 1.12836 |
| train | mace_ictc_bridge_u_makefx_train | 4096 | 2 | 2 | 80.8933 | 64.5293 | 2.70556 |
| inference | mace_ictc_bridge_u_eager | 4096 | 2 | 2 | 68.0178 |  | 0.955784 |
| inference | mace_ictc_bridge_u_aoti | 4096 | 2 | 2 | 27.7022 | 31.1004 | 2.34676 |
| train | mace_ictc_cueq_product_eager | 4096 | 2 | 2 | 188.153 |  | 1.16321 |
| train | mace_ictc_cueq_product_makefx_train | 4096 | 2 | 2 | 73.1045 | 60.0017 | 2.99382 |
| inference | mace_ictc_cueq_product_eager | 4096 | 2 | 2 | 66.7699 |  | 0.973647 |
| inference | mace_ictc_cueq_product_aoti | 4096 | 2 | 2 | 24.3444 | 27.0168 | 2.67044 |
| train | mace_torch_e3nn | 8192 | 1 | 1 | 89.0223 |  | 1 |
| inference | mace_torch_e3nn | 8192 | 1 | 1 | 22.6285 |  | 1 |
| train | mace_torch_cueq | 8192 | 1 | 1 | 36.4834 |  | 2.44008 |
| inference | mace_torch_cueq | 8192 | 1 | 1 | 16.8957 |  | 1.33931 |
| train | mace_ictc_bridge_u_eager | 8192 | 1 | 1 | 111.012 |  | 0.801916 |
| train | mace_ictc_bridge_u_makefx_train | 8192 | 1 | 1 | 56.7381 | 36.2634 | 1.569 |
| inference | mace_ictc_bridge_u_eager | 8192 | 1 | 1 | 36.1909 |  | 0.625254 |
| inference | mace_ictc_bridge_u_aoti | 8192 | 1 | 1 | 18.7809 | 21.1487 | 1.20487 |
| train | mace_ictc_cueq_product_eager | 8192 | 1 | 1 | 102.614 |  | 0.867545 |
| train | mace_ictc_cueq_product_makefx_train | 8192 | 1 | 1 | 46.2744 | 27.7988 | 1.92379 |
| inference | mace_ictc_cueq_product_eager | 8192 | 1 | 1 | 33.1405 |  | 0.682805 |
| inference | mace_ictc_cueq_product_aoti | 8192 | 1 | 1 | 15.2916 | 18.7401 | 1.4798 |
| train | mace_torch_e3nn | 8192 | 2 | 2 | 374.558 |  | 1 |
| inference | mace_torch_e3nn | 8192 | 2 | 2 | 113.951 |  | 1 |
| train | mace_torch_cueq | 8192 | 2 | 2 | 55.6126 |  | 6.73513 |
| inference | mace_torch_cueq | 8192 | 2 | 2 | 20.6278 |  | 5.52415 |
| train | mace_ictc_bridge_u_eager | 8192 | 2 | 2 | 395.536 |  | 0.946963 |
| train | mace_ictc_bridge_u_makefx_train | 8192 | 2 | 2 | 168.481 | 68.5192 | 2.22315 |
| inference | mace_ictc_bridge_u_eager | 8192 | 2 | 2 | 141.075 |  | 0.807733 |
| inference | mace_ictc_bridge_u_aoti | 8192 | 2 | 2 | 55.3074 | 32.0215 | 2.06032 |
| train | mace_ictc_cueq_product_eager | 8192 | 2 | 2 | 376.563 |  | 0.994676 |
| train | mace_ictc_cueq_product_makefx_train | 8192 | 2 | 2 | 150.209 | 62.1287 | 2.49358 |
| inference | mace_ictc_cueq_product_eager | 8192 | 2 | 2 | 134.815 |  | 0.84524 |
| inference | mace_ictc_cueq_product_aoti | 8192 | 2 | 2 | 49.24 | 28.3946 | 2.3142 |
