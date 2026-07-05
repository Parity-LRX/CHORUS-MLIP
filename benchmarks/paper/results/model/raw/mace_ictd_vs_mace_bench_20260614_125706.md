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
- atoms: `16,32,64,128,256,512`
- avg_degree: `16`
- channels: `8`
- num_interactions: `2`
- correlation: `2`
- configs: `1:2,2:3`
- train_iters: `3`
- infer_iters: `10`
- aoti: `True`
- include_pure_u: `False`

## OK rows

| task | mode | atoms | lmax | max_ell | ms | compile_s | speedup_vs_e3nn |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | mace_torch_e3nn | 16 | 1 | 2 | 290.224 |  | 1 |
| inference | mace_torch_e3nn | 16 | 1 | 2 | 16.7521 |  | 1 |
| train | mace_torch_cueq | 16 | 1 | 2 | 41.1034 |  | 7.06083 |
| inference | mace_torch_cueq | 16 | 1 | 2 | 18.2562 |  | 0.917612 |
| train | mace_ictc_bridge_u_eager | 16 | 1 | 2 | 32.8717 |  | 8.82899 |
| train | mace_ictc_bridge_u_makefx_train | 16 | 1 | 2 | 11.3335 | 40.2706 | 25.6076 |
| inference | mace_ictc_bridge_u_eager | 16 | 1 | 2 | 12.5002 |  | 1.34015 |
| inference | mace_ictc_bridge_u_aoti | 16 | 1 | 2 | 1.41965 | 26.8825 | 11.8002 |
| train | mace_ictc_cueq_product_eager | 16 | 1 | 2 | 40.7169 |  | 7.12785 |
| train | mace_ictc_cueq_product_makefx_train | 16 | 1 | 2 | 15.1526 | 38.3292 | 19.1534 |
| inference | mace_ictc_cueq_product_eager | 16 | 1 | 2 | 15.3964 |  | 1.08805 |
| inference | mace_ictc_cueq_product_aoti | 16 | 1 | 2 | 2.65058 | 21.6261 | 6.32016 |
| train | mace_torch_e3nn | 16 | 2 | 3 | 273.909 |  | 1 |
| inference | mace_torch_e3nn | 16 | 2 | 3 | 32.0158 |  | 1 |
| train | mace_torch_cueq | 16 | 2 | 3 | 53.8955 |  | 5.08222 |
| inference | mace_torch_cueq | 16 | 2 | 3 | 23.5426 |  | 1.35991 |
| train | mace_ictc_bridge_u_eager | 16 | 2 | 3 | 46.082 |  | 5.94395 |
| train | mace_ictc_bridge_u_makefx_train | 16 | 2 | 3 | 15.0622 | 70.5818 | 18.1852 |
| inference | mace_ictc_bridge_u_eager | 16 | 2 | 3 | 17.3777 |  | 1.84235 |
| inference | mace_ictc_bridge_u_aoti | 16 | 2 | 3 | 1.95674 | 33.3973 | 16.3618 |
| train | mace_ictc_cueq_product_eager | 16 | 2 | 3 | 57.4086 |  | 4.77122 |
| train | mace_ictc_cueq_product_makefx_train | 16 | 2 | 3 | 21.7988 | 66.502 | 12.5653 |
| inference | mace_ictc_cueq_product_eager | 16 | 2 | 3 | 21.7959 |  | 1.46889 |
| inference | mace_ictc_cueq_product_aoti | 16 | 2 | 3 | 4.12129 | 29.6295 | 7.76839 |
| train | mace_torch_e3nn | 32 | 1 | 2 | 182.816 |  | 1 |
| inference | mace_torch_e3nn | 32 | 1 | 2 | 16.9801 |  | 1 |
| train | mace_torch_cueq | 32 | 1 | 2 | 41.637 |  | 4.39071 |
| inference | mace_torch_cueq | 32 | 1 | 2 | 18.6357 |  | 0.91116 |
| train | mace_ictc_bridge_u_eager | 32 | 1 | 2 | 33.3042 |  | 5.48928 |
| train | mace_ictc_bridge_u_makefx_train | 32 | 1 | 2 | 11.8585 | 39.1066 | 15.4165 |
| inference | mace_ictc_bridge_u_eager | 32 | 1 | 2 | 13.2211 |  | 1.28432 |
| inference | mace_ictc_bridge_u_aoti | 32 | 1 | 2 | 1.40325 | 23.3299 | 12.1006 |
| train | mace_ictc_cueq_product_eager | 32 | 1 | 2 | 39.9576 |  | 4.57525 |
| train | mace_ictc_cueq_product_makefx_train | 32 | 1 | 2 | 15.2945 | 38.9741 | 11.9531 |
| inference | mace_ictc_cueq_product_eager | 32 | 1 | 2 | 15.6615 |  | 1.08419 |
| inference | mace_ictc_cueq_product_aoti | 32 | 1 | 2 | 2.62235 | 21.1601 | 6.47515 |
| train | mace_torch_e3nn | 32 | 2 | 3 | 166.522 |  | 1 |
| inference | mace_torch_e3nn | 32 | 2 | 3 | 32.7688 |  | 1 |
| train | mace_torch_cueq | 32 | 2 | 3 | 53.6564 |  | 3.10349 |
| inference | mace_torch_cueq | 32 | 2 | 3 | 24.0694 |  | 1.36143 |
| train | mace_ictc_bridge_u_eager | 32 | 2 | 3 | 46.0348 |  | 3.61731 |
| train | mace_ictc_bridge_u_makefx_train | 32 | 2 | 3 | 15.266 | 69.5167 | 10.908 |
| inference | mace_ictc_bridge_u_eager | 32 | 2 | 3 | 17.7493 |  | 1.8462 |
| inference | mace_ictc_bridge_u_aoti | 32 | 2 | 3 | 1.9727 | 35.7602 | 16.6111 |
| train | mace_ictc_cueq_product_eager | 32 | 2 | 3 | 57.1446 |  | 2.91405 |
| train | mace_ictc_cueq_product_makefx_train | 32 | 2 | 3 | 21.4236 | 62.3127 | 7.77283 |
| inference | mace_ictc_cueq_product_eager | 32 | 2 | 3 | 21.9858 |  | 1.49045 |
| inference | mace_ictc_cueq_product_aoti | 32 | 2 | 3 | 4.23927 | 33.0758 | 7.72982 |
| train | mace_torch_e3nn | 64 | 1 | 2 | 126.895 |  | 1 |
| inference | mace_torch_e3nn | 64 | 1 | 2 | 16.6004 |  | 1 |
| train | mace_torch_cueq | 64 | 1 | 2 | 41.347 |  | 3.06903 |
| inference | mace_torch_cueq | 64 | 1 | 2 | 18.6999 |  | 0.887727 |
| train | mace_ictc_bridge_u_eager | 64 | 1 | 2 | 33.1692 |  | 3.82569 |
| train | mace_ictc_bridge_u_makefx_train | 64 | 1 | 2 | 11.2676 | 12.0231 | 11.2619 |
| inference | mace_ictc_bridge_u_eager | 64 | 1 | 2 | 12.7943 |  | 1.29748 |
| inference | mace_ictc_bridge_u_aoti | 64 | 1 | 2 | 1.49222 | 22.3534 | 11.1246 |
| train | mace_ictc_cueq_product_eager | 64 | 1 | 2 | 41.2512 |  | 3.07615 |
| train | mace_ictc_cueq_product_makefx_train | 64 | 1 | 2 | 15.124 | 12.0277 | 8.39031 |
| inference | mace_ictc_cueq_product_eager | 64 | 1 | 2 | 15.6459 |  | 1.06101 |
| inference | mace_ictc_cueq_product_aoti | 64 | 1 | 2 | 2.67839 | 16.4548 | 6.1979 |
| train | mace_torch_e3nn | 64 | 2 | 3 | 151.826 |  | 1 |
| inference | mace_torch_e3nn | 64 | 2 | 3 | 32.2507 |  | 1 |
| train | mace_torch_cueq | 64 | 2 | 3 | 53.3575 |  | 2.84545 |
| inference | mace_torch_cueq | 64 | 2 | 3 | 23.8209 |  | 1.35388 |
| train | mace_ictc_bridge_u_eager | 64 | 2 | 3 | 46.3394 |  | 3.27639 |
| train | mace_ictc_bridge_u_makefx_train | 64 | 2 | 3 | 15.2599 | 22.2542 | 9.94934 |
| inference | mace_ictc_bridge_u_eager | 64 | 2 | 3 | 18.0175 |  | 1.78997 |
| inference | mace_ictc_bridge_u_aoti | 64 | 2 | 3 | 1.98039 | 26.6489 | 16.285 |
| train | mace_ictc_cueq_product_eager | 64 | 2 | 3 | 57.3178 |  | 2.64885 |
| train | mace_ictc_cueq_product_makefx_train | 64 | 2 | 3 | 21.5532 | 22.2996 | 7.04424 |
| inference | mace_ictc_cueq_product_eager | 64 | 2 | 3 | 22.7741 |  | 1.41611 |
| inference | mace_ictc_cueq_product_aoti | 64 | 2 | 3 | 4.14619 | 23.5962 | 7.77839 |
| train | mace_torch_e3nn | 128 | 1 | 2 | 57.6104 |  | 1 |
| inference | mace_torch_e3nn | 128 | 1 | 2 | 16.649 |  | 1 |
| train | mace_torch_cueq | 128 | 1 | 2 | 41.1908 |  | 1.39862 |
| inference | mace_torch_cueq | 128 | 1 | 2 | 18.5015 |  | 0.899873 |
| train | mace_ictc_bridge_u_eager | 128 | 1 | 2 | 32.9224 |  | 1.74988 |
| train | mace_ictc_bridge_u_makefx_train | 128 | 1 | 2 | 11.8757 | 42.2909 | 4.85112 |
| inference | mace_ictc_bridge_u_eager | 128 | 1 | 2 | 13.1416 |  | 1.26689 |
| inference | mace_ictc_bridge_u_aoti | 128 | 1 | 2 | 1.49386 | 23.8528 | 11.145 |
| train | mace_ictc_cueq_product_eager | 128 | 1 | 2 | 40.4255 |  | 1.4251 |
| train | mace_ictc_cueq_product_makefx_train | 128 | 1 | 2 | 15.5192 | 35.8266 | 3.7122 |
| inference | mace_ictc_cueq_product_eager | 128 | 1 | 2 | 15.6958 |  | 1.06073 |
| inference | mace_ictc_cueq_product_aoti | 128 | 1 | 2 | 2.91322 | 26.37 | 5.71498 |
| train | mace_torch_e3nn | 128 | 2 | 3 | 114.222 |  | 1 |
| inference | mace_torch_e3nn | 128 | 2 | 3 | 32.3587 |  | 1 |
| train | mace_torch_cueq | 128 | 2 | 3 | 53.2813 |  | 2.14375 |
| inference | mace_torch_cueq | 128 | 2 | 3 | 23.7073 |  | 1.36493 |
| train | mace_ictc_bridge_u_eager | 128 | 2 | 3 | 46.1786 |  | 2.47348 |
| train | mace_ictc_bridge_u_makefx_train | 128 | 2 | 3 | 15.9003 | 73.9664 | 7.18364 |
| inference | mace_ictc_bridge_u_eager | 128 | 2 | 3 | 18.3998 |  | 1.75864 |
| inference | mace_ictc_bridge_u_aoti | 128 | 2 | 3 | 1.98008 | 34.271 | 16.3421 |
| train | mace_ictc_cueq_product_eager | 128 | 2 | 3 | 57.0093 |  | 2.00357 |
| train | mace_ictc_cueq_product_makefx_train | 128 | 2 | 3 | 22.2454 | 66.5849 | 5.13463 |
| inference | mace_ictc_cueq_product_eager | 128 | 2 | 3 | 22.4284 |  | 1.44276 |
| inference | mace_ictc_cueq_product_aoti | 128 | 2 | 3 | 4.17281 | 30.6571 | 7.75465 |
| train | mace_torch_e3nn | 256 | 1 | 2 | 57.8718 |  | 1 |
| inference | mace_torch_e3nn | 256 | 1 | 2 | 16.8532 |  | 1 |
| train | mace_torch_cueq | 256 | 1 | 2 | 41.9362 |  | 1.38 |
| inference | mace_torch_cueq | 256 | 1 | 2 | 18.8852 |  | 0.892403 |
| train | mace_ictc_bridge_u_eager | 256 | 1 | 2 | 33.2186 |  | 1.74215 |
| train | mace_ictc_bridge_u_makefx_train | 256 | 1 | 2 | 11.7717 | 38.3543 | 4.91618 |
| inference | mace_ictc_bridge_u_eager | 256 | 1 | 2 | 12.9523 |  | 1.30117 |
| inference | mace_ictc_bridge_u_aoti | 256 | 1 | 2 | 1.55388 | 29.8876 | 10.8459 |
| train | mace_ictc_cueq_product_eager | 256 | 1 | 2 | 42.471 |  | 1.36262 |
| train | mace_ictc_cueq_product_makefx_train | 256 | 1 | 2 | 15.752 | 36.6249 | 3.67393 |
| inference | mace_ictc_cueq_product_eager | 256 | 1 | 2 | 15.8165 |  | 1.06555 |
| inference | mace_ictc_cueq_product_aoti | 256 | 1 | 2 | 2.73444 | 21.7655 | 6.16331 |
| train | mace_torch_e3nn | 256 | 2 | 3 | 113.739 |  | 1 |
| inference | mace_torch_e3nn | 256 | 2 | 3 | 32.4538 |  | 1 |
| train | mace_torch_cueq | 256 | 2 | 3 | 53.1136 |  | 2.14143 |
| inference | mace_torch_cueq | 256 | 2 | 3 | 23.5349 |  | 1.37896 |
| train | mace_ictc_bridge_u_eager | 256 | 2 | 3 | 46.283 |  | 2.45747 |
| train | mace_ictc_bridge_u_makefx_train | 256 | 2 | 3 | 15.9261 | 76.4607 | 7.14167 |
| inference | mace_ictc_bridge_u_eager | 256 | 2 | 3 | 18.4082 |  | 1.76301 |
| inference | mace_ictc_bridge_u_aoti | 256 | 2 | 3 | 2.00608 | 34.3689 | 16.1777 |
| train | mace_ictc_cueq_product_eager | 256 | 2 | 3 | 57.439 |  | 1.98017 |
| train | mace_ictc_cueq_product_makefx_train | 256 | 2 | 3 | 22.4554 | 68.5305 | 5.06511 |
| inference | mace_ictc_cueq_product_eager | 256 | 2 | 3 | 23.1447 |  | 1.40221 |
| inference | mace_ictc_cueq_product_aoti | 256 | 2 | 3 | 4.24783 | 31.1379 | 7.64009 |
| train | mace_torch_e3nn | 512 | 1 | 2 | 58.039 |  | 1 |
| inference | mace_torch_e3nn | 512 | 1 | 2 | 16.7958 |  | 1 |
| train | mace_torch_cueq | 512 | 1 | 2 | 41.649 |  | 1.39353 |
| inference | mace_torch_cueq | 512 | 1 | 2 | 18.8988 |  | 0.888723 |
| train | mace_ictc_bridge_u_eager | 512 | 1 | 2 | 33.4775 |  | 1.73367 |
| train | mace_ictc_bridge_u_makefx_train | 512 | 1 | 2 | 11.7632 | 37.8585 | 4.93395 |
| inference | mace_ictc_bridge_u_eager | 512 | 1 | 2 | 12.8486 |  | 1.30721 |
| inference | mace_ictc_bridge_u_aoti | 512 | 1 | 2 | 1.4656 | 23.7043 | 11.46 |
| train | mace_ictc_cueq_product_eager | 512 | 1 | 2 | 40.7028 |  | 1.42592 |
| train | mace_ictc_cueq_product_makefx_train | 512 | 1 | 2 | 16.109 | 42.5178 | 3.60289 |
| inference | mace_ictc_cueq_product_eager | 512 | 1 | 2 | 16.3717 |  | 1.0259 |
| inference | mace_ictc_cueq_product_aoti | 512 | 1 | 2 | 2.75001 | 21.6192 | 6.10754 |
| train | mace_torch_e3nn | 512 | 2 | 3 | 113.631 |  | 1 |
| inference | mace_torch_e3nn | 512 | 2 | 3 | 32.2504 |  | 1 |
| train | mace_torch_cueq | 512 | 2 | 3 | 53.8565 |  | 2.10988 |
| inference | mace_torch_cueq | 512 | 2 | 3 | 24.0302 |  | 1.34208 |
| train | mace_ictc_bridge_u_eager | 512 | 2 | 3 | 46.5344 |  | 2.44187 |
| train | mace_ictc_bridge_u_makefx_train | 512 | 2 | 3 | 15.7211 | 68.9197 | 7.22793 |
| inference | mace_ictc_bridge_u_eager | 512 | 2 | 3 | 17.9516 |  | 1.79652 |
| inference | mace_ictc_bridge_u_aoti | 512 | 2 | 3 | 2.07332 | 41.709 | 15.555 |
| train | mace_ictc_cueq_product_eager | 512 | 2 | 3 | 59.4039 |  | 1.91285 |
| train | mace_ictc_cueq_product_makefx_train | 512 | 2 | 3 | 21.9981 | 61.1675 | 5.16549 |
| inference | mace_ictc_cueq_product_eager | 512 | 2 | 3 | 22.3301 |  | 1.44426 |
| inference | mace_ictc_cueq_product_aoti | 512 | 2 | 3 | 4.20484 | 30.4603 | 7.66983 |
