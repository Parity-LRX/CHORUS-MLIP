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
- atoms: `8192`
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
| train | mace_torch_e3nn | 8192 | 1 | 2 | 392.231 |  | 1 |
| inference | mace_torch_e3nn | 8192 | 1 | 2 | 58.3022 |  | 1 |
| train | mace_torch_cueq | 8192 | 1 | 2 | 44.8986 |  | 8.73593 |
| inference | mace_torch_cueq | 8192 | 1 | 2 | 18.3356 |  | 3.17973 |
| train | mace_ictc_bridge_u_eager | 8192 | 1 | 2 | 233.351 |  | 1.68086 |
| train | mace_ictc_bridge_u_makefx_train | 8192 | 1 | 2 | 105.486 | 48.6875 | 3.71832 |
| inference | mace_ictc_bridge_u_eager | 8192 | 1 | 2 | 82.2642 |  | 0.708719 |
| inference | mace_ictc_bridge_u_aoti | 8192 | 1 | 2 | 34.4349 | 30.1986 | 1.69311 |
| train | mace_ictc_cueq_product_eager | 8192 | 1 | 2 | 220.929 |  | 1.77537 |
| train | mace_ictc_cueq_product_makefx_train | 8192 | 1 | 2 | 93.0181 | 45.9439 | 4.21672 |
| inference | mace_ictc_cueq_product_eager | 8192 | 1 | 2 | 77.7884 |  | 0.749497 |
| inference | mace_ictc_cueq_product_aoti | 8192 | 1 | 2 | 30.4028 | 24.0591 | 1.91766 |
| train | mace_torch_e3nn | 8192 | 2 | 3 | 870.931 |  | 1 |
| inference | mace_torch_e3nn | 8192 | 2 | 3 | 217.995 |  | 1 |
| train | mace_torch_cueq | 8192 | 2 | 3 | 90.8333 |  | 9.58823 |
| inference | mace_torch_cueq | 8192 | 2 | 3 | 30.3711 |  | 7.17771 |
| inference | mace_ictc_bridge_u_eager | 8192 | 2 | 3 | 273.624 |  | 0.796695 |
| inference | mace_ictc_bridge_u_aoti | 8192 | 2 | 3 | 93.9568 | 41.9357 | 2.32016 |
| inference | mace_ictc_cueq_product_eager | 8192 | 2 | 3 | 261.674 |  | 0.833079 |
| inference | mace_ictc_cueq_product_aoti | 8192 | 2 | 3 | 85.9662 | 37.4905 | 2.53582 |

## Skips and errors

| task | mode | atoms | lmax | max_ell | status | note/error |
|---|---:|---:|---:|---:|---|---|
| train | mace_ictc_bridge_u_eager | 8192 | 2 | 3 | error | OutOfMemoryError: CUDA out of memory. Tried to allocate 480.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 242.69 MiB is free. Including non-PyTorch memory, this process has 23.26 GiB memory in use. Of the allocated memory 22.19 GiB is allocated by PyTorch, and 656.69 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https |
| train | mace_ictc_bridge_u_makefx_train | 8192 | 2 | 3 | error | OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 308.69 MiB is free. Including non-PyTorch memory, this process has 23.20 GiB memory in use. Of the allocated memory 22.12 GiB is allocated by PyTorch, and 651.16 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https |
| train | mace_ictc_cueq_product_eager | 8192 | 2 | 3 | error | OutOfMemoryError: CUDA out of memory. Tried to allocate 480.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 294.69 MiB is free. Including non-PyTorch memory, this process has 23.21 GiB memory in use. Of the allocated memory 21.36 GiB is allocated by PyTorch, and 1.41 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https:/ |
| train | mace_ictc_cueq_product_makefx_train | 8192 | 2 | 3 | error | OutOfMemoryError: CUDA out of memory. Tried to allocate 544.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 394.69 MiB is free. Including non-PyTorch memory, this process has 23.12 GiB memory in use. Of the allocated memory 21.17 GiB is allocated by PyTorch, and 1.50 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https:/ |
