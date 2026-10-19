# 径向 gate/伴随融合迭代：中显存档与 Final 数值路径

全部数值计算、计时、汇总通过 `sbatch / GPU-A100 / XHPC-A100`。固定原 checkpoint、全部参数、50 个截断邻居、FP32/TF32 off；virial 为 FP64。显存为十进制 GB，包含 CUDA Graph 私有池的 reserved。

## Persistent：保留本次前向的深层 gate

原生 TP/VJP 直接输出每边距离梯度，不物化完整 gate 梯度；深层保留 gate 值以避免重算，首层保留原径向网络并选用无多余输入伴随的调度。所有实时输入每次重新计算，不跨调用缓存 gate。默认最快档和全有界容量档均保留。

| 原子数 | 配置 | ms | reserved GB | 相对同卡 DPA lower |
| --- | --- | ---: | ---: | ---: |
| 10000 | 原最快默认 | 71.515 | 9.651 | 1.752× |
| 10000 | 保留 gate，tile=65536 | 72.603 | 6.872 | 1.725× |
| 10000 | 保留 gate，tile=262144 | 72.117 | 7.980 | 1.737× |
| 10000 | DPA4-mini compile + 实际 Triton 2 | 125.267 | 12.566 | 1.000× |
| 32768 | 原最快默认 | 233.489 | 31.896 | 1.736× |
| 32768 | 保留 gate，tile=65536 | 239.667 | 20.907 | 1.691× |
| 32768 | 保留 gate，tile=262144 | 237.289 | 22.788 | 1.708× |
| 32768 | DPA4-mini compile + 实际 Triton 2 | 405.309 | 40.943 | 1.000× |

DPA 的完整 full/lower 入口输出一致性在计时尺寸检查，未加速原式另在 1024 原子核对；CHORUS 对照为登记的全尺寸原式 E/F/virial，32,768 原子使用已核验的完整依赖 halo。通过点同时检查改变坐标后的实时重放。Persistent 的两倍 DPA 目标尚未完成。

65,536 原子保留 gate 档通过 eager 原式数值检查，但在 CUDA Graph 捕获时 OOM。该点不计作通过。全有界档继续用于更大图，原始失败和日志保留。

| 65,536 原子配置 | 状态 / ms | reserved GB |
| --- | ---: | ---: |
| default | OOM | — |
| public_cached_tiles_65536 | OOM | — |
| public_cached_tiles_262144 | OOM | — |
| public_destination_tiles_65536 | 573.184 | 17.555 |
| public_destination_tiles_262144 | 546.855 | 20.508 |

## Final：映射布局与力归约

先逐项关闭相位、密度、通道、激活、元素映射、谐波、几何和大图调度，再做组合对照。恢复原通道/激活/元素映射，并使用稳定 FP32 力归约的候选接受同一个登记参照及阈值。

| 对照 | 重复次数 | 力分量全部通过次数 | 单次超差分量范围 |
| --- | ---: | ---: | ---: |
| baseline | 5 | 0 | 1–4 |
| native_activations | 5 | 0 | 2–3 |
| native_channel | 5 | 0 | 1–2 |
| native_density | 5 | 0 | 1–4 |
| native_geometry | 5 | 0 | 1–2 |
| native_harmonics | 5 | 0 | 2–3 |
| native_phase | 5 | 0 | 2–4 |
| native_species | 5 | 0 | 2–4 |
| no_large_schedule | 5 | 1 | 0–2 |
| original_products | 5 | 1 | 0–2 |
| split_phase | 5 | 0 | 2–3 |
| unfactored_adjoint | 5 | 0 | 1–3 |
| native_first_maps | 5 | 0 | 1–1 |
| native_first_message | 5 | 1 | 0–3 |
| native_maps | 5 | 4 | 0–1 |
| only_messages | 5 | 3 | 0–1 |
| native_maps_forcecsr | 20 | 20 | 0–0 |
| native_activation_species_forcecsr | 20 | 0 | 1–3 |
| native_activations_forcecsr | 20 | 0 | 1–3 |
| native_channel_activation_forcecsr | 20 | 20 | 0–0 |
| native_channel_forcecsr | 20 | 11 | 0–2 |
| native_channel_species_forcecsr | 20 | 20 | 0–0 |
| native_species_forcecsr | 20 | 0 | 1–4 |

该组合同时改变映射计算布局与最终力归约，尚不能把误差归于唯一一个算子。原默认路径的超差记录继续有效。以下为该显式候选的独立完整计时与原式/重放核验；不替换默认 Final，也不改写先前 10,000 原子的结果。

| 32,768 原子路径 | 同卡点 | 状态 / ms | reserved GB | 相对同卡 DPA lower |
| --- | --- | ---: | ---: | ---: |
| 原通道/激活/元素映射 + CSR 力 | dpa4_full | 415.019 | 41.054 | 0.989× |
| 原通道/激活/元素映射 + CSR 力 | dpa4_lower | 410.498 | 40.943 | 1.000× |
| 原通道/激活/元素映射 + CSR 力 | default | 315.437 | 33.865 | 1.301× |
| 原通道/激活/元素映射 + CSR 力 | none_forcecsr_tiles_65536 | 314.550 | 33.865 | 1.305× |
| 原通道/激活 + HP 元素映射 + CSR 力 | dpa4_full | 407.530 | 41.054 | 0.990× |
| 原通道/激活 + HP 元素映射 + CSR 力 | dpa4_lower | 403.620 | 40.943 | 1.000× |
| 原通道/激活 + HP 元素映射 + CSR 力 | default | 265.590 | 31.973 | 1.520× |
| 原通道/激活 + HP 元素映射 + CSR 力 | none_forcecsr_tiles_65536 | 265.748 | 31.973 | 1.519× |

## 回归与接口

整套远端回归：241 项，failure=0，error=0。16 帧 T1x 留出 E/F：PASS。覆盖固定/可微相位系数、第一及混合高阶导数、空边、通道尾部、实时参数更新、坐标/元素/拓扑/晶胞变化以及 CUDA Graph 状态失效。

新增公开开关为 `radial_cache_gates=True`，要求同时开启 `radial_jvp`、`radial_gate_pipeline` 和 `radial_destination`。该轮针对冻结推理，训练吞吐计时未更新。

原生启动块/路径分组筛选仅为算子级诊断，未形成明显收益，默认启动参数不变。模型和算子仍来自 ICTC-O3-HP，未引入外部 TP 库。

基础设施记录：source_v10 首次同步漏掉谐波基资产，子进程在模型加载时退出；后续完整快照修正。作业 92217 在 DPA 阶段取消，用独立输出目录修正 Slurm 逗号分隔环境参数的传递。两者均不计入数值通过或性能结果。
