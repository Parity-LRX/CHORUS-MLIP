# CHORUS 原生高性能执行：吞吐专项

本轮目标同时包含 Final 与 Persistent：保留原 C128/L2、correlation=3、全部参数、
checkpoint-U、相位/Hermitian 方程，在 ICTC-O3-HP 母版上优化执行；不引入 cuEq/OpenEq。

## 训练验收口径更新（2026-10-18）

训练性能只接受不使用 CUDA Graph capture/replay 的完整 E/F 损失、混合二阶反传
及优化器更新。收益必须来自算子实现、数据布局、计算图裁切或整网融合；允许编译，
但须关闭编译器内部的 CUDA Graph。输入坐标、元素及邻居表实时读取，并另测 N/E
变化的批次序列。冷启动单列，模型、可训练参数、精度和数值阈值保持不变。

下文历史 HP Graph 训练记录及其 1.46–1.58× DPA 加速比不再作为训练达标证据。
前轮无 Graph 的 HP eager 128/512 原子计时为 45.724/44.857 ms，同卡 DPA
compile 为 18.777/39.186 ms：目前尚未证明训练超过 DPA。原始记录不删除、不改写。
新协议的变化批次正反顺序复测中，旧 HP 为 44.197/44.700 ms，联合伴随候选为
43.928/44.869 ms，同卡 DPA 为 19.137/39.568 ms；候选没有明确整网收益，默认关闭。
协议、逐元素梯度异常和原始结果见[无 CUDA Graph 训练专项](TRAINING_NO_CUDAGRAPH.md)。

## 后续迭代：保留前向 gate 的中显存档

新增 `radial_cache_gates=True`：深层 gate 值在本次前反向内复用，TP/VJP 直接输出
边长梯度，省去完整 gate 梯度及 gate 值重算。首层保留原径向 MLP，并在匹配的
sm80 大图上使用不生成多余输入梯度的伴随。默认最快档与全有界容量档都保留。

同一卡上的正式 checkpoint、50 邻居、完整 E/F/virial 与坐标变化重放结果如下。
网络/力保持 FP32、TF32 off，virial 为 FP64，显存为包含 graph pool 的十进制 GB。

| 原子数 | 最快默认 ms / GB | 中显存档 ms / GB，tile=262144 | DPA lower ms / GB |
| --- | --- | --- | --- |
| 10,000 | 71.515 / 9.651 | 72.117 / 7.980 | 125.267 / 12.566 |
| 32,768 | 233.489 / 31.896 | 237.289 / 22.788 | 405.309 / 40.943 |

tile=65536 时，两尺寸为 72.603 ms / 6.872 GB、239.667 ms / 20.907 GB。
中显存档相对最快默认仅有小幅耗时增加，可在不采用全有界档的情况下减少显存；
Persistent 两倍 DPA4-mini compile＋实际 Triton 2 的目标仍未达到。
65,536 原子中显存档的 eager 原式核验通过，但 CUDA Graph 捕获 OOM；更大尺寸
继续使用下节全有界档。该轮完整回归 241 项及 16 帧 T1x 留出 E/F 核验通过。

Final 的逐项/组合对照找到稳定候选：保留 HP 元素映射，恢复原通道与激活映射，
配合稳定 FP32 力归约，32,768 原子连续 20 次检查全部力分量通过。
独立两次完整 E/F/virial 与坐标变化重放通过，用时 265.590/265.748 ms、
reserved 31.973 GB；同卡 DPA lower 为 403.620 ms / 40.943 GB，约 1.52×。
这是一条显式数值候选，尚未替换原快速默认路径，也未达到该尺寸的两倍目标。

[本轮逐点报告](results/large_capacity_a100/cached_gate_iteration/CACHED_GATE_ITERATION.md)
记录所有配置、原始误差、Final 分路径对照、OOM 和同卡计时。该轮仍是冻结推理
优化，没有更新训练吞吐结论。

## 更大原子数与有界显存径向伴随

已训练 C128/L2 Persistent 在 10,000–131,072 原子、每原子恰有 50 个截断邻居上，
完成原式完整 E/F/virial 与实时坐标重放核验。保留 seed=20260728 的几何构造、
原 checkpoint 和 1,029,967 个参数；网络及力为 FP32、TF32 off，virial 为 FP64。
本节显存统一使用十进制 GB，统计包含 CUDA Graph 私有池的 PyTorch reserved。

新路径沿距离传播径向 MLP 的值与切向量，分块生成 gate 及其距离导数；
原生 TP/VJP 当场收缩为每边一个距离梯度，不保留全图 gate 或全图 gate 梯度。
目标节点缓存伴随复用三流 cotangent，方向/相位 partial 只保留一个可复用边块。
全部 CG 路径、原 Hermitian/rank16 和跨层 charged 状态保持。

| 原子数 | 原最快默认 ms / GB | 新低显存路径 ms / GB |
| --- | --- | --- |
| 10,000 | 72.130 / 9.651 | 85.651 / 3.959 |
| 32,768 | 233.654 / 31.896 | 281.533 / 8.739 |
| 65,536 | 反向 gate 梯度分配 OOM | 555.796 / 17.555 |
| 100,000 | OOM | 872.166 / 26.523 |
| 131,072 | OOM | 1150.643 / 34.664 |

前两行新路径使用 tile=65,536，后三行使用 tile=131,072。
100,000 原子在同一作业、同一 A100 上，原有 source-CSR 低显存路径为
1018.621 ms，新目标端伴随为 872.166 ms，耗时约减少 14%；两者 reserved 相同。
131,072 原子的同卡 source/destination 对照为 1342.112/1150.643 ms，吞吐提高约 17%。
32,768 原子时，该路径较最快默认配置约慢 20%，reserved 减少约 73%。
因此它作为容量选项提供，原最快默认配置保持不变，Persistent 相对 DPA4-mini
的两倍吞吐目标仍未完成。

同卡 DPA4-mini C32 开启 compile、实际执行两个 level-2 `_TritonSO2ValuePath`：
32,768 原子 full/lower 为 409.585/405.371 ms，lower reserved 为 40.943 GB。
该大图检查了完整 full/lower 一致性，未加速原式另在 1024 原子核对；完整大图的
未加速原式覆盖不在此记录中。65,536 与 100,000 的官方优化入口在执行阶段 OOM。

更深的输出投影内联也完成数值核验，但没有吞吐收益：32,768 原子的仅前向内联/
前反向内联分别为 422.861/622.139 ms，未进入推荐配置。

公开可选接口为 `radial_jvp=True, radial_gate_pipeline=True, radial_destination=True`；
`radial_tile_edges` 控制分块。218 项整套回归、参数身份/可训练性检查及 16 帧
T1x 留出 E/F 核验通过。本轮是冻结推理的容量优化，训练路径与此前训练计时不变。

Final 在 32,768/65,536 原子的扩展存在近零力分量超差，默认路径重复检查也能出现
超差；这些扩展点未完成验收。独立局部原式审计记录了 FP32 原式求和变化，原有
阈值、原登记参照及失败记录均保留。此前 10,000 原子的 Final 五轮结果另列下文。

原始数据、核验与 profile 位于 `results/large_capacity_a100/`；
[完整大规模报告](results/large_capacity_a100/report_complete/LARGE_CAPACITY.md)与
[逐点汇总 JSON](results/large_capacity_a100/report_complete/capacity_summary.json)
包含可复核的通过、超差、OOM 和非默认候选。所有计算与汇总均使用
`sbatch/GPU-A100/XHPC-A100`，没有修改 MD、checkpoint 或模型配置。

## 角动量、通道及交互深度扩展（A100）

扩展矩阵覆盖 L2/L3、C128/C192/C256、2/3/4 个 interaction，包含单因素和联合增大；
两种 scope 均测试完整 E/F/virial。统一使用 seed=20261017 的固定初始化结构族，
correlation=3、rank16、平均邻居归一化 50；主矩阵为 1024 原子，每个原子恰有
50 个截断邻居，关键组合另测 10,000 原子。该结构族的参数量与后文已训练
checkpoint 不同，本节评估结构扩展的执行代价，新增尺寸尚未训练。

DPA4 对应使用 C32/C48/C64、相同 lmax 和 block 数，保留 mmax=1 及原 grid 分支。
两种模型的通道定义不同，完整报告同时列出实际参数量。所有配对均在同一作业、
同一物理 A100 上顺序测量；网络与力 FP32、TF32 off，virial FP64。
CHORUS 使用固定图 CUDA Graph；DPA 使用官方 compile full/lower 入口。
每点取 15 次同步 walltime 中位数，编译及原式参照均不计时。

以下为 DPA 实际执行 level-2 `_TritonSO2ValuePath` 的主矩阵点，耗时单位 ms；
加速比为预建边 DPA lower 耗时除以 CHORUS 耗时。

| CHORUS 配置 | DPA C | Final | Persistent | DPA lower | 加速比 Final / Persistent |
| --- | ---: | ---: | ---: | ---: | --- |
| L2 / C128 / 2 层 | 32 | 9.343 | 10.786 | 15.071 | 1.61 / 1.40× |
| L3 / C128 / 2 层 | 32 | 21.035 | 24.747 | 24.583 | 1.17 / 0.99× |
| L2 / C256 / 2 层 | 64 | 20.041 | 22.512 | 31.351 | 1.56 / 1.39× |
| L2 / C128 / 3 层 | 32 | 13.136 | 17.823 | 20.345 | 1.55 / 1.14× |
| L2 / C128 / 4 层 | 32 | 16.973 | 24.955 | 25.535 | 1.50 / 1.02× |
| L3 / C256 / 4 层 | 64 | 90.469 | 127.597 | 95.332 | 1.05 / 0.75× |

主矩阵 16 个 scope/结构点全部完成全尺寸原式数值核对及坐标变化重放检查。
最大 Persistent 的原式参照按原子分块计算局部 product，并重算原模块；先与
128 原子未分块原式核对，再完成 1024 原子参照。其全部力分量的最大绝对差为
3.51e-8、相对 L2 差为 1.78e-6，virial 相对 L2 差为 5.81e-8。
这个验证器不参与计时，原计时及此前验证 OOM 记录保留。

C192 配对的 DPA C48 虽设置了 level 2，但官方融合 value path 的支持宽度为
{32,64,96,128}，实际返回 `None`。compile 保持生效，C48 原始计时完整保留在报告中，
单独标为 T2-unavailable，不用于上述 level-2 融合加速比。

扩展趋势：C128→C256 时，Final/Persistent 耗时增加 2.15/2.09 倍，DPA 增加
2.08 倍；2→4 层时为 1.82/2.31 倍，对应 DPA 为 1.69 倍。L2→L3 的代价增长更快：
CHORUS 为 2.25/2.29 倍，DPA 为 1.63 倍。现有收益更容易延续到通道扩展与 Final
加深；高阶 TP 及 Persistent 跨层状态是接下来的主要优化方向。表中的近 1× 点
只说明本轮计时接近，不能据一次配对主张稳定的微小优势。

10,000 原子时，L3/C128/2 层 Final/Persistent 为 200.760/235.582 ms，
DPA C32/L3/2 block 为 212.106 ms，即 1.06/0.90×；reserved 为
18.38/19.06 GiB，对应 DPA 18.40 GiB。C192/L3/3 层完成 HP 普通预热后在
Graph 捕获核验流程 OOM，DPA C48 则在未加速参照 OOM；C256/L3/4 层双方均在
优化执行阶段 OOM。这里区分了捕获、参照与执行显存边界，没有把所有 OOM 混为一种。
这些扩展点也没有同时达到两倍与更低显存；此前训练 checkpoint 的两倍结果仍按
对应模型和规模列在后文。

完整记录：[架构对比与显存表](results/architecture_a100/report_complete/ARCHITECTURE_SCALING.md)、
[逐点数据及验证身份](results/architecture_a100/report_complete/architecture_summary.json)、
[原始主矩阵](results/architecture_a100/results_v2/)、
[10,000 原子压力矩阵](results/architecture_a100/results_large_v2/)、
[最大 Persistent 独立原式核对](results/architecture_a100/qualifications_chunk64_allocator_v4/l3c256d4/n1024/persistent.json)。
所有数值计算和汇总均由 `sbatch/GPU-A100/XHPC-A100` 完成，最高同时三张 A100。
本轮未使用 4090，未修改正式 MD 或既有 checkpoint。

## Final 10,000 / 50 达到两倍与更低显存（native_v64–70）

Final 原 checkpoint（792,198 参数）在 A100 上完成完整 E/F/virial 验收，
优化已进入 `generated` 后端的默认大型冻结推理路径。
同一作业内在同一物理 A100 上顺序测量 DPA4-mini 与 Final；DPA 实际创建 compile
cache，并执行两处 `_TritonSO2ValuePath` level 2。所有神经网络及力均为 FP32、
TF32 off，virial 归约为 FP64。10,000 原子均恰有 50 个截断邻居，500,000 条有向边。

| 独立作业 | Final ms | DPA full ms | DPA lower ms | 相对 full | 相对 lower |
| --- | ---: | ---: | ---: | ---: | ---: |
| 92113 | 61.089 | 126.160 | 123.867 | 2.065× | 2.028× |
| 92114 | 61.243 | 126.382 | 123.690 | 2.064× | 2.020× |
| 92115 | 61.558 | 127.112 | 124.442 | 2.065× | 2.022× |
| 92117，默认自动选择 | 61.984 | 127.637 | 124.744 | 2.059× | 2.013× |
| 92133，最终源码默认自动选择 | 61.725 | 127.932 | 125.391 | 2.073× | 2.031× |

`native_v66` 默认路径吞吐为 161,332 atom/s。包含 CUDA Graph 池的峰值 reserved 为
8.701 GiB，DPA lower 为 11.697 GiB，减少 25.61%；capture peak allocated 为
8.140 GiB，对比 DPA 的 10.300 GiB。三次独立候选配置复测的 reserved 均为
8.660 GiB；默认配置保留额外执行句柄后的实际占用如上，未选取较小候选值代替默认值。
最终 `native_v70` 默认路径的 reserved 为 8.660 GiB，capture allocated 为
8.118 GiB，对比 DPA lower 的 reserved 11.697 / allocated 10.300 GiB；reserved 减少
25.96%。五轮保守 lower 边界加速为 2.01–2.03×。

实际采用：目标节点 cotangent 复用、FP32 源梯度归约、coalesced 方向/相位 partial、
稳定 live CSR、输入所有权 Hermitian VJP、相位 MLP 与两个 head 整体融合、
周期边向量/力 scatter 融合，以及首个中性消息的四 warp/full-edge 调度。
深层 coherent 前向保留一 warp；所有原 TP、rank16 与相位路径保持。
默认结果的 dispatch 已记录中性大型核 1 个、相位整网 1 个及目标端 VJP 原生扩展，
没有导入外部 TP 库。

五个作业的能量、原子能量和全部 30,000 个力分量都通过原有
`atol=1e-4, rtol=1e-3`，virial 通过既有张量范数限；没有更换验收参照或放宽阈值。
默认力相对 L2 差为 7.14e-6，最大绝对差 2.71e-4，逐分量误差/允许界最大为
0.8313；virial 相对 L2 差为 3.37e-7。改变坐标后的 CUDA Graph 与实时执行再次核对通过。
原模型 N10000 单体显存不足，参照采用完整依赖 halo，先与 512 原子单体核对；
参照生成及编译均不进入计时。A100 152 项回归及 16 帧 T1x 留出 E/F 检查通过。

记录：[五轮同卡及训练回归汇总](results/throughput_a100/summary_final_release_v70.json)、
[最终源码默认同卡复测](results/throughput_a100/final_release_v70/)、
[四轮阶段汇总](results/throughput_a100/summary_final_default_v66.json)、
[默认执行、全部输出核对与 profile](results/throughput_a100/final_default_v66/)、
[三轮独立复测 a](results/throughput_a100/final_repeat_v64_a/)、
[b](results/throughput_a100/final_repeat_v64_b/)、
[c](results/throughput_a100/final_repeat_v64_c/)。
此前未通过的合并前向/重排候选及独立参照审计继续保留在
[v62 失败记录](results/throughput_a100/final_network_v62/)和
[完整坐标参照审计](results/throughput_a100/force_reference_full_v64/)，未用这些候选替代默认。

## Persistent 继续优化（native_v67 起）

接入上述默认大型图执行后，Persistent 的同协议完整 E/F/virial 已从上一轮
85.583 ms 降至约 71–72 ms；原 1,029,967 参数与 checkpoint 保持。
`native_v67` 为 71.006 ms，相对同卡 DPA full/lower 为 1.780 / 1.749×；
最终源码 `native_v70` 为 72.089 ms，相对同卡 DPA full/lower 128.418 / 126.375 ms
为 1.781 / 1.753×。后者 reserved 为 8.988 GiB，对比 DPA 11.697 GiB 少 23.16%，
capture allocated 为 8.376 GiB，对比 DPA 10.300 GiB。
所有输出及改变坐标后的 Graph 核对通过。正式记录见
[大型默认复测](results/throughput_a100/persistent_default_v67/)，
[逐模块与目标依赖热点](results/throughput_a100/persistent_hotspots_v67/)、
[最终同卡汇总](results/throughput_a100/summary_persistent_release_v70.json)。
继续测试的候选均完成原模型完整输出与坐标重放核对，当前保留原默认执行：

| 非默认候选 | ms |
| --- | ---: |
| 冻结首层目标端伴随，不分配 DX | 70.852 |
| 首层伴随＋径向 trunk 整体融合，tile16 / tile128 | 72.520 / 70.948 |
| 首层伴随＋coherent 前向 2 / 4 warp | 71.375 / 71.993 |
| 首层伴随＋目标节点邻居 2 / 4 路分块 | 71.729 / 72.368 |
| 首层伴随＋每组 1 / 2 条 TP 路径 | 80.824 / 76.254 |

这些独立候选没有显示可采用的整网收益；逐路径分组还增加了 partial 缓冲。
没有将微核计时优势或小幅运行波动写成默认提速。下一热点仍是深层完整 TP/VJP
及 64→1408 径向输出 GEMM 的跨算子数据流。Persistent 的两倍目标仍未完成。
候选记录：[首层伴随](results/throughput_a100/persistent_nodx_v68/)、
[前向与邻居分块](results/throughput_a100/persistent_forward_v69/)、
[径向融合](results/throughput_a100/persistent_trunk_v69/)、
[tile128](results/throughput_a100/persistent_trunk128_v70/)、
[TP 路径分组](results/throughput_a100/persistent_groups_v70/)。

扩展回归在 A100 上为 172 passed，包括首层缺失 X 目标、尾通道、多 head、
分组目标端伴随、径向融合的一阶/混合导数与实时参数更新；T1x 前 16 帧 E/F 检查通过。
Final 默认实现的 152 项原回归在 4090 上也全部通过（冷 JIT 时间不用于性能比较）。

最终源码训练回归保持全部 1,029,967 个参数可训练，两层 TP 系数均有训练依赖。
128 / 512 原子的历史 CUDA Graph 完整 E/F 损失及 AdamW step 为 12.837 / 24.767 ms，
同卡 DPA 训练 compile 为 18.777 / 39.186 ms；这组重放结果不用于训练加速结论。
无 Graph 的 HP eager 为 45.724 / 44.857 ms，两个尺寸均未超过该 DPA 对照。
逐参数梯度相对 L2 最大低于 8.2e-6；512 原子仍有原报告中的少量近零梯度逐元素
超限（eager 4 项 / capture 3 项），张量范数及首步更新核对通过，明细原样保留。
冻结推理新特化没有用于训练。记录见
[训练原始记录](results/throughput_a100/training_paired_v70/)。
最终作业 92133 / 92134 / 92135 与汇总 92136 均 COMPLETED 0:0；本轮结束时无运行或排队 A100 作业。

## 计算图裁切与 CSR 归约重排（native_v41–48）

本轮将 10,000 原子 / 50 邻居的完整 E/F/virial 从 93.241 ms 降至 85.583 ms，
耗时减少 8.21%。同卡重新实测 DPA4-mini compile＋Triton level 2 为 125.862 ms，
当前加速 1.471×，尚未达到 2×。该目标对应不超过 62.931 ms；仍需再减少
26.47% 的当前耗时，不能把本轮进展写成已达成目标。

实际接入的改动：

1. 将 CHORUS 的逐阶独立 slice 接到 ICTC 母版已有的单次多输出 split，
   反向一次拼接各阶 cotangent，删除各 slice 的整宽补零与缓冲累加。
   中性/双流拆分改为多输出 split，实/虚双流改为 unbind，保留原视图和数值。
2. 五路输入均需要梯度的大型 C128/L2 消息，使用四路 CSR 邻居所有权及确定性
   DX 合并；静态化有效梯度标志。结构自动选择限定于 sm80 与足够边数，
   其它结构仍走原联合伴随。高阶训练伴随不切换到该一阶特化。
3. TP 与相位上下文伴随读取当前 autograd 目标的实际需求；首层冻结节点上下文
   不再分配或原子累加没人消费的投影梯度。原参数及训练高阶依赖均保留。

原模型核心没有可直接丢弃的大支路。实际 autograd 目标核查中，深层消息的 X、
方向谐波、gate、实相位与虚相位五路梯度全部被消费；末层 product 原本已只输出
标量。没有删除有效 SO(3) 阶、Hermitian 路径、Persistent 递归或力/virial 输出。

| 原子数 / 邻居 | DPA lower ms | 原 Persistent ms | 新 Persistent ms | 新 / DPA 加速 |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 50 | 52.666 | 39.550 | 36.169 | 1.456× |
| 10000 / 50 | 125.862 | 93.241 | 85.583 | 1.471× |

| 原子数 / 邻居 | DPA 保留 GiB | 新默认保留 GiB | 低显存 JVP 保留 GiB | JVP ms |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 50 | 4.834 | 5.211 | 4.320 | 39.174 |
| 10000 / 50 | 11.697 | 9.072 | 6.369 | 93.243 |

保留显存包含 CUDA Graph 池。10,000 原子默认仍比 DPA 少 22.44%，但比旧默认
8.900 GiB 略高；4096 原子默认仍多用 7.80%，并未同时满足全部规模的低显存目标。
旧/新版本耗时分别来自各自完整配对；同版本候选筛选另在同一进程中测原方案→候选
→原方案，原始样本全部保留。每次正式配对均在同一物理 A100 上依次运行 DPA
与 CHORUS；共同输入、FP32/TF32 off、预建图、15 次同步中位数和 FP64 virial
边界与上一节相同，未用 A100/4090 混合时间推导性能。

10,000 原子新默认的深层联合伴随由 22.146 ms 降至约 17.917 ms。
主要三类 SGEMM 仍共约 23.81 ms，TP 前向约 7.07 ms，冻结首层伴随约 4.10 ms。
即便理想化地省去整个剩余深层伴随，也不足以单独达到 2×；继续优化需同时覆盖
径向输出投影及其 VJP、消息前向与跨算子数据流，不能只删 Python 节点。

筛选保留了未采用的结果：分组边伴随＋独立源 DX 为 104.259 ms；显式逐边 DX
为 91.985 ms 但额外占约 2.3 GB；相位径向仿射/gather/SiLU 融合约 93.092 ms，
对照约 93.1–93.5 ms，未见稳定收益。以上均未替换默认路径。

训练同时复测完整 E/F 损失、混合反传和 AdamW，全部 1,029,967 个原参数可训练：

| 原子数 / 邻居 | DPA 训练 compile ms | HP eager ms（旧→新） | 历史 HP Graph ms（旧→新） | 历史 Graph / DPA，不验收 |
| --- | ---: | ---: | ---: | ---: |
| 128 / 50 | 18.525 | 46.267→43.973 | 13.331→12.882 | 1.438× |
| 512 / 50 | 39.552 | 46.436→43.712 | 25.455→24.891 | 1.589× |

HP eager 吞吐提高约 5.2% / 6.2%。表中的固定图重放仅保留为历史诊断，不用于训练
加速结论；本段也未测可变形状数据加载的端到端速度。DPA 训练专用 compile
与其仅推理生效的 level-2 path 区分不变。新逐参数梯度相对 L2 最大约 9.21e-6，
首步更新参数相对 L2 最大约 1.42e-6；512 原子少量近零梯度的逐元素失败继续在
JSON 中列明，不改成冻结参数或宣称位级相同。

113 项回归在 A100 与 4090 各全部通过，是同一组 113 项。最终 `native_v50`
归档复测为 18.64 s / 14.71 s（JIT 缓存已热）；不把测试执行时间用作吞吐结论。
新增覆盖多输出 split 的 strided/prefix 视图、部分目标及精确一/二阶导数，CSR4
的尾通道/分组 head/掩码，以及相位目标梯度裁切。原 T1x 前 16 帧 E/F 复核通过。
10,000 原子的原公式完整依赖域验证再次通过，力相对 L2 差 4.86e-6、最大绝对差
1.23e-4；virial 相对 L2 差 2.05e-7。改变坐标后的 Graph 重放核对也通过。

正式作业 `92048`、`92049_0–1`、`92051`、`92052`、`92053`、`92055`
均 COMPLETED 0:0。4090 最终归档的脚本 PID 42455 已结束，日志为
113 passed / DONE；A100 当前无排队或运行作业。算术实现来自 `native_v48`；
最终 `native_v50` 补充 sm80 自动选择的硬件边界与说明，并再次完成全部回归及
A100 T1x 复核，未改变正式计时的 A100 路径。

记录：[本轮汇总](results/throughput_a100/summary_hotspots_v48.json)、
[完整 E/F/virial](results/throughput_a100/large50_v48/)、
[训练](results/throughput_a100/training_paired_v48/)、
[裁切前的目标依赖/shape profile](results/throughput_a100/hotspots_v42/diagnosis.json)、
[裁切后的 profile](results/throughput_a100/hotspots_v48/diagnosis.json)。

## 1 万原子 / 50 邻居与训练联合伴随（native_v32–38）

10,000 原子用例已完成，实际有向边数为 500,000，每个原子的截断邻居数均为 50。
共同输入采用带小扰动的正交周期晶格，轴比 1:1.2:1.4、截断半径 5 Å，
seed=20260728；不是把 56 邻居的立方晶格截短为 50。模型仍为原始 Persistent
C128/L2、correlation=3、rank16、1,029,967 参数。以下每行在同一物理 A100 上
顺序测 DPA4-mini 和两个 Persistent 路径，均为完整 E/F/virial、预建图、15 次
同步 walltime 中位数。FP32 网络与力、TF32 off；virial 归约为 FP64。
DPA 实际使用 compile core 和两个 level-2 `_TritonSO2ValuePath`。

| 原子数 / 邻居 | DPA lower ms | Persistent 默认 ms | Persistent 径向 JVP ms | 默认 / DPA 加速 | JVP / DPA 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 / 50 | 52.673 | 39.550 | 41.335 | 1.332× | 1.274× |
| 10000 / 50 | 125.393 | 93.241 | 97.296 | 1.345× | 1.289× |

| 原子数 / 邻居 | DPA 保留 GiB | 默认保留 GiB | 径向 JVP 保留 GiB |
| --- | ---: | ---: | ---: |
| 4096 / 50 | 4.834 | 5.141 | 4.320 |
| 10000 / 50 | 11.697 | 8.900 | 6.369 |

显存包含 CUDA Graph 私有池，取 capture peak、steady reserved 和计时 reserved
最大值。10,000 原子默认路径比 DPA 少 23.9%，径向 JVP 少 45.6%；4096 原子默认
路径仍多 6.3%。径向 JVP 继续作为低显存选项，没有替代默认最快路径。

默认吞吐从 103,564 增至 107,249 atom/s，仅增加 3.56%；JVP 路径增加 3.72%。
扩大体系已接近当前实现的平台区，不等同于达到 A100 的绝对硬件极限。
10,000 原子默认路径的 profiler 中，深层联合 TP 伴随为 22.15 ms，TP 前向为
7.06 ms，首层边伴随为 4.03 ms；主要三类 SGEMM 合计约 23.97 ms。
当前仍不能称为“远超 compile＋Triton level 2 DPA4-mini”。

### 大体系数值核对

原始逐边展开路径在 40 GB A100 上无法单体执行 10,000 原子参考计算。
验证器按能量中心划分完整二跳依赖域，保留原边顺序、原坐标和周期位移，
计算每组中心能量对全部 halo 坐标及应变的导数，再装配完整 10,000 原子的
能量、力和 virial。没有裁剪 halo 力，也没有更改模型或截断半径；参考分块不计时。
该装配先与 512 原子的单体原公式核对，并另通过 1/2/3 交互层的非规则图检查。

10,000 原子默认/JVP 路径的力相对 L2 差为 4.83e-6 / 5.42e-6，最大绝对差为
1.30e-4 / 1.35e-4；原子能量相对 L2 差约 5.3e-7。virial 相对 L2 差约 2.0e-7，
原始总量最大绝对差 0.045 / 0.049 eV，每原子为 4.5e-6 / 4.9e-6 eV。
CUDA Graph 在改变坐标后与实时 eager 再次核对通过，未计时缓存不变输出。

virial 使用旋转不变的 Frobenius 范数判据
`||delta V||_F <= 1e-5 ||V||_F + N*1e-7 eV`，同时报告原始分量及每原子最大误差。
最初逐分量相对判据在接近零的剪切分量上失败；原公式重复计算本身也有约
1e-3 eV 的总量变化。旧失败日志保留，未改动网络精度或 virial 归约精度。

### 完整能量 / 力损失训练

新增原生三流联合 mixed-adjoint，同时计算 X、方向谐波、径向 gate、实相位和
虚相位的混合导数及输出 cotangent 导数；三阶及以上保留精确可微原公式。
原始 checkpoint 的 TP 系数在训练时可学习，现通过实时 gate 乘积保留其全部梯度，
不冻结原参数。部署加载器执行冻结前的 trainability 被完整记录并恢复；
两层 TP 系数均参与训练，1,029,967 个原参数全部保留。

以下为同卡顺序配对的完整训练 step：能量、保守力、能量/力 MSE、二阶反传和
AdamW 更新。共同压力目标为 `(energy/N)^2 + 100*mean(force^2)`，固定零标签，
AdamW lr=1e-5、weight_decay=0；它用于执行吞吐，不替代论文训练协议。
原权重和空优化器状态分别恢复，正反顺序各测一轮，预热后每轮 8 次。
不计数据加载、邻居构建、JIT 或捕获准备。两种原参考模式均单独实测。

| 原子数 / 邻居 | DPA 训练 compile ms | 原参考 eager ms | HP eager ms | 历史原参考 Graph ms | 历史 HP Graph ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 50 | 21.783 | 66.007 | 46.267 | 53.850 | 13.331 |
| 512 / 50 | 39.464 | 200.445 | 46.436 | 191.317 | 25.455 |

原生 eager 相对 CHORUS 原参考提高 1.43× / 4.32×，但在这两个规模仍慢于 DPA。
表中 Graph 计时仅保留为历史诊断，不用于训练吞吐或跨模型加速结论。普通训练的
变批次及变 N/E 测试改用本文顶部的新协议；原生 mixed-adjoint 已由显式 HP 训练路径使用。

512 原子保留显存：原参考 eager 10.011、HP eager 1.805、原参考 Graph 10.691、
HP Graph 2.492、DPA 2.590 GiB。HP eager 相对原参考下降 82.0%，HP Graph 下降
76.7%。128 原子 HP Graph 为 0.713 GiB，仍高于 DPA 的 0.629 GiB。
DPA 的 `use_compile=True` 确实创建训练编译缓存；其 level-2 专用 value path 被
上游 `not self.training` 条件限定为推理，训练测其原生可微 compile 路径。

全部参数梯度逐张量相对 L2 最大约 1.1e-5，资格阈值 1e-4；近零张量另检查
1e-7 绝对误差。原参考与 HP 第一次 AdamW 更新后的参数相对 L2 最大约 1.42e-6。
零附近梯度会使首步 AdamW 对符号敏感，少量逐元素检查失败及约 2e-5 的最大单权重
差保留在 JSON；不宣称位级相同。Graph 与同状态 eager 的更新参数最大差不超过
5.96e-8，并核对了重放中实时更新的优化器状态。

97 项原生回归在 A100（59.69 s）及 4090（256.49 s）各全部通过；同一组测试
不重复计数。另有 A100 的 4 项依赖域/范数旋转检查通过。T1x 前 16 帧
139 原子 / 1058 边的原 checkpoint E/F 复核通过。未修改既有正式 MD 或部署工件。

原始记录：[汇总](results/throughput_a100/summary_large_training.json)、
[1 万原子](results/throughput_a100/large50_v38/)、
[4096 原子](results/throughput_a100/large50_v35/)、
[同卡训练](results/throughput_a100/training_paired_v38/)、
[held-out 复核](results/throughput_a100/fusion_validation_v37/persistent_heldout.json)。
最终作业 `92024_0`、`92032_1`、`92033`、`92030`、`92036` 和汇总 `92037`
均 COMPLETED 0:0；4090 `native_v32` 日志以 97 passed / DONE 结束。
过程中的未通过数值检查、OOM 和脚本准备错误保留在
[诊断日志](logs/throughput_a100/)，未用这些失败用例报告加速。

## 径向 gate / TP / VJP 融合：本轮结果（native_v29）

本轮实现了可选的一维径向 VJP：沿边长传播原径向 MLP 的值与导数，
在原生 ICTC TP 伴随中直接归约为每条边的距离梯度，避免完整 gate 梯度矩阵。
径向导数投影按边块复用显存；网络、checkpoint、模型容量、密度秩和精度未变。
原公式保留精确高阶可微回退。该方案取得显存收益，没有进一步提高吞吐，
因此 `radial_jvp=False` 仍为默认，原最快执行路径保持不变。

以下为完整 E/F/virial 正式同卡配对。每行在同一物理 A100 上顺序运行 DPA full/lower、
新低显存 Final/Persistent 及原最快 Persistent。DPA 实际开启 compile 和 Triton level 2；
双方 TF32 off，神经网络/力 FP32、virial FP64。预建图、15 次同步 walltime 中位数，
不包含 CHORUS 邻居构建，不混用不同卡型时间。

| 原子数 | DPA lower ms | 原最快 Persistent ms | 径向 VJP Persistent ms | 相对 DPA | 相对原最快耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 6.576 | 3.608 | 3.932 | 1.67× | 增加 9.0% |
| 1024 | 13.249 | 8.134 | 8.407 | 1.58× | 增加 3.4% |
| 4096 | 43.960 | 29.321 | 30.276 | 1.45× | 增加 3.3% |

| 原子数 | DPA 保留 GiB | 原最快 Persistent GiB | 径向 VJP Persistent GiB | 相对原最快显存下降 | 相对 DPA 显存下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.154 | 0.186 | 0.180 | 3.2% | 增加 16.5% |
| 1024 | 1.123 | 0.936 | 0.922 | 1.5% | 17.9% |
| 4096 | 4.320 | 2.826 | 2.348 | 16.9% | 45.7% |

显存为 capture peak、steady reserved 和计时 reserved 的最大值，包含图私有池。
4096 原子进程 resident 为 2944 对 DPA 的 4936 MiB。128 原子仍比 DPA 多用显存。
力相对 L2 差为 1.43e-6–1.83e-6，最大绝对差不超过 4.85e-6；virial 相对 L2
差为 1.54e-7–2.19e-7。Persistent 仍为原 1,029,967 参数，无外部 TP 库导入。
Final 对照为 3.138/6.821/24.888 ms，DPA full 为 8.236/15.328/46.016 ms。
“Persistent 远超 compile＋Triton level 2 DPA4-mini”的速度目标尚未实现。

当前 91 项回归在 A100（137.66 s）和 4090（378.04 s）各全部通过，是同一组
91 项的双硬件复测。新增覆盖径向 Bessel/Gaussian 及 cutoff/归一化的方向导数、
值/一阶/混合高阶导数、分组 head、尾通道、空图、实时固定权重更新、冻结首层、
训练权重回退，以及 CUDA Graph 改变坐标/元素/拓扑/cell 后的结果和状态失效。
原 T1x 验证集前 16 帧（139 原子/1058 边）同 checkpoint E/F 再次 PASS；
eager 原参考/径向 VJP 路径为 20.608/17.752 ms（1.161×），单独报告，不替代上表。

保留两类显式消融接口：`radial_fusion=True` 分块消费宽隐藏层 gate VJP；
`radial_gate_pipeline=True` 同时流水生成 gate。它们可进一步降低显存，但吞吐未优于
默认路径。寄存器内 IEEE FP32 GEMM、逐通道 gate 重算、常驻块和独立权重布局候选
均没有进入默认执行。完整逐通道重算曾达到约 242 ms，不能因显存更少而声称优化成功。
当前时间仍集中在联合 TP 伴随与严格 FP32 的宽径向投影，省去中间量本身不足以
抵消额外投影和调度成本。

正式原始记录：[summary_paired_radial_jvp_v29.json](results/throughput_a100/summary_paired_radial_jvp_v29.json)、
[逐次配对](results/throughput_a100/paired_radial_jvp_v29/)、
[T1x 对照](results/throughput_a100/fusion_validation_v29/persistent_heldout.json)。
候选与分块消融：[summary_radial_candidates.json](results/throughput_a100/summary_radial_candidates.json)。
A100 回归 `92007`、配对 `92008_0–2`、汇总 `92011/92014` 均 COMPLETED 0:0；
4090 `native_v29` 完整回归日志以 91 passed/DONE 结束。
一次生成代码参数拼接编译失败已修正并完整重跑；一次 CPU 汇总源目录错误已修正，
不影响任何数值任务，失败日志均保留。所有计算遵守远端任务目录和 sbatch 规则。

## 当前默认最快配对（native_v18）

Persistent 相对上一版的完整 E/F/virial 时间缩短 10%–21%；相对实际开启
`DP_COMPILE_INFER=1, DP_TRITON_INFER=2` 的 DPA4-mini 为 1.50×–1.80×。
中、大体系保留显存更少；128 原子仍更高。因此“远超且所有规模显存更少”的
双目标尚未同时达到。

每行在同一物理 A100 上顺序测 DPA full/lower、当前 Final/Persistent 和上一版
Persistent。固定原 checkpoint、32 邻居周期结构、seed=20260728、严格 FP32 网络与力、
TF32 off，15 次同步 walltime 中位数；双方 virial 为 FP64 归约。CHORUS 用 CUDA Graph，
DPA 用其实际 compile core 与两个 `_TritonSO2ValuePath`，并校验开关与执行缓存。
这是预建图边界的模型吞吐，不是完整 MD step 吞吐。

| 原子数 | DPA lower ms | 上一版 Persistent ms | 当前 Persistent ms | 相对 DPA | 本轮时间下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 6.466 | 4.563 | 3.598 | 1.80× | 21.1% |
| 1024 | 13.458 | 9.171 | 8.064 | 1.67× | 12.1% |
| 4096 | 44.313 | 32.824 | 29.571 | 1.50× | 9.9% |

| 原子数 | DPA 保留 GiB | 当前 Persistent 含图池保留 GiB | Persistent 相对变化 |
| --- | ---: | ---: | ---: |
| 128 | 0.154 | 0.186 | 增加 20.3% |
| 1024 | 1.123 | 0.936 | 减少 16.7% |
| 4096 | 4.320 | 2.826 | 减少 34.6% |

4096 原子进程 resident 为 3430 对 4936 MiB；128 原子为 726 对 670 MiB。
含图池保留统计使用 capture peak、steady reserved 和计时峰值的最大值，
不使用遗漏图私有池的 hot allocated 作为省显存结论。
力相对 L2 差为 1.31e-6–1.51e-6，最大绝对差不超过 4.25e-6；virial 相对 L2
差为 1.89e-7–2.66e-7。Persistent 仍为原 1,029,967 参数。
Final 同卡对照为 2.956/6.697/24.374 ms，原始 JSON 一并保留。

原始汇总：[summary_paired_fusion_v18.json](results/throughput_a100/summary_paired_fusion_v18.json)；
逐次记录：[paired_fusion_v18](results/throughput_a100/paired_fusion_v18/)。
最终配对 `91970_0–2`、CPU 汇总 `91971` 均 COMPLETED 0:0。

本轮保留的布局/线性投影/元素分组/反向调度改动详见“Persistent 整网融合增量”。
当前主要时间仍在深层联合 TP 伴随、径向门控 MLP 及其回传；
下一阶段需要跨 gate 生成、TP 和 VJP 的融合，继续优化孤立的小算子不足以带来数倍级整网领先。

## 上一轮正式配对（native_v10）

正式基线限定为 `DP_COMPILE_INFER=1, DP_TRITON_INFER=2, DP_TF32_INFER=0`。
同一物理 A100 上顺序执行，完整 E/F/virial，预建图边界，15 次同步 walltime 中位数。
网络和力为 FP32、TF32 off；双方 virial 总量使用 FP64 归约。CHORUS 使用固定形状
CUDA Graph，DPA 使用其实际编译核心与 `_TritonSO2ValuePath`。这不是相同编译器的
算子微测量，也不是 MD step 吞吐。

| 原子数 | DPA4-mini lower ms | Final ms | Persistent ms | Final 加速 | Persistent 加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 6.442 | 3.343 | 4.563 | 1.93× | 1.41× |
| 1024 | 13.244 | 7.013 | 9.182 | 1.89× | 1.44× |
| 4096 | 44.933 | 25.866 | 33.221 | 1.74× | 1.35× |

DPA 的 full 接口（含邻居构建）分别为 8.174/14.871/46.733 ms；主表使用更严格的
lower 基线，不把邻居构建收益计入 CHORUS 的模型加速。DPA lower 连边向量也已预建，
CHORUS 仍从坐标计算边向量。两者使用各自兼容运行时：DPA PyTorch 2.11.0+cu126，
CHORUS PyTorch 2.7.1+cu128，不用跨 GPU 类型的 walltime 计算比值。

| 原子数 | DPA lower 保留 GiB | Final 含图池保留 GiB | Persistent 含图池保留 GiB |
| --- | ---: | ---: | ---: |
| 128 | 0.154 | 0.178 | 0.189 |
| 1024 | 1.123 | 0.791 | 0.984 |
| 4096 | 4.320 | 2.973 | 2.928 |

4096 原子 Persistent 的保留显存下降 32.2%；进程 resident 为 3538 对 4936 MiB。
128 原子时保留显存和进程 resident 均略高，不能概括为所有规模显存更少。
allocator allocated、reserved、图捕获峰值及 resident 原始统计均保留。

三个规模下力相对 L2 差约 1.2e-6–1.5e-6，virial 相对 L2 差约 1.5e-7–3.0e-7。
未换模型、checkpoint、种子、通道数、密度秩或层数。Final/Persistent 参数分别为
792,198/1,029,967，DPA4-mini 为 650,428；这里比较现有模型，而非等参数架构。

原始汇总：[summary_level2_efv_v10.json](results/throughput_a100/summary_level2_efv_v10.json)；
逐次计时/执行计划：[正式配对 JSON](results/throughput_a100/paired_level2_efv_v10/)。
正式作业 `91957_0–2` 及汇总 `91960` 均 COMPLETED 0:0。

## Persistent 整网融合增量

本轮保留四项改动：三流 TP 的角分量优先物理布局与通道 GEMM 直接连接；
末层 charged update 与 Hermitian rank projection 的冻结权重代数融合；
按实时元素索引分组的 IEEE FP32 GEMM；按是否需要 charged-input VJP 选择
源节点 CSR 联合伴随或逐边块归约。训练和可训练权重保留精确可微回退。
没有更改 checkpoint、模型容量、密度秩、相位定义或 TF32 策略。

完整回归扩展至 48 项。A100（548.12 s）与 4090（337.49 s）分别全部通过，
是同一组 48 项的双硬件复测，不重复计数。包括联合消息的值、一阶及
混合高阶导数、冻结首层几何伴随、非整块通道、空图、实时元素/权重更新、
末层投影的 Hermitian/charge2 对照，以及 1/2/3 层的安装边界。
单层对照使用合法的 Final 模型；Persistent 原架构要求至少两层。
早期新增测试曾错误构造单层 Persistent，修正测试夹具后重跑，未修改模型约束。

原 T1x 验证集前 16 帧（139 原子/1058 边），Persistent 同 checkpoint 的完整 E/F
再次 PASS；交替配对的 eager 参考/HP 中位时间为 20.336/16.864 ms（1.206×）。
原始记录：[persistent_heldout.json](results/throughput_a100/fusion_validation_v16/persistent_heldout.json)。
这与合成周期结构的 CUDA Graph 吞吐分别报告。

两项候选未纳入当前源码：全部改为逐边伴随后，4096 原子的整网 E/F/virial
从约 30.93 增至 33.03 ms；C128 块间协作归约也未胜过较快的联合 CSR 调度。
同样测试了双方一致的 `expandable_segments:True` allocator 配置，128 原子
Persistent 的含图池保留显存由约 190 MiB 增至 238 MiB，未采用该配置。
原始候选/消融数据与失败日志保留，不从较慢候选中选择局部 kernel 指标声称整网收益。

## 已实现

- 母版新增 `CoherentGraphMessage`：原始 gate 单份存储、三流联合图 TP、联合伴随、
  degree 独立多输出，避免大缓冲补零和 slice backward。
- CHORUS 相位首层按节点/径向因子化，并融合 gather/SiLU；六种系数参数化有专核。
- Hermitian/charge2 密度与一阶/混合二阶导数保留原 CG 常数。
- 原 reduced-CG checkpoint 的 U 直接加载，母版对实际 U 做原生电路因子化，
  不用外部库重建基底。元素权重库使用共享 GEMM。
- Persistent charged update 改为共享 GEMM，去掉数千批小 BMM。
- 固定 N/E/B 的 CUDA Graph 接口支持变化后的坐标、元素、cell、拓扑；提供短程
  E/F/virial 入口。邻居构建、E0 与已有正式 MD/AOTI 部署包不在该接口内。

实现和调用见 [接入指南](../../docs/ICTC_O3_HP.md)。

## DPA4 基线审计：早期比值作废

本机指定 DPA4/SeZM 版本在 eval 下由 `DP_COMPILE_INFER` 控制编译，
`model.use_compile=True` 仅控制训练路径。最初脚本只设后者，因而其约 95–97 ms
的 4096 原子结果是未充分开启加速的诊断，不能用于声称超越编译版 DPA4-mini。
原始日志/结果保留，后续对比必须按下列策略重跑。

- `DP_COMPILE_INFER=1`，且断言 `model.should_use_compile()` 为真、实际 compile core
  缓存非空；不以配置布尔值代替执行证据。
- `DP_TRITON_INFER=2`：严格 FP32 融合 Triton；同时测 level 1。
- 源码虽存在 level 3，但其 FP16x3 补偿 GEMM 不属于严格 FP32；按用户要求已停止
  该档试验，不纳入正式对比。比较同 checkpoint 关闭 compile/Triton 的能量、力与 virial。
- 神经网络参数 FP32，TF32 关闭。DPA 上游 energy/virial 使用 FP64 累积，其 dtype
  会记录在 JSON 中；不能笼统写成所有输出均 FP32。
- profiler、实际绑定的 Triton value-path 类和编译缓存槽数进入结果。
- 同时报含邻居构建的 full 和预建 compact-edge 的 lower 边界；不能把两种边界混算。

基线入口：[profile_dpa_boundary.py](profile_dpa_boundary.py)；
level 1/2 独立单卡审计：[run_dpa_fast_audit.sbatch](run_dpa_fast_audit.sbatch)。

## CHORUS 的独立执行测量（不是相对优化版 DPA4 的结论）

原 Persistent checkpoint、周期抖动简单立方结构、cutoff=5 Å、每原子恰好 32 条边、
seed=20260728，完整能量与坐标反向力。A100-SXM4-40GB，严格 FP32、TF32 off。
该结构用于执行与规模测试，不是平衡 MD 精度验证。

| 4096 原子阶段 | E/F ms | 图池及 allocator 保留 GiB |
| --- | ---: | ---: |
| 三倍 gate 的图核融合 | 58.575 | 7.14 |
| 共享 charged GEMM + 单 gate 联合三流 | 37.766 | 3.41 |
| degree 多输出，消除 slice backward 大缓冲 | 32.846 | 2.97 |

第三阶段 128/1024 原子分别为 4.631/9.201 ms。力相对 L2 差约 1.5e-6，
4096 原子力最大绝对差 4.87e-6；实际 checkpoint 参数 1,029,967，未删减容量。
这是当前阶段测量，尚不等同于“Persistent 远超 compile+Triton DPA4-mini”。
共同 E/F/virial 输出的最终配对结果须另列，不把 E/F 数据混入该表。
正式 E/F/virial 对比已列在本报告首节。

图重放时 `peak_allocated` 不含私有图池，不能单独作为低显存证据。
报告 capture peak、包括图池的 reserved 和 nvidia-smi 进程 resident；编译冷启动
不进入稳态计时。不同硬件时间不混合计算加速比。

## 正确性与过程记录

4090 上 native_v6 的 35 项测试全部通过（329.89 s）：原核心/扩展测试、原 U 重用、六种相位
系数、联合消息值/一阶/混合二阶导数、分组 head、尾通道、空边及实时权重更新。
另覆盖多输出、Final/Persistent 的 CUDA Graph 变化坐标/元素/拓扑/cell、virial
应变导数和状态失效检查。随后 native_v9 对新增 Python 相位干预失效检查复测 4 项
（5.56 s）全部通过。
FP64 virial 修正后复测 4 项（5.58 s）通过，随后含多结构/零边的 6 项回归
（5.98 s）通过。后两项为新增用例，其余为既有用例的复测，不重复计算独立用例数。

两个原 checkpoint 在原 T1x 验证集前 16 帧（139 原子/1058 边）的 E/F 对照均 PASS：
Final 原参考/HP eager 为 15.910/14.500 ms；Persistent 为 20.803/18.750 ms。
这些小图 eager 的增益约 1.10×，不把其结果替换成 CUDA Graph 合成结构的收益。
原始文件：[Final](results/throughput_a100/native_heldout_v8_final.json)、
[Persistent](results/throughput_a100/native_heldout_v8_persistent.json)。

新增功能和试验中的失败均保留日志：高阶参考 gate 维度错误已修复；不同 ABI
版本共用扩展缓存导致调用签名冲突，改为带 ABI 版本的扩展名；DPA compact-edge
入口按当前 SeZM 实际接口修正；未更换 checkpoint/种子或物理协议掩盖失败。
大体系 virial 校验曾发现近零分量误差；独立坐标应变、边应变及不同精度归约诊断
定位到 FP32 总量原子加法。修正为 FP64 总量归约，并用独立 FP64 仿射边应变检查，
保持网络/力 FP32，未放宽原 `atol=1e-4, rtol=1e-3` 校验阈值。
诊断：[virial_diagnostic.json](results/throughput_a100/virial_diagnostic.json)。
一次取消作业遗留 JIT 构建锁，已在独立 sbatch CPU 分配中按原 build.ninja 补完构建，
保留旧锁并唤醒等待者，未修改源代码或编译选项。

额外尝试捕获 DPA lower 的 CUDA Graph，当前接口因未 pinned 的 CPU 缓冲输入拷贝失败；
日志保留，不据此声称 DPA 永远不能捕获，也未把该失败改成 slower fallback 后冒充
graph 结果。正式 DPA 基线采用其已确认生效的 compile+Triton level 2。

远端根目录：

- A100：`/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017`。
- 4090：`/home/ylzhang/lrx/chorus_hp_throughput_20261017`。

所有 A100 计算均通过 `sbatch/GPU-A100/XHPC-A100`，最高同时三张 A100；
4090 最高一条测试进程，源码/缓存/PID/日志均在上述 lrx 任务目录。
正式轨迹、模型、checkpoint、训练集和已有部署包未被修改。
收尾时本任务无活动/等待的 Slurm 作业，4090 测试 PID 已退出。
