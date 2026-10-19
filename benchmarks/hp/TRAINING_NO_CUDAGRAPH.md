# 无 CUDA Graph 的训练优化

训练吞吐由算子、内存布局、计算图裁切及整网融合提升，不使用 CUDA Graph capture、
replay 或固定地址的训练重放。编译可用于融合，但编译器内部 CUDA Graph 必须关闭。

## 最新验收结果

A100、CHORUS-Persistent（`persistent_t1x.pth`，1,029,967 个可训练参数）、50 邻居、
完整 E/F 损失 / 混合反传 / AdamW。单独计时进程采用延长预热、
正反顺序各 50 步，三个实时批次轮换。数值资格来自同源、同 checkpoint、同输入和
同执行选项的独立原式对照；输出、全部参数梯度及参数更新检查通过。

| 原子数 | CHORUS ms | DPA compile ms | 训练加速 | reserved GB：CHORUS / DPA |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 62.972 | 131.643 | 2.090× | 4.230 / 10.383 |
| 4096 | 114.571 | 254.134 | 2.218× | 8.420 / 20.728 |

两轮单独配对均超过 2×，reserved 显存减少约 59%。FP32、AMP/TF32 off，全程没有
CUDA Graph 或外部 TP 算子库。最新 342 项回归通过。完整原式资格作业为 92284，
单独计时为 92289，回归为 92285，均正常完成。
见[最终同卡配对与逐项记录](results/training_native_a100/stable_training_v1/report/SELECTED_TRAINING_RESULTS.md)。

## 大原子数与大模型扩展

2026-10-18 扩展测试沿用 `source_v23` 算子/整网实现，不改变训练计时路径。
原 Persistent checkpoint 扩展到 6144、8192、10000、16384 原子；模型矩阵包括
C128/lmax3/2 层、C256/lmax2/2 层、C128/lmax2/4 层、C192/lmax3/3 层及
C256/lmax3/4 层，各在 1024/4096 原子测试 Persistent 与 Final。

模型扩展复用既有固定构造器初始化及其原始 U 张量，全部卷积 TP 系数可训练；
这些是模型规模的吞吐实验，不是新训练的精度结果。DPA 使用对应 lmax/块数和
C32/C48/C64 配置，同时记录双方实际参数量，不把不同通道定义视为等参数量。
三个变化批次、正反顺序各 50 步，完整 E/F 反传/AdamW；无 CUDA Graph、无 AMP/TF32。

初始加载器的 U 维度不匹配日志保留在 `large_training_v1`；修复为加载原初始化的
U 张量后，大模型记录进入 `large_training_v2`，不覆盖原失败。大尺寸 OOM、128 原子
原式核验和本尺寸原式核验分别记录，OOM 不用于计算速度比。
原子数扩展作业 `92290_0` 已完成，同卡正反序各 50 步：

| 原子数 | Persistent ms | DPA compile ms | 加速 | reserved GB：Persistent / DPA |
| --- | ---: | ---: | ---: | ---: |
| 6144 | 168.055 | 376.097 | 2.238× | 12.575 / 30.564 |
| 8192 | 222.161 | 501.627 | 2.258× | 16.750 / 41.027 |
| 10000 | 267.981 | 两次 OOM | — | 20.433 / — |
| 16384 | 435.470 | 两次 OOM | — | 33.437 / — |

两个可配对尺寸的 reserved 均约减少 59%。本轮双方统一使用 expandable allocator；
8192 原子的 DPA 已跑通，覆盖旧容量筛选中该尺寸 OOM 的结论。以上大尺寸的完整
原式梯度 oracle 尚未完成，数值资格与既有 2048/4096 原子结果分开标注。
原始记录在 `results/training_native_a100/large_training_v1/raw/atom_scale/`。
大模型 `92294_[1-5%3]`、汇总 `92297` 及热点采样 `92303` 均已完成，
总 GPU 并发未超过三张。最大模型为 Persistent 15,441,468 参数 /
Final 12,384,408 参数。结构扩展的显存收益保留，但并未普遍达到两倍吞吐；
高角动量组合模型目前慢于对应 DPA 配置。
见[完整扩展表](results/training_native_a100/large_training_v1/report/LARGE_TRAINING_RESULTS.md)
和[结果分析与热点](TRAINING_SCALE_FINDINGS.md)。

## 验收协议

- 完整 step：原子能量、总能量、保守力、E/F 损失、混合二阶反传、AdamW 更新。
- 原 checkpoint、全部可训练参数、原模型及相位/Hermitian 方程保持不变。
- 网络与力使用 FP32，关闭 AMP/TF32；不改变梯度或参数更新的数值阈值。
- 计时包含常规 Python/调度开销；分别记录同步 walltime、GPU event 和显存。
- 单尺寸多批次与变 N/E 序列分开报告。坐标、元素和拓扑实时消费，不复用旧输出。
- CHORUS 原式、旧 HP、新 HP 交替同卡比较；DPA 使用真实可微 compile 训练路径。
  DPA 的 level-2 value path 受上游 `not self.training` 限制，不声称训练启用了该核。
- 编译/JIT 与预热不计入稳态，但单独记录。微核、完整 step 和数据流水线分开归因。
- 原式输出、全部参数梯度及优化器更新逐项核验；超差和 OOM 均保留。

按用户随后确认的优先级，小的逐元素舍入偏差保留记录，但不阻塞吞吐优化。
原输出阈值、逐张量梯度范数阈值和优化器更新阈值继续强制执行，非有限值报错；
`PASS_WITH_ELEMENTWISE_EXCEPTIONS` 与完整逐元素 `PASS` 仍分开标注。

## 已有基线

下表是前轮原始同卡记录，尚不是新的变批次测试。计时单位 ms。

| 原子数 / 邻居 | HP eager，无 Graph | DPA 可微 compile | 当前结论 |
| --- | ---: | ---: | --- |
| 128 / 50 | 45.724 | 18.777 | HP 未领先 |
| 512 / 50 | 44.857 | 39.186 | HP 未领先 |

历史 HP Graph 的 12.837/24.767 ms 及 1.46–1.58× 加速比不作为训练验收结果。
原始 JSON 保持不变：
[前轮逐项记录](results/throughput_a100/summary_final_release_v70.json)。

## 本轮优化

首先处理相位上下文 gather–SiLU 的训练伴随。原前向已经融合，但在力损失训练时，
其可微 VJP 仍展开为多次 gather、逐元素计算及 scatter。新增 ICTC-O3-HP 联合一阶/
混合二阶核，保留实时 cotangent、参数依赖及三阶以上的精确可微回退。该候选默认关闭，
通过 `configure_model(..., training=True, phase_context_mixed_fusion=True)` 显式测试；
当前配对没有显示明确整网收益，不替换原训练执行。

新结果只有通过远端无 Graph 核验与同卡配对后才记入这里；不以微核收益代替整网收益。

## 2026-10-18 同卡正反顺序复测

完整 step、三个变化批次、不使用 CUDA Graph；各两轮，每轮 12 次。

| 原子数 / 50 邻居 | 旧 HP ms | 联合伴随候选 ms | DPA compile ms |
| --- | ---: | ---: | ---: |
| 128 | 44.197 | 43.928 | 19.137 |
| 512 | 44.700 | 44.869 | 39.568 |

本次相位伴随融合没有形成明确的整网吞吐收益，两种尺寸仍慢于 DPA。
变 N/E 的 128/256/512 序列也完成了完整 step，但新旧 HP 均有少量参数梯度逐元素
超差；输出、逐张量梯度范数及首步更新检查通过，不等于完整逐元素验收通过。

计时保留在不可变 `source_v1` / `source_v4` 快照中。当前 `source_v5` 恢复原默认训练
路径，仅以 `phase_context_mixed_fusion=True` 暴露该实验核；默认/实验两种路径的
47 项针对性回归通过，覆盖一至三阶导数、可学习参数、部分求导目标和状态身份。
原始逐元素异常、显存、每次计时、变尺寸检查和失败日志见
[逐点报告](results/training_nograph_a100/report_complete/TRAINING_NOGRAPH_RESULTS.md)。

## 原生训练融合后续

母版新增两条训练接口：元素库的原生分组 GEMM/权重伴随，以及将 TP 系数提前
合入径向输出层的小权重库。CHORUS 通过显式开关测试，所有参数继续实时更新。
独立宽 gate 乘积伴随通过测试但没有稳定整网提速，不启用为默认。
新配对将 profiler 移到所有计时轮次之后，隔离诊断对后续轮次的影响；每个 step
仍重新构建 autograd 图，不使用 CUDA Graph 或外部 TP 库。

`source_v10` 同卡正反顺序配对结果：

| 原子数 / 50 邻居 | 现有 HP ms | 原生元素库 + 径向头融合 ms | DPA compile ms | 融合 reserved GB / DPA |
| --- | ---: | ---: | ---: | ---: |
| 128 | 45.051 | 41.366 | 19.155 | 0.338 / 0.698 |
| 512 | 46.728 | 42.054 | 39.574 | 1.177 / 2.854 |
| 2048 | 84.115 | 72.653 | 132.692 | 4.534 / 11.146 |

当前融合让训练吞吐相对现有 HP 增加约 9–16%；2048 原子相对 DPA 为 1.826×，
reserved 减少约 59%。128/512 小图仍未超过 DPA，不能表述为全面达到两倍。
最新执行路径及实验核覆盖的 342 项回归通过。64 项早期针对性检查还覆盖零 TP 系数、
实时参数、缺失元素、部分求导目标、混合/三阶导数、参数身份与 checkpoint keys。
原始数据见[原生训练配对](results/training_native_a100/report_complete/NATIVE_TRAINING_RESULTS.md)。

4096/8192 扩展初次在未计时的原式梯度校验中 OOM，未计为融合实现失败或速度结果。
后续校验单独尝试分块 checkpoint、保存张量 CPU 卸载和显存分配器设置；这些措施
均不进入正式 GPU step 的模型执行，原始 OOM 日志保留。

4096 原子的完整原式校验与正反顺序配对已完成：旧 HP 154.132 ms，融合
133.348 ms，DPA compile 254.773 ms，即 1.911×。双方使用相同 expandable allocator；
融合 reserved 为 8.422 GB，DPA 为 20.728 GB。三个变化批次、全部参数梯度及
优化器更新检查通过，两次融合运行均无逐元素异常。原式校验的 checkpoint/CPU
卸载只用于未计时的 oracle，正式训练 step 不使用这些措施。
记录见[4096 原子同卡配对](results/training_native_a100/scale4096/report/NATIVE_TRAINING_RESULTS.md)。

8192/10000 原子的容量筛选分别为 256.325/308.087 ms，reserved 为
17.731/21.720 GB；DPA 在两个尺寸均连续两次 OOM。这些尺寸的同规模原式
校验尚未完成，只作为容量记录，不计算 DPA 的速度比。
记录见[大规模容量筛选](results/training_native_a100/capacity_screen/report/TRAINING_CAPACITY_SCREEN.md)。

## 目标节点复用的训练主线

后续改动集中在联合 TP 的实际热点：删除会被完整覆盖的梯度缓冲清零；按实时
autograd 依赖裁掉无用分支；将输出 cotangent 的混合 JVP 改为目标节点所有权归约；
一阶伴随在目标节点内复用宽三流 cotangent，再归约边/源节点梯度。极坐标相位系数
的混合伴随也合并为单核。邻居拆分和输入角动量分组没有明确整网收益，保持关闭。

对应训练选项（原模型已为 CUDA FP32，卷积 TP 使用 pytorch 后端）：

```text
--hp-operators generated --hp-checkpoint-contraction
--hp-training-species-fusion --hp-radial-coefficient-fusion
--hp-phase-coefficient-mixed-fusion --hp-training-destination-jvp
--hp-training-destination-first 1
```

这组开关保留原参数、实时拓扑和每步 autograd 构建，不启用 CUDA Graph。
342 项回归及 CLI 参数入口检查通过，记录为 `native_full_regression_92285.xml`。

6144 原子同卡筛选的两轮为 184.672/184.873 ms，DPA 为 376.476/377.124 ms。
原式完整梯度参照在 GPU 上 OOM；分配器/缓存基础设施尝试也未跑通，因此该点
保留为吞吐筛选，不标为同尺寸数值资格已完成。两次 OOM 日志和全部计时均保留。
