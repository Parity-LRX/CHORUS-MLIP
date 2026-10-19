# 高角动量大模型训练优化：实现与实测

本轮目标：Persistent、Final 在完整 E/F 损失、混合反传及 AdamW 更新中，
相对同卡 DPA4 compile 至少 1.5×，且 allocated/reserved 峰值显存均更少。
FP32、TF32/AMP off，不使用 CUDA Graph，不借用外部 TP 算子库。
母库仍为 ICTC-O3-HP。构造器状态、U、种子、原参数及 trainability 保持原样。

## source_v40：近参数预算的正式配对

最新正式表包含三种模型的共同 1024 原子测点与最大模型的 512 原子测点，共 8 项；
正反两轮实测均超过 2×，allocated 和 reserved 峰值显存均低于各自近参数量的
DPA4 compile。共同的 1024 原子测点如下；每步包含 E/F 损失、全参数混合梯度和 AdamW 更新，
不使用 CUDA Graph。GB 为 10^9 字节。

| CHORUS 配置 | scope | CHORUS ms | DPA compile ms | 吞吐比 | reserved GB：CHORUS / DPA |
| --- | --- | ---: | ---: | ---: | --- |
| C128/lmax3/2层 | Persistent | 71.826 | 167.377 | 2.330× | 4.079 / 10.991 |
| C128/lmax3/2层 | Final | 61.897 | 164.258 | 2.654× | 3.951 / 10.169 |
| C192/lmax3/3层 | Persistent | 174.655 | 423.154 | 2.423× | 9.370 / 23.683 |
| C192/lmax3/3层 | Final | 143.518 | 384.894 | 2.682× | 8.382 / 21.978 |
| C256/lmax3/4层 | Persistent | 336.117 | 826.129 | 2.458× | 16.656 / 41.249 |
| C256/lmax3/4层 | Final | 280.870 | 670.558 | 2.387× | 14.680 / 36.465 |

本轮新一阶路径组调度在最大模型 1024 原子上，进一步将 Persistent 从 349.483 ms
降至 336.117 ms，Final 从 285.132 ms 降至 280.870 ms，吞吐增益分别约 4.0% 和 1.5%。
第四卡的 C128/C192 短筛选经正式配对确认：同卡相对旧原生调度，吞吐分别再增
6.9%/5.8% 和 9.9%/6.5%（Persistent/Final）。两者改用缓存组 16＋一阶/混合路径组专门化，
DPA 参数匹配保持不变。径向有界融合保留为容量选项。
上述原生调度增益与更换 DPA 参数预算后的模型间速度比分列。

完整最新 8 测点、两轮原始数据、显存及原式资格见
[正式配对报告](results/training_native_a100/high_l_optimization_v20261018/report_parameter_matched/LARGE_TRAINING_RESULTS.md)。
此前 12 测点（含 C128/2048、C192/512 的旧调度）保留在
[前轮配对](results/training_native_a100/high_l_optimization_v20261018/report_parameter_matched_previous/LARGE_TRAINING_RESULTS.md)。
同尺寸原式资格已覆盖 C128/1024、C192/512 和 C256 Final/512；更大尺寸计时与资格分栏。
最大 Persistent/512 的原参考反传仍 OOM：边分块、节点 SC 分块、节点＋通道 SC 分块
三次尝试均保留。其原生训练完成全部 50 步正反计时，三批 128 原子原式核验通过，
但不标注为 512/1024 原子原式核验通过。

## 新比较口径：同角动量、同深度、近参数预算

2026-10-18 后续比较固定双方相同 `lmax` 与 interaction/块数，仅扫描 DPA4 通道数，
保持其 rank、focus、SO(2) mixing、grid 分支及自动 FFN 设置不变。
选择依据为实际可训练参数量，不使用吞吐或显存选择对照架构。
Persistent 与 Final 分别匹配。92370 已完成预先规定的 C32..160、步长 4 扫描：

| 配置 | scope | CHORUS C / 参数 M | DPA C / 参数 M | DPA 相对参数差 |
| --- | --- | --- | --- | ---: |
| lmax3 / 2层 | Persistent | 128 / 1.938724 | 48 / 1.752128 | −9.62% |
| lmax3 / 2层 | Final | 128 / 1.559448 | 44 / 1.583031 | +1.51% |
| lmax3 / 3层 | Persistent | 192 / 6.568752 | 76 / 6.506340 | −0.95% |
| lmax3 / 3层 | Final | 192 / 5.235992 | 72 / 5.407815 | +3.28% |
| lmax3 / 4层 | Persistent | 256 / 15.441468 | 108 / 15.373021 | −0.44% |
| lmax3 / 4层 | Final | 256 / 12.384408 | 96 / 12.152174 | −1.88% |

C128 Persistent 的最近对照仍有 9.62% 差异；DPA C52 已跳到 2.316937 M，
不为缩小参数差而另改自动宽度或其他架构选项。其余五项差异均在 3.3% 内。
近参数预算不等同于等精度；性能配对保持同卡、同变化批次、同完整训练步骤。
以下 source_v37 数据是旧固定配置基线，DPA C32/C48/C64，不匹配参数预算；
保留用于核函数优化前后追踪，不挪用其速度比作为新配对结果。

## source_v37 旧固定配置基线

此前 source_v37 轮在旧固定 DPA 配置下，整体 1.5× 目标尚未全部达到。其 1024 原子、
50 邻居、三个变化批次、正反各 50 步完整训练中的结果：

| 配置 | Persistent ms / 相对 DPA | Final ms / 相对 DPA | DPA compile ms | reserved GB：Persistent / Final / DPA |
| --- | ---: | ---: | ---: | --- |
| C128/l3/2层 | 76.208 / 1.336× | 65.131 / 1.564× | 101.852 | 4.079 / 3.951 / 7.567 |
| C192/l3/3层 | 191.919 / 1.246× | 152.084 / 1.572× | 239.115 | 9.370 / 8.298 / 14.647 |
| C256/l3/4层 | 349.559 / 1.167× | 284.612 / 1.433× | 407.791 | 16.656 / 14.680 / 25.042 |

所有测点的 allocated/reserved 峰值均更低。C128/C192 的 Final 两轮均超过 1.5×；
Persistent 与最大 Final 尚未达标。C128 两种 scope 的 1024 原子三批原式核验均通过，
没有逐元素异常；其余两个模型通过三批 128 原子核验，本尺寸参考实现 OOM。
全回归 396 项通过。

最终记录：[1024 原子配对](results/training_native_a100/high_l_optimization_v20261018/report_latest/LARGE_TRAINING_RESULTS.md)、
[更大原子数配对](results/training_native_a100/high_l_optimization_v20261018/report_selected/LARGE_TRAINING_RESULTS.md)、
[完整工件索引](results/training_native_a100/high_l_optimization_v20261018/ARTIFACTS.md)。

本地仅编辑、阅读；全部数值工作在 GPU-A100 / XHPC-A100 的 sbatch 作业中执行。
任务根：`/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018`。

## 实现和边界

- 后续 `training_first_schedule` 提供 `pruned` / `grouped` / `compact` /
  `compact_grouped`：一阶 VJP 依据实际 autograd 目标裁切无用输出，可为路径组
  独立分配寄存器，或按磁分量立即消费 CG 列；不删掉参数梯度依赖的中间值。
- `training_radial_first_bounded=True` 在 destination CSR 边块内生成一阶 gate VJP，
  随即消费隐藏层与实时 bank 投影。缓存径向混合路径已接入单遍输出 JVP 和组专门化。
  它降低 gate 反传的峰值存储，速度/容量候选分别选择。
- `training_mixed_compact=True`：按磁分量立即消费原 CG 基和切向基，缩短寄存器生命周期，
  因子化重复的混合伴随标量，保留 X/H/G/phase/输出 cotangent 全部目标。
- `training_mixed_cache=8/16/32/64`：warp 持有目标节点/路径组/通道片，跨邻居复用
  三流 cotangent。源特征采用 FP32 原子累加；边方向/相位 partial 有唯一写者再归约。
  当前要求通道数为 32 的倍数。三阶及以上保持原可微公式。
- `training_radial_tp_fusion=True`：新实时径向 bank–TP 组合伴随；前向 gate 仅本次调用内
  复用，不保存完整一阶 gate cotangent，不生成完整混合 gate 切向量。混合 gate 梯度分块
  投影回隐藏层和权重。实时径向权重、TP 零系数和高阶导数均保留。
  该实验当前较慢，不能当作已实现速度目标。
- `training_radial_cached=True`：用有界 GEMM 生成径向切向 gate，紧接着执行目标节点
  缓存 TP 混合伴随并消费 gate VJP；分块覆盖全部边，不保存完整一阶 gate cotangent。
  `radial_tile_edges` 控制缓冲区，训练 CLI 为 `--hp-training-radial-tile-edges`，默认 8192。
- `training_first_cache=True`：将分量级缓存用于一阶伴随。全部目标及高阶测试通过，
  当前整网比既有 destination-first 调度更慢，保持实验选项，不作为推荐速度配置。
- `training_mixed_fuse_jvp=True`：在目标节点持有者内，同时累加原混合伴随与输出
  cotangent 的 JVP，复用同一个 CG/切向基遍历。每个路径/磁分量/通道的输出都有唯一
  写者，不用输出原子累加。原生算子和 CHORUS 接口均为显式选项。
- `training_mixed_grouped=True` 与缓存调度组合时，按路径组分别实例化原生 kernel，
  分别分配寄存器；不改变 CG 路径或梯度目标。
- `training_terminal_fusion=True`：末端 live channel map 与原 rank projection 重新结合，
  每步重算小权重乘积并保留其梯度，不缓存可训练权重的冻结乘积。
- 训练 self-connection 接入已有原生元素分组 GEMM；标量元素映射也可用相同 live-bank 原语。
- `backward_parameters(loss, model)` / `--hp-parameter-gradients-only`：保留完整 E/F 参数混合
  梯度，只不计算优化器不使用的输入叶子梯度，仍支持梯度累积。DPA 对照采用相同
  `autograd.grad + accumulation` 口径。该边界不适用于还需消费输入叶子梯度的调用者，
  当前仅资格覆盖单设备。

所有选项默认关闭或保留原默认路径。未经过核验的候选不作正式性能结论。

## 验证记录

- 92366：新一阶原生核 8 项通过；后续测试工具的相对几何路径失败，已改为显式 SOURCE 路径。
- 92367：分块一阶生成器的循环标记计数失败；按全部路径组修复，未改原式或验收阈值。
- 92368：修复后实时径向的全目标/高阶测试 10 项通过，后续工具路径由 92371 替代。
- 92369：source_v38 一阶四候选，两 scope 各三批 128 原子核验和 1024 原子短筛选完成。
- 92371：source_v40 有界一阶/径向单遍 JVP 四候选，同样全部通过并完成筛选。
- 92370：同 lmax、同深度、按参数量选择 DPA 通道的扫描与固定初始化生成完成。
- 92372：source_v40 完整回归 414 项通过；92381 的匹配/报告协议 3 项通过。
- 92375：source_v40 / tools_v68，近参数量 DPA 与同卡旧/新原生调度的正式配对。
  master 的未启动前驱 92373、92374 已替换，未使用其任何计时数据。
- 92382：最大 C256 的独立 64 GB host-memory 计时作业；与高内存原式核验分开排程。
  92375_2 仅在 PENDING 状态替换，保留其他两个正在运行的 array task。
- 92383：用户允许四张卡后增加的第四张 A100；补测 C128/C192 的一阶裁切/分组调度，
  先做三批 128 原子资格，再做 1024 原子短筛选，不改变正在执行的正式配对。
- 92384：最大模型 512 原子的独立原式核验，256 GB host-memory，参考支路分块和 CPU 暂存。
  只在两个 160 GB 作业释放内存后启动，不以其单步测量作为吞吐。
- 92385：正式配对与核验的统一报告；92378 的旧 array 依赖在 PENDING 状态替换。
- 92388：source_v41 的完整回归 422 项通过。新增 8 项确保显式一阶调度在普通 backward、
  无 `training_destination_first` 且输入 X 固定时仍实际分派到新核，并验证原式梯度。
  本轮计时和大尺寸原式资格继续使用冻结的 source_v40；没有中途改动其快照。
- 92390：参考实现按节点分块的内存核验。原式分块及报告协议 6 项通过后，
  在同模型、同输入、同准则下重做最大 Persistent/512 的原式核验，仍 OOM；不进入计时。
- 92391：最终报告汇入独立原式资格；原有 OOM 尝试仍留在原目录和 qualification_attempts 中。
- 92393 / 92394：第四卡短筛选后的 C128/C192、1024 原子、50 步正反配对，
  加入 `first_scale_screen_v15` 候选；新旧原生与相同 DPA 参数匹配在同卡复测。
- 92400：参考 SC 进一步按原参数的可微通道视图分块，保留节点与全部 U 路径。
  ragged 通道、cor=2/3、高角动量及混合梯度和报告协议共 11 项通过；
  最大 Persistent/512 的原参考混合反传仍 OOM。相同 6 项分块数值测试在空闲 4090 也通过，
  4090 不提供任何本表性能数字，任务工件均在独立 lrx 目录。
- 92416：最终报告按 case 引用新一阶配对与既有最大模型配对，已完成；
  不移动或改写原始记录，所有原式参考尝试列在 qualification_attempts。
  92402 仅在 PENDING 状态替换；92417 的最终报告协议 5 项通过。

本轮自有计时、核验和报告作业均已结束，没有遗留计算作业。

- 92308：旧高阶核线程/输入角动量分组筛选，source_v23 / tools_v43，完成。
- 92311：compact 高阶筛选，source_v24 / tools_v44，完成。
  C128/lmax3/1024 原子 Persistent 短筛选约 108 ms → 92 ms；单独这一改动不够。
- 92312：source_v25 / tools_v45，14 项定向测试通过，涵盖实时径向 bank 切向量、
  全部混合目标和更高阶导数；随后径向实验筛选完成。
- 92317_0：source_v27 的旧 parameter-target API 筛选；其余未启动任务由新协议替代。
- 92319：source_v28 / tools_v48，三种架构、两个 scope 的 live terminal / 径向 / compact
  原式核验与短筛选。原式核验独立于计时。
- 92322：source_v28 完整回归，364 项全部通过，日志/XML 已留存。
- 92323：source_v29 / tools_v50，目标节点缓存混合伴随测试和筛选。
- 92326：source_v29 / tools_v51，三种架构、7 个规模的正反配对，各 50 步，完成。
- 92327 / 92332：有界径向候选先后出现编译重名、归约偏移错误；失败快照与日志保留。
  修复未改动模型、初始化、输入种子或数值验收阈值。
- 92336：source_v32 / tools_v54，12 项 TP/径向联合混合梯度及高阶测试通过；三架构、
  两 scope 的原式 128 原子三批核验和短测完成。径向融合省显存，但尚未获得吞吐收益。
- 92339：source_v33 / tools_v55，更大路径缓存组的原式核验和筛选。
- 92343_0：1024 原子、带参考分块/CPU 暂存的原式核验；仅参考支路使用暂存，不进入计时。
  未启动的 tasks 1–2 由最终选定候选的同尺寸核验替代。
- 92345：source_v34 / tools_v56，去除较慢的一阶实验选项，筛选 8k/32k/64k 径向边块。
- 92348：source_v35 / tools_v57，按 scope 从通过核验的候选选最快配置，进行新一轮
  50 步同卡配对；末尾独立执行带分块/CPU 暂存的 1024 原子原式核验。
- 92349：source_v35 全回归，382 项全部通过。
- 92352：source_v36 / tools_v58，单遍混合 TP/输出 JVP；6 项全目标/高阶定向测试通过。
  C128/l3/1024 原子短测 Persistent 76.785 ms、Final 65.473 ms，仅小幅改善。
- 92357：source_v37 / tools_v59，最大模型的分组实例化试验，先核验再筛选。
- 92355 / 92359：source_v36 / source_v37 全回归分别 388 / 396 项全部通过。
- 92361：source_v37 的最终 1024 原子、50 步正反配对。C128/C192 选择单遍 JVP，
  C256 选择分组＋单遍 JVP；不选择较慢的一阶伴随候选。
- 92348、92361 及汇总 92362、92363 均已完成，没有遗留运行作业。

最大模型的单遍/分组 profiler 中，高 L 混合伴随阶段（含输出 cotangent JVP）
由约 60.6 ms 降到 39.2 ms。一阶 TP 伴随仍约 49.6 ms；三组宽径向 GEMM 分别约
15.9、15.1、13.6 ms。上述 GEMM 是算子事件，不再与对应设备 GEMM 核时间叠加。

## 速度配置

以下为本轮三种固定架构的候选选择；最终速度结论以配对表为准。

```python
from chorus.hp import backward_parameters, configure_model

# case: "l3" / "l3c192d3" / "l3c256d4"，保持原模型及初始化。
configure_model(
    model, policy="generated", training=True, checkpoint_contraction=True,
    topology_backend="device", training_species_fusion=True,
    radial_coefficient_fusion=True, phase_coefficient_mixed_fusion=True,
    training_destination_first=1, training_mixed_destination=True,
    training_mixed_cache=16,
    training_mixed_fuse_jvp=True,
    training_mixed_grouped=True,
    training_first_schedule="grouped",
    training_terminal_fusion=True,
)
# loss 包括由 create_graph=True 得到的力损失。
backward_parameters(loss, model)
optimizer.step()
```

所有原可训练参数参与更新。容量配置可改用 8k 边块的实时径向 TP/VJP，
但本轮其吞吐较低，不与速度配置混为一项结果。

原式资格：三批 128 原子、输出、全参数梯度范数、逐元素异常记录、AdamW 更新。
小图资格不冒充大图原式资格；同尺寸原式 OOM 与融合 step OOM 分开记录。
大规模正式结果如下（source_v35，均为中位数；GB=10^9 字节）。

| 配置 | 原子数 | Persistent ms / 相对 DPA | Final ms / 相对 DPA | DPA compile ms | reserved GB：Persistent / Final / DPA |
| --- | ---: | ---: | ---: | ---: | --- |
| C128/l3/2层 | 1024 | 77.118 / 1.317× | 66.136 / 1.536× | 101.563 | 4.079 / 3.930 / 7.567 |
| C128/l3/2层 | 4096 | 273.433 / 1.389× | 237.294 / 1.600× | 379.746 | 15.947 / 15.261 / 30.044 |
| C192/l3/3层 | 1024 | 196.156 / 1.214× | 153.984 / 1.547× | 238.173 | 9.370 / 8.298 / 14.647 |
| C192/l3/3层 | 2048 | 382.294 / 1.214× | 301.962 / 1.537× | 464.121 | 18.312 / 16.425 / 29.182 |
| C256/l3/4层 | 1024 | 360.476 / 1.117× | 287.661 / 1.400× | 402.598 | 16.656 / 14.680 / 25.042 |
| C256/l3/4层 | 1536 | 531.306 / 1.117× | 428.054 / 1.387× | 593.644 | 24.889 / 22.895 / 37.457 |
| C256/l3/4层 | 1664 | 574.227 / 1.121× | 459.529 / 1.401× | 643.773 | 26.925 / 24.763 / 40.546 |

source_v35 的 C128/1024 原子、两个 scope 均通过本尺寸三批原式核验。
C192/C256 的本尺寸原式反传仍 OOM，即使参考支路启用分块和 CPU 暂存；它们的三批
128 原子核验均通过。更大原子数计时与本尺寸核验分开，不作范围外的合格标注。

配对表和原始记录：
[大规模完整配对](results/training_native_a100/high_l_optimization_v20261018/report_selected/LARGE_TRAINING_RESULTS.md)、
[候选速度/显存筛选](results/training_native_a100/high_l_optimization_v20261018/report_candidates/CANDIDATES.md)。

最大模型 1024 原子的独立 profiler：TP 混合核 157.859 → 40.860 ms。新热点为
一阶 TP 伴随 49.254 ms、目标 cotangent JVP 20.814 ms，以及宽径向 GEMM。
上层 autograd 事件与其子核重叠，不相加作为时间分解。

按此前 source_v35 的旧固定配置配对，三个大模型的 Persistent、Final 未全部达到 1.5×。
该旧轮所有规模的 allocated 和 reserved 峰值均更低；新参数预算配对见本文开头。

三种架构沿用既有固定状态：C128/lmax3/2层、C192/lmax3/3层、C256/lmax3/4层。
DPA 对应 C32/C48/C64、相同 lmax/块数；通道定义和参数量不同，不作等精度声明。
DPA 的 inference Triton level 2 虽已配置，上游训练分支并不执行该推理 value 核。
