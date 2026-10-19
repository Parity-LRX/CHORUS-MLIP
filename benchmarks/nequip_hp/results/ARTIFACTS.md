# 运行与资格记录

任务根目录：`/public/home/sps-xia/rxlin/nequip_chorus_oeq_native_20261018`。
所有数值任务均使用 `sbatch / GPU-A100 / XHPC-A100`；本地只读写源码、文档和已生成记录。
没有修改共享 Python 环境，也没有修改原来的 NequIP 训练、MD 或源码目录。

## 执行边界

PyTorch `2.11.0+cu126`，OpenEquivariance `0.6.8` 的现有预编译扩展。
每份 before/after 记录由同一作业在同一张 A100 上顺序产生，固定模型初始化与输入。
每个方案参数量、rank、CG 路径及 checkpoint 键相同。CUDA Graph API 被运行期拒绝，
Inductor cudagraphs 关闭；不使用 AOTI，也不使用 cuEquivariance。

推理为完整 E/F/virial。训练为 E/F/virial 损失、力混合反传和 AdamW，
不是能量前向或单个 TP 核的计时。NequIP 原生 FP32 virial 定义不变。
初始化、邻居构建、核编译和数值核验均在计时之外；4 次预热后逐次同步记录 walltime。
峰值 allocated/reserved 均保留。每个尺寸训练前重新加载相同状态，方案间不复用训练后的权重。

## 源码快照

快照同时含 `nequip_root/`（NequIP v0.6.2 加 overlay）、`chorus/` 和 `tools/`。

| 快照 | 作用 |
| --- | --- |
| source_v1 | 原公开 overlay 的 OEQ TP-only 基线和环境验证 |
| source_v2 | rank 批次化、相位前投影、共享 bank 映射的首次数值资格 |
| source_v4 | 首次性能矩阵：三维 bank Linear 退化为 BMM，推理回退，仅作诊断 |
| source_v6 | 显式二维 GEMM 修正，训练与推理恢复收益 |
| source_v7 | Final 中性层接入 OEQ TP convolution；保留完整对照 |
| source_v8 | 标量首层保留普通 TP、相位 hook 兼容与完整单元测试；正式 C64 矩阵 |
| source_v9 | 未启用相位融合时跳过 hook 检查；论文 C84 补测和最终单元回归 |

当前 overlay 对应 source_v9 的数值实现。基准工具后来只增加了随代码分发的
历史基线文件定位，不改变被测算子。历史适配器保存在
`../reference/openequivariance_dense_rank.py` 和远端 `legacy/`，不进入生产执行路径。
它保留旧 rank-by-rank 单位阵 OEQ Hermitian TP，作为主速度基线。
旧模型的相位初始化不作为差异来源；三个执行方案都加载当前相同权重。

## 作业

| 作业 | 内容 | 结果 |
| --- | --- | --- |
| 92471 | 环境及原 OEQ TP 的 E/F、力损失梯度、旋转核验 | PASS |
| 92472 | 首轮 3 组密度、4 组整网资格 | PASS |
| 92473–92476 | 修正 Slurm 尺寸列表传递前的启动 | CANCELED，不进入性能结论 |
| 92477–92480 | source_v4 诊断性能矩阵 | COMPLETE，推理回退版本未选用 |
| 92481、92490、92499 | 阶段单元回归 | 每次 21 passed |
| 92485–92488 | source_v6 显式 GEMM 性能矩阵 | COMPLETE |
| 92491–92493 | 完整资格及中性卷积差分核验 | PASS |
| 92494–92497 | source_v7 性能矩阵 | COMPLETE |
| 92500 | source_v8：3 组密度、4 组整网，含三种执行方案的完整输出/梯度 | PASS |
| 92501 | source_v8 单元回归，含干预 hook、变化拓扑和孤立节点 | 23 passed |
| 92502 | C64/R16 Final，L2/两层 | COMPLETE |
| 92503 | C64/R16 Persistent，L2/两层 | COMPLETE |
| 92504 | C64/R16 Final，L3/三层 | COMPLETE |
| 92505 | C64/R16 Persistent，L3/三层 | COMPLETE |
| 92506 | C64 正式汇总 | COMPLETE |
| 92507 | C84/R16 Final，L2/两层 | COMPLETE |
| 92508 | C84/R16 Persistent，L2/两层 | COMPLETE |
| 92509 | source_v9 最终单元回归 | 23 passed |
| 92510 | C84 论文宽度汇总 | COMPLETE |

## 本地记录

- `report/`：正式 C64 同卡结果；`paper_width_report/`：C84 补测。
- `raw/*_v8/`、`raw/*_v9/`：逐次计时、显存、完整输出差异和参数梯度差异。
- `raw/optimized_v8/optimization_qualification.json`：算子/整网差分、软属性、混合二阶梯度、U(1)/O(3) 核验。
- `raw/unit_v9/unit_92509.xml`：最终单元回归。
- `logs/`：正式作业命令、资源和输出；完整历史日志留在远端任务根目录。
- `diagnostics/` 与其余 `raw/`：未选用或中间配置，包含首次推理回退结果。
- 原初始化 `.pt` 和所有缓存留在远端任务目录，不为展示报告复制大文件。

profiling 中实际出现 `libtorch_tp_jit::jit_conv_backward`，确认 Final 的非标量
中性层执行了 OEQ fused convolution，而不只是配置了开关。
相位诊断/干预 hook 存在时保留原相位网络调用，防止观测或干预被融合绕过。

显存没有统一下降：例如部分小配置训练的 reserved 上升；C64/L3 Final 在
1024 原子的 inference reserved 也略高。相关数字保留在表内，不能概括成全档更省显存。
这些是固定初始化结构的执行测试，不是新训练精度或 MD 物理验证。
