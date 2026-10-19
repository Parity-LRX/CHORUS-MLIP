# NequIP–CHORUS 原生 OpenEquivariance 优化

当前实现同时覆盖 Final 与 Persistent 的推理和训练，不使用 CUDA Graph 或 AOTI。
普通 TP 保留 OpenEquivariance；CHORUS 的 rank 批次化、相位投影、元素 bank 映射
与 Final 中性层融合不改变参数、rank、CG 路径或模型方程。

- [正式同卡结果](results/report/NEQUIP_OEQ_NATIVE.md)
- [论文 C84/R16 宽度补测](results/paper_width_report/NEQUIP_OEQ_NATIVE.md)
- [对 DPA4 compile / Triton level 2 的同卡近参数量比较](dpa4_pair/results/report/NEQUIP_DPA4_MATCHED.md)
- [同骨干无 CHORUS 基线与数据流优化协议](dataflow_round/README.md)
- [最新数据流优化及 DPA4 同卡回归](dataflow_round/results/report/NEQUIP_NATIVE_DATAFLOW.md)
- [源码快照、数值资格与作业记录](results/ARTIFACTS.md)
- [NequIP 安装与开关](../../integrations/nequip/README.md)

## 对照与资格

三种执行方案加载同一份初始化和 checkpoint 键：

1. `tp_only`：原本公开 overlay 的 OEQ 普通 TP，其余 CHORUS 运算为 PyTorch。
2. `legacy`：远端已有的 OEQ Hermitian 优化版，使用 rank-by-rank 单位阵和元素 gather/BMM。
3. `optimized`：当前原生优化；主加速比对照 `legacy`，不只对照较慢的 TP-only 路径。

数值核验包括完整 E/F/virial、所有损失参数梯度、混合二阶导数、旋转等变性、
U(1) 不变性、任意软元素属性，以及诊断/干预 hook。中性 TP convolution 另外检查
变化的 N/E、非均匀邻居和孤立节点。优化保留非零相位初始化。

性能使用 C64/R16、L2/两层及 L3/三层。每个作业内在同一张 A100 顺序测量全部方案。
另用 C84/R16、L2/两层在 1024 原子下复测 Final 与 Persistent 的推理和训练。
网络与力为 FP32，TF32/AMP 关闭；NequIP 原生 virial 定义和 FP32 dtype 不变。
推理包含完整 E/F/virial；训练包含 E/F/virial 损失、力混合反传及 AdamW 更新。
推理覆盖 128/1024/4096 原子，L2 另测 10000 原子；训练测 128/1024 原子，均为 50 邻居。
固定初始化仅用于执行开销比较，不作为重新训练后的精度或物理结果。

四次预热后记录逐次同步 walltime；邻居构建、原式核验与算子冷启动不计入稳态。
数据图张量复制、完整输出和优化器更新包含在相应 step 中。
峰值 allocated、reserved 和原始样本同时保留，不只挑显存或速度的有利项。
reserved 并非每一档都下降；全部尺寸和显存回退项均保留在结果表中。

## 远端复现

所有数值脚本只在分配好的 batch 作业内运行。A100 入口：

```bash
sbatch --export=ALL,SOURCE=/TASK/immutable_source,RUN=persistent_l3,MODE=benchmark,SCOPE=persistent,CHANNELS=64,LMAX=3,LAYERS=3,SIZES=128:1024:4096 \
  /TASK/immutable_source/tools/run_qualification.sbatch
```

`SOURCE` 同时包含原版 NequIP 加当前 overlay、CHORUS 和本目录脚本。
`SIZES` 在 Slurm `--export` 中使用冒号，避免与环境变量的逗号分隔符冲突。
资格入口为 `MODE=optimization`；单元测试为 `MODE=unit`。所有缓存和产物留在任务根目录。

正式环境使用现有 OpenEquivariance 0.6.8 预编译扩展和 PyTorch 2.11，不修改共享环境。
旧 Torch ABI 的可选 PyG 扩展被隔离，NequIP 原本的纯 PyTorch scatter 实现仍保留。
这条支线不改变 ICTC-O3-HP 母版路线，也不改动既有 MD 或训练工件。

## 最新数据流优化

已补齐无 CHORUS 的 NequIP＋OEQ 同骨干基线。新的通用骨干数据流优化也用于该基线，
公共 checkpoint 张量逐键复用，不另调宽度。四组模型与 DPA4 配对保持不变。

1024 原子、50 邻居、完整 E/F/virial 下，本轮 Final/Persistent 推理提升
1.323×/1.312×（L2）及 1.405×/1.482×（L3），训练提升
1.241×/1.224×（L2）及 1.294×/1.341×（L3）。L3 Persistent 在 4096 原子下推理提升 1.531×。

1024 原子的四组训练均快于开启 compile 的 DPA4，但所有尺寸推理仍慢于 DPA4；
128 原子的训练也未超过 DPA4。推理 allocated 全部下降，训练 allocated/reserved 则有个别回退。
机制、同骨干基线、所有尺寸和显存见[本轮结果](dataflow_round/README.md)。

## 数据流优化前与 DPA4 的同卡配对

另按相同 lmax / interaction 层数、0.5%–3.7% 参数量差完成两轮比较。
1024 原子 / 50 邻居时，Final 的完整 E/F/virial 训练速度为 DPA4 的 1.202×（L2/两层）
和 1.068×（L3/三层）；Persistent 两层基本持平，三层为 0.796×。
四组推理速度均低于 DPA4，且推理 reserved 显存更高。
DPA4 推理实际执行 compile + level-2 value 内核；训练执行 compile，推理专用 value 内核不参与训练。
具体通道配对、全部尺寸和显存见[正式报告](dpa4_pair/results/report/NEQUIP_DPA4_MATCHED.md)，
复现规则见[配对协议](dpa4_pair/README.md)。
