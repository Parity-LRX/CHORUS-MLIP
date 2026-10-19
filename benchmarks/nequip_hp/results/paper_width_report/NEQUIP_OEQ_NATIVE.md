# NequIP–CHORUS：原生 OpenEquivariance 优化

同卡顺序测量，网络 FP32、TF32/AMP off，不使用 CUDA Graph 或 AOTI。
推理包含完整 E/F/virial；训练包含 E/F/virial 损失、力混合反传和 AdamW。
主基线是远端已有的 OpenEquivariance Hermitian 优化版；TP-only 结果另列。
各执行方案加载同一份固定初始化，参数、rank、CG 路径和相位方程不变。

通道配置：84x0e；rank=16。

## 1024 原子、50 邻居

| scope | L / 层数 | task | 原优化 ms | 新优化 ms | 加速 | reserved GB 原/新 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| final | 2 / 2 | inference | 27.273 | 24.856 | 1.097× | 3.238/3.081 |
| final | 2 / 2 | train | 70.723 | 62.981 | 1.123× | 5.899/6.247 |
| persistent | 2 / 2 | inference | 33.066 | 29.936 | 1.105× | 3.412/3.263 |
| persistent | 2 / 2 | train | 87.028 | 77.129 | 1.128× | 6.967/6.019 |

## 全部尺寸

| scope | L / 层数 | N | task | TP-only ms | 原优化 ms | 新优化 ms | 加速 | reserved GB 原/新 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| final | 2 / 2 | 1024 | inference | 28.075 | 27.273 | 24.856 | 1.097× | 3.238/3.081 |
| final | 2 / 2 | 1024 | train | 73.744 | 70.723 | 62.981 | 1.123× | 5.899/6.247 |
| persistent | 2 / 2 | 1024 | inference | 37.540 | 33.066 | 29.936 | 1.105× | 3.412/3.263 |
| persistent | 2 / 2 | 1024 | train | 96.912 | 87.028 | 77.129 | 1.128× | 6.967/6.019 |

GB 使用十进制峰值 reserved；allocated 与逐次计时保存在原始 JSON 中。
邻居构建、原式核验和 OEQ 核冷启动不计入稳态。输入图复制、完整输出和优化器更新计入相应 step。
此表比较固定初始化结构的执行开销，不是重新训练后的等精度或物理预测比较。

## 实现

- Hermitian 收缩将独立 rank 放到批次轴，使用单重度 OEQ TP 和标量 +/-1 常量权重。
- 相位网络首层按边/源节点/目标节点分块，节点投影在 gather 前每个节点只计算一次。
- 元素输出映射先做共享 bank GEMM，再按节点属性混合，避免逐节点复制整个权重 bank；保留软属性梯度。
- Final 的角动量非零中性层使用 OEQ 融合 TP convolution；标量首层保留普通 TP，避免额外 workspace。
- 相位诊断或干预 hook 存在时自动保留原相位网络调用，以维护原来的观测和干预语义。
- 保留原 Parameter 对象、checkpoint 键、非零相位初始化与所有 Hermitian 实/虚路径。
