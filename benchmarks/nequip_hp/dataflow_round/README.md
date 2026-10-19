# NequIP–CHORUS 数据流优化轮次

目标是减少 Persistent 完整 E/F/virial 推理中的宽边张量、布局复制和梯度累加，
同时回归 Final/Persistent 的训练与推理。沿用上一轮固定的四组模型、初始化、
公共周期输入与 DPA4 参数配对，不修改网络结构或训练超参，不使用 CUDA Graph/AOTI。

- [正式同卡结果：全部尺寸、训练/推理及显存](results/report/NEQUIP_NATIVE_DATAFLOW.md)
- [原始样本与资格汇总 JSON](results/report/summary.json)
- [源码与作业身份](results/ARTIFACTS.md)

## 本轮结果

1024 原子、50 邻居，完整 E/F/virial：

| L / 层数 | scope | 推理 ms 前 → 后 | 训练 ms 前 → 后 | 推理 allocated / reserved 降幅 |
| --- | --- | ---: | ---: | ---: |
| 2 / 2 | Final | 22.849 → 17.272 | 58.747 → 47.337 | 9.7% / 1.6% |
| 2 / 2 | Persistent | 26.678 → 20.340 | 70.285 → 57.403 | 9.4% / 5.1% |
| 3 / 3 | Final | 51.748 → 36.836 | 135.516 → 104.746 | 12.8% / 2.8% |
| 3 / 3 | Persistent | 70.476 → 47.543 | 183.639 → 136.927 | 11.9% / 12.1% |

L3 Persistent 在 4096 原子下，推理从 257.985 降到 168.553 ms，提升 1.531×；
allocated/reserved 分别下降 12.0%/12.1%。128 原子时提升较小。

与同卡 DPA4 相比，1024 原子的 Final/Persistent 训练分别为
1.494×/1.227×（L2）和 1.389×/1.067×（L3）。所有尺寸的推理仍慢于 DPA4，且推理显存更高；
128 原子的训练也仍慢于 DPA4。训练显存有回退：1024 原子的 L2 Final/Persistent allocated
增加 1.3%/2.2%，L3 Persistent reserved 增加 3.2%。全部记录保留在正式表中。

26 项单元测试、原式/混合梯度资格及 64 个正式隔离进程全部通过，无 OOM 或非有限输出。

## 同骨干基线

`no_chorus` 关闭 CHORUS，保留同一组 NequIP 通道、lmax、interaction 层数及所有
公共骨干 checkpoint 张量。基线参数量因移除 CHORUS 而减少，不另调宽度补齐。

初始诊断包含三种路径：无 CHORUS 的 OEQ 融合卷积、无 CHORUS 的显式边消息路径、
优化前的完整 CHORUS。正式比较中的无 CHORUS 基线也使用本轮通用数据流优化；
它与优化后 CHORUS 共享相同骨干实现。

## 本轮改动

- irrep 特征与权重一次 `split`，减少独立切片反传对整块张量的填零和累加。
- 对满足原 e3nn Linear 路径约束的内部共享权重，执行显式通道 GEMM；
  保留原始参数和路径归一化，将归一化乘法放到参数矩阵侧。
- 线性邻居归一化放到聚合后的节点侧，减少对宽边消息的除法和反传中间量。
  二次 Hermitian density 的归一化位置保持原式。
- OEQ Hermitian 输入按 irrep 一次拆分，保留原 rank-as-batch TP。
- hook、非 CUDA、不受支持的 Linear 结构及 TorchScript 调用走原 Linear。

`NEQUIP_CHORUS_NATIVE_DATAFLOW=0` 可关闭新的 Linear 执行和聚合归一化重排；
一次 split 的等价 irrep 拆分仍保留。执行计划只缓存 Python 布局元数据，不缓存权重或激活。

## 热点核对

L3/三层 Persistent、1024 原子的同卡单次 profiling 中：

| 算子 | 优化前 Self CUDA ms | 优化后 Self CUDA ms |
| --- | ---: | ---: |
| `aten::add_` | 16.343 | 4.531 |
| `aten::copy_` | 11.007 | 8.041 |
| `aten::mm` | 7.291 | 7.321 |
| OEQ TP backward | 5.665 | 5.674 |

下降主要来自累加与搬运；TP 和 GEMM 耗时基本不变。单次 profiling 用于定位，
正式加速比取未启用 profiler 的两轮 step 计时。
原始记录见[优化前](results/raw/formal_v5/l3d3/persistent/previous_inference_r0.profile.txt)
和[优化后](results/raw/formal_v5/l3d3/persistent/optimized_inference_r0.profile.txt)。

## 正式协议

每组在同一张物理 A100 上顺序运行
`no_chorus → previous → optimized → dpa`，再反序执行第二轮；每个方案/任务独立进程。

- 推理 N=128/1024/4096；训练 N=128/1024，均为 50 邻居、4 Å cutoff。
- 三组构型交替；排除邻居搜索，计入输入复制、几何重算、完整 E/F/virial。
- 相同 E/F/virial 损失、混合反传、AdamW(lr=1e-5, weight_decay=0)。
- 每轮预热 9 次，推理采样 30 次、训练 18 次；同步 walltime 合并取中位数。
- 同时保留 allocated/reserved、原始样本、逐轮中位数与 profiler。
- 网络/力 FP32，TF32/AMP off；保留各框架原生归约 dtype。
- DPA4 推理核验 compile 缓存、level-2 实例及实际内核；训练保留 compile，
  上游推理专用 level-2 value 核不参与训练。
- 完整输出及所有损失参数梯度对照各自原式；独立资格覆盖混合二阶导数、
  O(3)/U(1)、软元素属性和 hook。单元测试共 26 项。

所有构造、数值检查、计时与汇总均由 `sbatch / GPU-A100 / XHPC-A100` 执行。
固定配对规则见[上一轮配对协议](../dpa4_pair/README.md)。

```bash
sbatch --export=ALL,SOURCE=/TASK/source_v5,MODE=formal,RUN=formal_v5,CASE=l3d3,SCOPE=persistent \
  /TASK/source_v5/tools/dataflow/run.sbatch
```

任务根目录：`/public/home/sps-xia/rxlin/nequip_chorus_dataflow_20261018`。
`pairs`、`inputs` 只读复用上一轮固定工件；新的源码、缓存、日志和结果均留在本任务根目录。
