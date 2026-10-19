# NequIP–CHORUS / DPA4 同卡近参数量比较

比较对象是当前原生 OEQ Final/Persistent 与 DPA4 compile + Triton level 2。
两边均不使用 CUDA Graph 或 AOTI；网络及力为 FP32，TF32/AMP 关闭。
所有构造、资格、基准和汇总都在 `sbatch / GPU-A100 / XHPC-A100` 内执行。

- [正式结果：全部尺寸、训练/推理和显存](results/report/NEQUIP_DPA4_MATCHED.md)
- [原始计时与汇总 JSON](results/report/summary.json)
- [源码、作业和工件身份](results/ARTIFACTS.md)

## 配对规则

先保留 NequIP 的 lmax=2/两层、lmax=3/三层及 rank=16，分别比较两个 scope。
DPA 的 n_blocks 与 NequIP 的 num_layers 对齐，lmax 相同；其余描述符选项沿用既有 DPA4 模板，
仅统一 4 Å cutoff，并调整通道数。DPA mmax=1、degree_channel radial rank=1 等分支不被裁减。

Triton level-2 value 核只支持 focus 宽度 32/64/96/128。
在这些宽度中选最接近原 NequIP 参数量的 DPA，再在 NequIP 的 8..192、步长 4 网格中选近参数量配置。
选择仅使用构造得到的参数量，不参考耗时或显存。NequIP 参数公式用五个独立构造点核验，
选中配置还会实际重建、重新计数并保存初始化；差异超过 5% 即停止。

| lmax / 层数 | scope | NequIP C / 参数量 | DPA C / 参数量 | DPA 参数差 |
| --- | --- | ---: | ---: | ---: |
| 2 / 2 | Final | 76 / 643,307 | 32 / 650,428 | +1.11% |
| 2 / 2 | Persistent | 72 / 653,753 | 32 / 650,428 | −0.51% |
| 3 / 3 | Final | 56 / 1,216,650 | 32 / 1,171,389 | −3.72% |
| 3 / 3 | Persistent | 40 / 1,216,150 | 32 / 1,171,389 | −3.68% |

模型间对照不要求相同通道数、张量积结构或 FLOPs；它检验同 lmax/interaction 深度、近参数预算下的实际执行效率。
这是固定初始化的系统测试，不是已训练模型的等精度比较。

## 计时边界

- 相同三组周期构型，H/C/N/O 四元素，4 Å cutoff，恰好每原子 50 邻居。
- 推理 N=128/1024/4096；训练 N=128/1024。
- 两边排除邻居搜索，输入复制、位移重算、完整 E/F/virial 均计入。
- DPA compact 输入独立对照其原生周期邻居构建接口，确认能量、力与 virial 相同。
- 训练使用相同 `(E/N)^2 + mean(F^2) + 0.01 mean((V/N)^2)`、混合反传及 AdamW，lr=1e-5、weight_decay=0。
- 每组一张 GPU，所有子进程使用同一 GPU UUID；两轮 NequIP→DPA、DPA→NequIP，进程隔离。
- 每次 9 次预热；每轮推理 30 次、训练 18 次，逐次同步 walltime。三组输入交替变化。
- 对各自原式核验完整输出与所有损失参数梯度；保留原生 reduction dtype。
- allocated/reserved 均记录；不能把 reserved 回退隐藏为 allocated 改善。

DPA 训练启用 compile 且保留 `DP_TRITON_INFER=2`，但上游训练逻辑不执行推理专用的 level-2 value 核。
推理则同时检查编译缓存、level-2 路径实例和实际 profiler 记录，不把环境变量等同于内核生效。

## 远端入口

任务根目录：`/public/home/sps-xia/rxlin/nequip_chorus_dpa4_pair_20261018`。
NequIP 和 DPA 代码均保存到任务源码快照，不修改现有库或既有训练/MD 任务。

```bash
sbatch --export=ALL,SOURCE=/TASK/source_v2,MODE=benchmark,RUN=formal_v2,CASE=l2d2,SCOPE=final \
  /TASK/source_v2/tools/pair/run.sbatch
```

`prepare.py` 构造配对与公共输入；`worker.py` 执行原式资格和完整计时；
`driver.py` 在同一分配内交替调用两边；`report.py` 只汇总完整、同卡的正式配对。
