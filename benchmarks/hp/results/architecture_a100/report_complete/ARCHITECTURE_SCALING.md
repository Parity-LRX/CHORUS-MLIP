# CHORUS / DPA4 架构扩展同卡比较

固定初始化的执行扩展测试；不是训练精度、等容量或长轨迹验证。Final 与 Persistent 使用相同结构矩阵，correlation=3、rank16。

神经网络/力 FP32、TF32 off，virial FP64。DPA compile=1、Triton level2，保留原 mmax=1。每行在同一作业、同一物理 A100 顺序测量。

CHORUS 使用已验证的固定图 CUDA Graph；DPA 使用官方 compiled full/lower 入口。每点取 15 次同步 walltime 中位数；编译、参照生成和捕获准备均不计时。

DPA C48 不在官方融合 value path 的支持宽度 {32,64,96,128} 内；其 compile 生效，但 level2 value path 回退，单独标为 T2-unavailable，不据此给融合基线加速比。

原公式在测量尺寸验证通过后才给加速比；参照显存受限但可计时的点仅列原始耗时。OOM 与数值失败分别记录。

## 1024 原子 / 50 邻居

| 配置 | Scope | CHORUS / DPA C | 参数量 CHORUS / DPA | CHORUS ms | DPA lower ms | 加速比 | reserved GiB CHORUS / DPA | CHORUS 状态 | DPA 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| L2 / 2 层 | final | 128 / 32 | 941654 / 650428 | 9.343 | 15.071 | 1.61 | 1.15 / 1.24 | PASS | PASS / T2 |
| L2 / 2 层 | persistent | 128 / 32 | 1179423 / 650428 | 10.786 | 15.071 | 1.40 | 1.21 / 1.24 | PASS | PASS / T2 |
| L3 / 2 层 | final | 128 / 32 | 1559448 / 863356 | 21.035 | 24.583 | 1.17 | 2.11 / 1.93 | PASS | PASS / T2 |
| L3 / 2 层 | persistent | 128 / 32 | 1938724 / 863356 | 24.747 | 24.583 | 0.99 | 2.11 / 1.93 | PASS | PASS / T2 |
| L2 / 2 层 | final | 192 / 48 | 1869910 / 1296800 | 15.470 | 25.295 | — | 1.78 / 2.04 | PASS | PASS / T2-unavailable |
| L2 / 2 层 | persistent | 192 / 48 | 2299487 / 1296800 | 17.378 | 25.295 | — | 1.87 / 2.04 | PASS | PASS / T2-unavailable |
| L2 / 2 层 | final | 256 / 64 | 3109462 / 2616164 | 20.041 | 31.351 | 1.56 | 2.51 / 2.34 | PASS | PASS / T2 |
| L2 / 2 层 | persistent | 256 / 64 | 3779999 / 2616164 | 22.512 | 31.351 | 1.39 | 2.56 / 2.34 | PASS | PASS / T2 |
| L2 / 3 层 | final | 128 / 32 | 1552470 / 866333 | 13.136 | 20.345 | 1.55 | 1.48 / 1.45 | PASS | PASS / T2 |
| L2 / 3 层 | persistent | 128 / 32 | 2028008 / 866333 | 17.823 | 20.345 | 1.14 | 1.62 / 1.45 | PASS | PASS / T2 |
| L2 / 4 层 | final | 128 / 32 | 2163286 / 1082238 | 16.973 | 25.535 | 1.50 | 1.86 / 1.71 | PASS | PASS / T2 |
| L2 / 4 层 | persistent | 128 / 32 | 2876593 / 1082238 | 24.955 | 25.535 | 1.02 | 2.09 / 1.71 | PASS | PASS / T2 |
| L3 / 3 层 | final | 192 / 48 | 5235992 / 2392833 | 49.626 | 58.467 | — | 4.25 / 3.86 | PASS | PASS / T2-unavailable |
| L3 / 3 层 | persistent | 192 / 48 | 6568752 / 2392833 | 66.609 | 58.467 | — | 4.41 / 3.86 | PASS | PASS / T2-unavailable |
| L3 / 4 层 | final | 256 / 64 | 12384408 / 5908454 | 90.469 | 95.332 | 1.05 | 7.04 / 5.00 | PASS | PASS / T2 |
| L3 / 4 层 | persistent | 256 / 64 | 15441468 / 5908454 | 127.597 | 95.332 | 0.75 | 7.25 / 5.00 | PASS_WITH_SUPPLEMENTAL_ORIGINAL | PASS / T2 |

## 10000 原子 / 50 邻居

| 配置 | Scope | CHORUS / DPA C | 参数量 CHORUS / DPA | CHORUS ms | DPA lower ms | 加速比 | reserved GiB CHORUS / DPA | CHORUS 状态 | DPA 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| L2 / 2 层 | final | 128 / 32 | 941654 / 650428 | 72.560 | 122.051 | 1.68 | 10.41 / 11.70 | PASS | PASS / T2 |
| L2 / 2 层 | persistent | 128 / 32 | 1179423 / 650428 | 82.959 | 122.051 | 1.47 | 10.90 / 11.70 | PASS | PASS / T2 |
| L3 / 2 层 | final | 128 / 32 | 1559448 / 863356 | 200.760 | 212.106 | 1.06 | 18.38 / 18.40 | PASS | PASS / T2 |
| L3 / 2 层 | persistent | 128 / 32 | 1938724 / 863356 | 235.582 | 212.106 | 0.90 | 19.06 / 18.40 | PASS | PASS / T2 |
| L3 / 3 层 | final | 192 / 48 | 5235992 / 2392833 | — | — | — | — / — | CAPTURE_WORKFLOW_OOM | REFERENCE_OOM |
| L3 / 3 层 | persistent | 192 / 48 | 6568752 / 2392833 | — | — | — | — / — | CAPTURE_WORKFLOW_OOM | REFERENCE_OOM |
| L3 / 4 层 | final | 256 / 64 | 12384408 / 5908454 | — | — | — | — / — | EXECUTION_OOM | EXECUTION_OOM |
| L3 / 4 层 | persistent | 256 / 64 | 15441468 / 5908454 | — | — | — | — / — | EXECUTION_OOM | EXECUTION_OOM |

## 时间增长（相对同批 L2 / C128 / 2 层基线）

| 原子数 | 配置 | Scope | CHORUS 耗时倍数 | DPA 耗时倍数 |
| ---: | --- | --- | ---: | ---: |
| 1024 | l3 | final | 2.25 | 1.63 |
| 1024 | l3 | persistent | 2.29 | 1.63 |
| 1024 | c192 | final | 1.66 | 1.68 |
| 1024 | c192 | persistent | 1.61 | 1.68 |
| 1024 | c256 | final | 2.15 | 2.08 |
| 1024 | c256 | persistent | 2.09 | 2.08 |
| 1024 | d3 | final | 1.41 | 1.35 |
| 1024 | d3 | persistent | 1.65 | 1.35 |
| 1024 | d4 | final | 1.82 | 1.69 |
| 1024 | d4 | persistent | 2.31 | 1.69 |
| 1024 | l3c192d3 | final | 5.31 | 3.88 |
| 1024 | l3c192d3 | persistent | 6.18 | 3.88 |
| 1024 | l3c256d4 | final | 9.68 | 6.33 |
| 1024 | l3c256d4 | persistent | 11.83 | 6.33 |
| 10000 | l3 | final | 2.77 | 1.74 |
| 10000 | l3 | persistent | 2.84 | 1.74 |

所有参数量均为对应实际构造模型计数。两种通道定义不同，不按通道数字主张等容量；本轮不包含新增架构的训练或精度结论。

## 验证与资源边界

PASS_WITH_SUPPLEMENTAL_ORIGINAL 表示计时文件之外的完整尺寸原公式核对通过；原计时及此前失败记录不修改。

CAPTURE_WORKFLOW_OOM 发生在 HP 普通全尺寸预热之后的 CUDA Graph 捕获/重放核验流程，不等同于普通前向无法运行。REFERENCE_OOM 表示未加速参照受限，不能据此断言加速入口无法运行。

最大 Persistent 的独立参照采用逐原子 product 分块及原模块重算，并先与 128 原子未分块原式核对；分块仅用于参照，不进入计时。CPU 暂存的早期尝试及内存保护取消记录保留。

汇总状态：PASS: 19, PASS_WITH_SUPPLEMENTAL_ORIGINAL: 1, CAPTURE_WORKFLOW_OOM: 2, EXECUTION_OOM: 2。

Slurm 状态及退出码见 [slurm_accounting.txt](slurm_accounting.txt)。
