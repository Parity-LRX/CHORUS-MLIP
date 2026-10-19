# 工件与运行身份

远端根目录：`/public/home/sps-xia/rxlin/nequip_chorus_dataflow_20261018`。
所有数值工作与汇总使用 `sbatch / GPU-A100 / XHPC-A100`；本地仅编辑和同步记录。

## 固定输入与源码

- `pairs`、`inputs` 只读链接到上一轮任务
  `/public/home/sps-xia/rxlin/nequip_chorus_dpa4_pair_20261018` 的固定配对与公共周期输入。
- `source_v1`：基线诊断工具及优化前实现。
- `source_v2`：本轮数据流实现，原式/混合梯度资格通过。
- `source_v3`：相同数学实现，补齐完整单元测试所需的既有 `chorus` 兼容包路径。
- `source_v4`：增加优化后详细 shape/memory profiling 与正式汇总器。
- `source_v5`：正式轮次；无 CHORUS 基线也使用本轮通用骨干数据流优化。
- `previous` 始终从上一轮不可变 `source_v2/nequip_root` 导入，不被本轮源码覆盖。
- Python、Torch 2.11.0+cu126、OpenEquivariance 0.6.8、DPA 源码与原生 reduction dtype 沿用上一轮。
  不安装或修改共享环境，不接触旧训练/MD 工件。

四组 NequIP 配置、rank、通道、层数、参数与初始化保持不变；DPA4 配对和所有训练超参保持不变。
移除 CHORUS 的基线只载入其构造器所需的公共 checkpoint 张量，逐键检查名称与形状。
基线不增加通道或重新初始化公共骨干。

## 作业

| Job | 内容 |
| --- | --- |
| 92548–92551 | 四组同骨干无 CHORUS / materialized / 完整 CHORUS 的初始诊断 |
| 92552 | 首次完整单元测试；因缺少兼容包导入路径失败，13 项已通过 |
| 92553 | 三个 density 与四个整网原式/混合梯度资格，通过 |
| 92554 | 补齐包路径后的完整单元测试，26 项全部通过 |
| 92555–92556 | L3 Final / Persistent 优化前后推理与训练筛查，通过 |
| 92557–92560 | 初版正式调度，因统一无 CHORUS 基线的通用数据流实现而主动取消；不纳入汇总 |
| 92561 | L2/两层 Final 正反序两轮，四种方案同卡 |
| 92562 | L2/两层 Persistent 正反序两轮，四种方案同卡 |
| 92563 | L3/三层 Final 正反序两轮，四种方案同卡 |
| 92564 | L3/三层 Persistent 正反序两轮，四种方案同卡 |
| 92565 | 依赖四组全部成功后的远端正式汇总 |

92548–92551、92553–92556 以及 92561–92565 均为 `COMPLETED / 0:0`。
四组正式比较的 64 个隔离进程全部成功，每组同一 GPU UUID；无 OOM、非有限输出或遗留运行作业。
两轮中的每个训练进程均核验完整输出和所有损失参数梯度，所有训练更新后参数及梯度保持有限。
汇总同样由指定 A100 batch 生成，没有在本地执行数值聚合。

每个计时进程记录完整命令、PID、作业号与 GPU UUID；`commands.json` 记录隔离子进程次序及退出码。
CUDA Graph capture/replay 由运行期 API guard 拒绝，Inductor cudagraphs 关闭。
DPA4 推理同时核验编译缓存、Triton level-2 实例和具体 GPU 内核，训练不调用上游推理专用 value 核。

## 本地记录

`raw/` 保留原始 JSON、完整 worker 日志、profiler 与 shape/memory profiling；
`logs/` 保留 batch 命令与资源记录；`report/` 为远端 batch 生成的正式汇总。
共享输入、初始化、编译缓存留在远端，未在本地运行数值脚本或汇总。
