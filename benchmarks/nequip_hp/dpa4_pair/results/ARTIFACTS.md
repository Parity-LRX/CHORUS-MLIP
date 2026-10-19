# 工件与运行身份

远端根目录：`/public/home/sps-xia/rxlin/nequip_chorus_dpa4_pair_20261018`。
本地只编辑代码/文档和同步记录；参数统计、构型生成、数值资格、编译、计时及汇总全部由指定 A100 batch 作业执行。

## 源码和环境

- `source_v1/nequip_root`：上一轮已经核验的 NequIP 原生 OEQ overlay（source_v9）。
- `source_v1/dpa_root/deepmd`：现有 DPA4 环境的独立源码快照；原安装目录不修改。
- `source_v1/tools/pair/prepare.py`：只按构造参数量配对，时间早于任何性能结果。
- `source_v2`：正式资格和计时使用的完整快照，所有子进程的完整命令留在 `commands.json` 和 batch 日志。
- `report_tools_v3`：汇总器补充识别 Inductor 内联后的具体 Triton 内核名；不改变被测模型或计时。
- 当前分发的 worker 也补充了这些 profiler 名称，数学执行和计时边界与 source_v2 相同。

两边统一 PyTorch `2.11.0+cu126`。NequIP 使用现有 OpenEquivariance `0.6.8` 预编译扩展。
各自 Python 环境保留原依赖，旧 ABI 的可选 PyG 扩展被隔离；没有安装或改写共享环境。
网络及力 FP32，TF32/AMP off，保留各框架原生 reduction dtype。
CUDA Graph capture/replay 由运行期 API guard 拒绝，Inductor cudagraphs 关闭。

## 作业标识

| Job ID | 内容 |
| --- | --- |
| 92533 | 参数量构造、公共周期输入、固定初始化 |
| 92535 | 两层 Final 的 DPA / NequIP 推理和训练 pilot，含原式及参数梯度 |
| 92536 | lmax=2 / 两层 / Final 正式两轮同卡比较 |
| 92537 | lmax=2 / 两层 / Persistent 正式两轮同卡比较 |
| 92538 | lmax=3 / 三层 / Final 正式两轮同卡比较 |
| 92539 | lmax=3 / 三层 / Persistent 正式两轮同卡比较 |
| 92541 | 四组完整记录的远端汇总 |

以上作业全部 `COMPLETED / 0:0`。四组正式配对、32 个隔离子进程及汇总均通过；
没有 OOM、数值失败或遗留运行作业。完整输出、训练参数梯度、同卡 UUID、compile 和 T2 实际内核资格均被核对。

所有作业使用 `GPU-A100 / XHPC-A100`；每组包含两边各两个独立进程回合，并检查同一 GPU UUID。
第一轮 NequIP→DPA，第二轮 DPA→NequIP。运行时三张 A100 先可用，第四组由调度器等待资源后执行。
各组之间不拼接硬件计时，GPU UUID 与作业身份保留在汇总 JSON。

## 本地记录布局

- `preparation/`：参数量构造探针、预先声明的通道网格和配对选择。
- `pairs/`：每组模型配置与精确参数量；初始化 `.pt` 留在远端。
- `raw/`：两轮逐次计时、数值差异、峰值 allocated/reserved、完整命令和 profiler。
- `report/`：四组同卡的正式 Markdown 和 JSON 汇总。
- `logs/`：batch 资源、命令和运行结果。

共享输入 `.pt`、原初始化、Triton/Inductor/OEQ 缓存全部留在远端任务目录，不复制到本地。

## 资格与计时定义

每个子进程先在同一固定初始化上对照各自原式的完整 E/F/virial；训练同时核验全部损失参数梯度。
DPA compact 图另外对照其原生周期邻居构建接口，核验相同几何上的输出。
正式计时包含输入复制、位移重算和完整输出；训练另包含统一 E/F/virial 损失、混合反传及 AdamW。
每种尺寸使用三组变化输入，9 次预热，两轮各 30 次推理或 18 次训练；邻居搜索、资格与冷编译不计时。

DPA 推理确实执行 level-2 value 路径：profiler 中的 `_rotate_mix_fwd_kernel`、
`_stack_gemm_m0_kernel` 与 `_stack_gemm_m1_kernel` 对应原始 `so2_rotate_mix` 和 `so2_mixing_stack`。
Inductor 内联后，原 custom-op 名可能消失，因此汇总器同时检查这些具体 GPU 内核。
训练保留 compile 和 level-2 配置，但 DPA 源码的 `not self.training` 分支不执行推理专用 value 核。

参数相近不表示表达方式、FLOPs 或训练精度相同；这些结果比较当前执行效率。
不能用 NequIP 自身 before/after 的提升替代此次跨架构结果。
