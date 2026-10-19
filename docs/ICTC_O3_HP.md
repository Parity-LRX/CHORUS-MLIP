# CHORUS 的 ICTC-O3-HP 执行后端

该后端改变执行方式，不改变 CHORUS 的参数化、相位系数、Hermitian 路径、密度秩、
Final/Persistent 作用域或已训练权重。原参考路径仍是默认；必须显式配置 HP。

本轮真实 checkpoint、双硬件配对计时和保留的压力输入诊断见
[2026-10-17 远端验证报告](../benchmarks/hp/RESULTS_20261017.md)。
后续原生 checkpoint-U、相位专核、Persistent 和 DPA4-mini 同卡比较见
[吞吐专项报告](../benchmarks/hp/THROUGHPUT_20261017.md)。
L3、C192/C256 与 3/4 层的固定初始化执行扩展见
[架构矩阵](../benchmarks/hp/results/architecture_a100/report_complete/ARCHITECTURE_SCALING.md)。
当前 C128/L2 大图特化的两倍收益不能直接外推到高阶及更深 Persistent；逐点计时、
实际 DPA Triton 路径和显存边界在该表分别记录。

## 安装与调用

### Installation and compatibility

论文对应的 package 版本号保持 CHORUS 0.1.0 和 ICTC-O3-HP 0.2.12。
`[hp]` extra 锁定 HP Git 提交 `9fdea47eef620013e9ea591f7d90534bb27671c8`，
HP API contract 1；不靠同一个版本号区分新旧实现。
HP 仍使用 `mace_ictc` Python namespace；不要在同一环境安装旧 MACE-ICTC
覆盖它。启用 HP 时同时核对 distribution 版本、实际导入的 namespace 版本和 API
contract；错误版本或旧源码遮蔽会给出明确诊断，`auto` 回到参考路径，
`generated` 报错。显式 `reference` 不检查或导入 HP。

从当前源码安装时，一条命令自动获取固定的 HP 修订，无需手工并排放置两个源码树：

```bash
python -m pip install -e '.[hp]'
# 或直接从 main 安装；其元数据仍锁定上面的不可变 HP 提交：
python -m pip install 'chorus-mlip[hp] @ git+https://github.com/Parity-LRX/CHORUS-MLIP.git@main'
```

也可以安装本轮构建的 CHORUS wheel，其 `[hp]` 仍自动获取同一 Git 依赖：

```bash
python -m pip install '/path/to/chorus_mlip-0.1.0-py3-none-any.whl[hp]'
```

这一路径需要 Git 与 GitHub HTTPS 访问；普通依赖仍从配置索引/当前环境解析。
直接 Git 依赖不会因为同目录放了同版本 HP wheel 就自动变成离线依赖。
不将旧索引上的同名 0.2.12 当作本轮实现，也不覆盖上传论文对应的软件包版本。

不装 `[hp]` 的普通安装和原有 `[full]` 组合均保持原依赖集合；参考路径仍为默认。
安装 HP 不会自动开启它，也不改变 checkpoint/训练设置。CHORUS 基础 Python 要求
仍为 >=3.9；HP extra 继承母版的 >=3.10 要求，CUDA 执行另要求 Linux 和兼容工具链。
本轮 HP 修订的命令使用独立 `ictc-hp-*` 前缀，不覆盖 CHORUS 原有 `mff-*` 命令。
配套安装、缺失 HP 时的参考回归及已安装 wheel 的原生数值验收见
[双库安装验收](../benchmarks/hp/release/README.md)。
库及新内核的使用遵循 ICTC-O3-HP 自身许可；这里不复制或重新授权它的实现。
吞吐专项需要本轮更新后的母版源码：`mace_ictc.hp.CoherentGraphMessage`、
`species_channel_linear` 和 `configure_symmetric_contraction(..., allow_reduced_cg=True)`。
仅有旧版 0.2.12 安装包不足以提供这些新增接口。

```python
from chorus.hp import check_model, configure_model

# Load the original checkpoint/model first, then move it to its execution device.
# The caller chooses FP32 and the original trainability flags explicitly.
print(check_model(model, training=True))
info = configure_model(model, training=True, policy="generated")
print(info)

# Existing forward / coordinate autograd / force-loss backward stay unchanged.
# Restore the original formulas, without rebuilding or changing parameters:
configure_model(model, policy="reference")
```

`generated` 对结构不支持的情况报错；`auto` 返回参考路径及原因。配置不执行
`requires_grad_(False)`，不会把 FP64 模型改成 FP32，也不替用户冻结仍可训练的 TP
基础系数。启用后关闭 TF32、设置 float32 matmul precision 为 `highest`；该进程设置
不会自动恢复。训练的径向网络、通道权重、SC 权重、相位和密度权重继续实时求导。

执行句柄不注册为模型子模块，因此 checkpoint keys、parameter identity 和优化器引用
保持不变。句柄不进入 `state_dict`；加载、替换 Parameter、改变 dtype/device 后重新配置。
直接编辑密度 CG 常量或 quadratic form 后也应重新配置，不能继续用旧计划。

训练 CLI 使用 `--hp-operators generated --hp-layout fused_pack`，并选择
`--dtype float32 --conv-tp-backend pytorch`。可学习的 TP 系数通过实时 gate 乘积
接入原生算子，其参数梯度与力损失混合导数均保留；不需要为启用 HP 添加
`--freeze-conv-tp-weight`。该开关仍是独立的模型架构选择。
继续训练已有 checkpoint 时先保留其原有 trainability，由覆盖检查决定能否启用。
不要同时开启 `--train-makefx-compile`。当前验证为单设备，每个 GPU 运行独立用例；
DDP/多设备训练尚不在此适配器的覆盖内。

训练性能验收不使用 CUDA Graph capture/replay，也关闭编译器内部的 cudagraphs。
训练收益以普通调度下完整 E/F 损失、混合二阶反传及优化器更新为准，另检查实时
批次和 N/E 变化。历史固定图重放数字仅保留作诊断。最新 A100 / 50 邻居复测在
2048/4096 原子分别为 62.972/114.571 ms，相对 DPA compile 为 2.090×/2.218×；
reserved 显存减少约 59%（4096 原子为 8.420 GB / 20.728 GB）。两轮单独配对均超过
2×。原式全参数梯度与参数更新检查通过，342 项回归通过；128/512 的旧配对未领先，
本次最终复测聚焦 2048/4096。
协议、原始逐点结果和后续扩展见
[无 CUDA Graph 训练专项](../benchmarks/hp/TRAINING_NO_CUDAGRAPH.md)。

训练融合开关为 `--hp-training-species-fusion`、`--hp-radial-coefficient-fusion` 与
`--hp-phase-coefficient-mixed-fusion`、`--hp-training-destination-jvp`，
需与 `--hp-operators generated` 同用。它们仅改变执行，不冻结参数，也不启用 CUDA Graph。

混合反传根据真实的 autograd 依赖专门编译，裁掉未被请求的目标分支；坐标、拓扑、
权重和 cotangent 仍逐步读取，N/E 也不固定。目标节点 JVP 归约可通过
`configure_model(..., training=True, training_mixed_destination=True)` 单独测试，
它将输出 cotangent 的逐边原子累加改为节点所有权归约，默认关闭。邻居拆分、
输入角动量分组等调度候选没有明确整网收益，继续保持关闭。
一阶伴随还可用 `--hp-training-destination-first 1`（API 参数
`training_destination_first=1`）复用目标节点的宽三流 cotangent，避免每条边重复读取；
2/4 为更多邻居 owner 的实验值，0 保留原选择。所有输入梯度和混合导数仍实时计算。

## 融合边界

1. ICTC-O3-HP 的原生 gather–TP–gate–scatter 消息核代替逐边角向张量。
2. `CoherentGraphMessage` 一次计算原 Cartesian TP，同时累加中性、实部、虚部。
   只保存原有 `[E,P*C]` 径向 gate，不扩展三倍 gate/节点特征。联合坐标伴随直接
   汇合三流 cotangent；各 degree 直接输出为下游通道映射布局，避免切片反向的
   补零和大缓冲相加。物理布局保持角分量在前、通道连续，直接连接共享 GEMM。
   需要 charged-input VJP 时使用源节点 CSR 联合伴随；不需要时使用逐边块归约，
   不为首层额外生成节点特征梯度。当前专用计划支持 1–4 个相位 head。
   sm80 的 C128/L2 五输入一阶伴随在 E>=327680 时复用目标节点 cotangent，
   直接归约源梯度并合并 coalesced 方向/相位 partial；较小大型图使用四路 CSR
   邻居所有权再合并节点 cotangent，其它结构保留原调度。独立算子的 `adjoint_splits=1` 可强制旧联合
   方案，默认 `0` 为结构选择。各阶特征与中性/双流使用多输出 split/unbind，
   避免独立切片反向的重复整宽补零。相位上下文也只返回当前求导目标需要的梯度。
   训练的联合 mixed-adjoint 同时收缩五类输入的切向量及实时输出 cotangent，
   不重建三个逐边消息或三份调制 gate。几何二阶目标同样覆盖；三阶及以上保留
   精确可微原公式。可学习 TP 系数与原径向 gate 相乘，原参数身份与优化器引用不变。
   更大 head 数保留三倍通道图核路径；`coherent_fusion=False` 可显式选择该对照。
   相位 MLP 首层拆成节点上下文投影与逐边径向投影，再融合 gather、偏置和 SiLU；
   `phase_context_mixed_fusion=True` 可显式测试母版 gather–SiLU 的联合训练 VJP/
   混合二阶核，实时 cotangent 和投影参数梯度均保留；三阶以上回到精确可微公式。
   该候选尚未显示明确整网吞吐收益，默认关闭，只允许与 `training=True,
   phase_fusion=True` 同用。极坐标/Cartesian 系数另有专核。
3. Hermitian 密度将保留的实/虚 CG 路径一次写出；提供一阶 VJP 和混合二阶伴随。
   `charge2` 参数匹配对照也保留其原符号。每个非零 CG 系数原值进入计划，不做阈值截断。
   一阶 VJP 改用输入所有权 gather，避免多项路径在同一输入上原子争用；混合/高阶
   导数仍由原可微方程给出。
4. bridge-U 收缩调用 HP 分解电路；原 U、basis bridge 和全部训练权重直接借用。
   将离散元素索引从模型传入，而非对可微 `node_attrs` 偷做 argmax。
   `checkpoint_contraction=True` 对实际保存的 reduced-CG U 做原生因子化，不重新
   生成或更换 CG 基底。加载时 `LAMMPS_MLIAP_MFF.from_checkpoint(...,
   reuse_checkpoint_u=True)` 直接使用原 native bridge-U checkpoint 中的 U。
5. 方向谐波、通道映射、元素权重库 GEMM 和 normalized SiLU 调用同一 HP 库。
   Persistent charged update 共享矩阵按节点/双流/磁分量合成一个 GEMM，避免
   数千个小 BMM。几何、径向网络、残差注入和能量读出的方程不变。
6. 冻结 eval 推理的末层 Persistent 更新与紧随的 Hermitian rank 投影相乘合并，
   直接计算 `P[g W_prev q_prev + (1-g) W_in q_in]`，不再物化末层完整 charged 状态。
   仅作用于没有下一层消费者的末层、完整密度路径；保留原通道数与密度秩。
   `terminal_fusion=False` 可关闭该代数融合。训练或可训练权重自动走原更新路径；
   冻结权重的乘积按原参数 identity/version/device/dtype 更新，CUDA Graph 仍由
   完整模型状态签名失效保护。
   冻结元素权重库还可按实时元素索引在设备分组，只计算各原子所需的一个权重库
   GEMM，并保留一阶及混合高阶输入伴随。训练中可显式测试
   `training_species_fusion=True`，使用母版的原生实时元素库 GEMM 与权重归约核；
   二者互为双线性伴随，保留可学习权重及混合高阶导数，不再为所有元素计算
   矩阵后用 `where` 选取。该训练选项默认关闭，以完整 E/F step 配对计时。
7. 可选的一维径向 VJP 沿边长传播原径向 MLP 的值与导数，在 TP 伴随中直接
   收缩为每条边的距离梯度，不物化完整 `[E,P*C]` gate 梯度。原 gate 保留一份，
   导数投影使用可复用的分块缓冲。该路径仅用于冻结、bias-free 的原径向 MLP
   和 Bessel/Gaussian 基；训练或不支持的径向结构保留原路径。高阶导数回到完整
   可微原公式，不使用截断或近似导数。它是低显存选项，不是新的默认速度优化。
   组合 `radial_gate_pipeline=True, radial_destination=True` 后，gate 值与导数均按
   边块生成并当场消费，不保留全图 gate；目标节点 cotangent 只按组加载，
   方向/相位 partial 使用有界缓冲。`radial_destination` 需要同时启用
   `radial_jvp` 和 `radial_gate_pipeline`，不改变参数或训练路径。
   额外启用 `radial_cache_gates=True` 则保留当前前向中深层的 gate 值，只分块
   生成其距离导数，省去完整 gate 梯度及 gate 重算。这是默认速度档与全有界
   容量档之间的选项。首层保持原径向 MLP，在匹配的大型 sm80 图上使用无多余
   输入梯度的伴随调度。缓存仅在同一次前反向内复用，坐标、元素或拓扑变化时
   仍重新计算 gate。该选项要求同时开启前三个径向开关。
8. `large_graph_fusion=True` 默认允许结构自动选择，但适配器只在 sm80、全参数已由
   调用者冻结、eval 推理及 E>=327680 时启用本轮大型图特化。支持的 32 宽相位
   MLP 与两个 head 合并为一个值核/一个 VJP 核；方向节点投影仍按节点 GEMM。
   周期边向量和 FP32 力 scatter 同时融合，virial 保持 FP64；首个中性 C128/L2
   消息使用四 warp/full-edge 调度。实时输入/拓扑每次读取，稳定 CSR 亦每次重建，
   不缓存轨迹输出。`large_graph_fusion=False` 可关闭这些大型图特化；训练、可训练
   参数或不匹配结构自动保留已验证的普通路径，不冻结或修改原参数。
9. 训练可显式测试 `radial_coefficient_fusion=True`，把可学习 TP 路径系数提前乘入
   径向末层的实时小权重矩阵，随后生成 gate；不再物化逐边的宽系数乘积。
   原 Parameter、零系数梯度及优化器引用保留，FP32 小舍入差异逐项记录。
   另有 `gate_coefficient_fusion=True` 测试独立乘积的联合伴随；当前小图配对
   没有显示稳定吞吐收益。两个系数融合选项均默认关闭。

`training_mixed_threads=32/64/128` 可独立测试联合 TP 混合反传的线程分块，
默认 `0` 保留原结构调度。该选项只允许训练，改变的是线程/归约调度而非公式；
保留原种子、检查点、全部梯度与 FP32，性能仍按完整 step 配对确认。

```python
# The caller has already made model FP32, frozen, and eval.
configure_model(model, policy="generated", checkpoint_contraction=True,
                radial_jvp=True, radial_gate_pipeline=True,
                radial_destination=True, radial_tile_edges=65536)
# Native fused execution; no CUDA Graph capture/replay.

# Middle-memory option: retain this forward's deep gates, not their gradients.
configure_model(model, policy="generated", checkpoint_contraction=True,
                radial_jvp=True, radial_gate_pipeline=True,
                radial_destination=True, radial_cache_gates=True,
                radial_tile_edges=65536)
```

`radial_tile_edges` 约束径向导数投影的边块大小；较小块减少显存，但增加调度与
矩阵乘开销。`radial_jvp=False`（默认）保留此前最快的联合 TP/VJP。
以下容量数字来自此前的 Graph 重放实验，作为历史容量记录，不进入当前原生无 Graph 性能表。
已训练 Persistent C128/L2 在 100,000 原子、50 邻居完整 E/F/virial 下通过核验，
该容量路径约 872 ms、reserved 26.52 GB；同卡 source-CSR 容量路径约 1019 ms。
进一步扩展至 131,072 原子后为 1151 ms、34.66 GB，同样通过完整原式与重放核验。
32,768 原子时 reserved 从最快默认的 31.90 GB 降至 8.74 GB，代价是耗时从
234 ms 增至 282 ms。Final 的更大尺寸仍处于力分量数值稳定性排查，未完成扩展验收。
完整范围及失败记录见[吞吐专项](../benchmarks/hp/THROUGHPUT_20261017.md)。
另有实验性 `radial_fusion=True`（宽隐藏层 VJP 分块）用于消融。
逐通道重算与寄存器 GEMM、常驻块调度和独立权重布局
候选没有进入默认执行路径；失败及消融记录保留在吞吐报告中。

需要逐边自密度的对照走 HP 独立边 TP，随后执行原有逐边密度流程，不将其错误改成
完整邻居密度。`stream_fusion=False` 可比较三次独立 HP 图聚合；
`density_fusion=False` 保留原 CG 密度公式；`symmetric_contraction=False` 保留原 SC。
旧训练 checkpoint 若使用 reduced-CG，可保留原 SC，或显式使用
`checkpoint_contraction=True` 加速实际保存的 U。训练 CLI 对应
`--no-hp-symmetric-contraction` / `--hp-checkpoint-contraction`；
`check_model` 也接受相同的 `symmetric_contraction=False` 覆盖检查。

```python
configure_model(model, training=True, policy="generated", layout="fused_pack",
                stream_fusion=True, density_fusion=True,
                symmetric_contraction=True, reduction="auto")
```

## 当前边界与验证

### 高角动量的无 Graph 训练

完整 E/F 训练可显式启用分量级目标缓存，减少高 L TP 混合伴随的重复读取：

```python
from chorus.hp import backward_parameters, configure_model

configure_model(
    model, policy="generated", training=True, checkpoint_contraction=True,
    topology_backend="device", training_species_fusion=True,
    radial_coefficient_fusion=True, phase_coefficient_mixed_fusion=True,
    training_destination_first=1, training_mixed_destination=True,
    training_mixed_cache=16, training_mixed_fuse_jvp=True,
    training_terminal_fusion=True,
)
# loss 已包含由 create_graph=True 得到的力损失；不 capture/replay。
backward_parameters(loss, model)
optimizer.step()
```

`backward_parameters` 计算所有可训练参数的完整混合梯度并累加到 `.grad`，仅省略
优化器不消费的输入叶子梯度。它目前限定单设备；需要输入 `.grad` 的调用方应继续使用
普通 backward。TP 系数、径向权重、相位和末端投影保持可训练。

`training_mixed_cache` 可选 8/16/32/64，要求通道数为 32 的倍数；缓存预算不代表
统一更快，应依据模型的配对结果选择。`training_terminal_fusion` 每步重新计算 live
小矩阵乘积，不缓存冻结权重。三阶及以上保留可微原式。

`training_mixed_fuse_jvp=True` 让混合 TP 遍历同时生成输出 cotangent JVP；它需要
`training_mixed_cache`，各路径输出保持唯一写者。`training_mixed_grouped=True` 可进一步
为缓存路径组分别实例化 kernel，避免所有组共用最大寄存器配置。两者均为显式候选，
以完整训练步而非单核计时选择。

容量优先的实验可用 `training_radial_tp_fusion=True, training_radial_cached=True`，
并以 `radial_tile_edges=8192/32768/65536` 控制径向 gate–TP/VJP 缓冲区。缓存径向路径
可以结合 `training_mixed_cache`、`training_mixed_fuse_jvp` 和 `training_mixed_grouped`，
在同一次分块 TP 遍历中生成混合目标与输出 cotangent JVP。
`training_radial_first_bounded=True` 进一步在每块一阶 TP gate VJP 生成后立即消费其径向
隐藏层/bank 投影，避免完整一阶 gate 梯度的峰值存储；它要求启用缓存径向路径。
`training_first_schedule="pruned"/"grouped"/"compact"/"compact_grouped"` 分别选择原式
一阶目标梯度裁切、独立路径组寄存器分配、磁分量立即消费及两者组合。裁切依赖实际
autograd 目标，不裁掉参数梯度所依赖的中间值。默认 `"legacy"` 保留既有调度。
显式选择也适用于普通 backward 和固定源特征，不要求同时设置 destination-first 开关。
`training_first_cache=True` 是另一种已核验但目前较慢的
一阶调度候选，不是推荐速度配置。各开关默认不启用。

大模型正式配对、原式梯度资格和失败记录见
[高角动量训练优化](../benchmarks/hp/TRAINING_HIGH_L_OPTIMIZATION.md)。

当前适配 CUDA FP32、native ICTC angular basis、baseline route、hidden/edge L<=3、
无 interaction attention；训练可保留固定或可学习 TP 基础系数。bridge-U 的 HP 路径覆盖 cor=1/2/3。
其余产品后端保持原实现；模块计数清楚列出实际替换部分，不能把它说成全部 SC 已加速。
通道数和交互层数由原模型决定。分组/分 degree 相位与局部归一化保留原方程。

Persistent 与 Final 的当前推理路线统一为原生算子融合，不使用 CUDA Graph，
也不再把 AOTI 作为默认或按尺寸切换的执行路线。直接调用：

```python
from chorus.hp import configure_model, energy_forces_virial

# The caller loads the original checkpoint, selects CUDA FP32, eval and frozen.
configure_model(model, policy="generated", checkpoint_contraction=True,
                topology_backend="device", capture=False)
out = energy_forces_virial(model, pos, Z, batch, src, dst, shifts, cell)
# out: energy[B], forces[N,3], atomic_energy[N], virial[B,3,3], without E0.
```

原生基准直接从原始 checkpoint 建立完整 E/F/virial 参照，再启用融合并计时，
不需要 `.pt2`、导出或 AOTI 编译。输入坐标、元素、cell、N/E 和拓扑内容可以变化，
CSR 实时重建，不缓存轨迹输出。权重、device 或 dtype 改变后重新配置执行句柄。
短程接口返回 `virial=-dE/dstrain`，不除体积；网络和力为 FP32，virial 为 FP64。
它不覆盖额外 long-range/dispersion 模块。原式参照与核编译冷启动不进入稳态时间。

DPA4 同样禁用 CUDA Graph，保留 compile/Triton level 2，并记录实际激活的 value
内核。当前正式同卡表、参数量差和小体系结果见
[原生无 Graph 推理](../benchmarks/hp/INFERENCE_NATIVE_NO_GRAPH.md)。
旧 AOTI 准备接口、固定 Graph API 和实验结果仅保留为兼容性/历史记录，
不用于当前路线选择；既有 LAMMPS 部署包和正式 MD 工件不变。

远端资格测试：

```bash
python -m pytest -q chorus/test/test_hp_backend.py
python benchmarks/hp/qualify_and_bench.py --scope persistent --channels 64 \
  --lmax 2 --out /TASK/results/persistent.json
```

比较同一权重的原子能量、总能量、力、全部力损失参数梯度；同时检查参数身份、
trainability、checkpoint schema、参考回退和 RNG。不重新训练来掩盖执行差异。
计时是同卡交替配对的完整能量+力路径，排除常量规划/JIT 冷启动，记录 allocated/
reserved 峰值。A100 与 4090 分开报告，不用跨硬件 walltime 推断算子收益。
Mac 不运行这些脚本；A100 使用 `sbatch/GPU-A100/XHPC-A100`，4090 产物留在任务 `lrx`。
