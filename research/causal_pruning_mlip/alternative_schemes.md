# CHORUS-MLIP 无蒸馏压缩与加速：替代方案全文补充检索

检索日期：2026-08-23  
范围：不限年份，重点覆盖 2023-2026。核心条目均核查主论文全文或开放期刊全文；“无蒸馏”指不使用教师 logits、隐藏状态或教师生成标签，原始能量/力/应力标签的微调仍单独计入恢复预算。

## 结论先行

还有很多方案，但它们解决的不是同一个问题。对当前 CHORUS，合理顺序是：

1. **先做精确算子与编译后端矩阵**，确认张量积、symmetric contraction、scatter、布局转换、能量-力自动微分中谁是真瓶颈。该步骤不改变函数、不需要蒸馏，也不需要重新训练。
2. **再做 kernel-aligned 的结构化闭包裁剪**：整组 irrep multiplicity、完整张量积 instruction/path、通道块、消息层、相关阶，而非单个权重或单个 m 分量。因果评分只是候选选择器，真正加速来自形状变小并重新编译。
3. **若张量积仍占主导，再研究代数替换**：CP 分解、Gaunt/Fourier、SO(3)→SO(2)、edge-to-node。它们可能比 rank 裁剪带来更大的算法级收益，但有的只近似等变，有的等价但需要重写架构。
4. **动态或空间自适应只适合有局部难区的模拟**。它不是通用模型压缩，最坏情况下所有原子都进入高精度分支，收益消失。
5. **低精度、直接力、MoE 和通用非结构化稀疏不能先验视为优解**：低精度会放大导数噪声；直接力破坏能量守恒；稀疏 MoE 常减少激活计算但增加总参数；非结构化稀疏在未达到硬件阈值时可能更慢。

当前仓库的证据进一步支持这个排序：MACE-CHORUS-Final R32→R8 参数减少约 16%，2048 原子能量-保守力吞吐只提高约 4%，推理峰值内存只减少约 3%。这不是“低秩无效”的普遍定理，但足以否定“参数越少，当前 A100 上就会近似同比加速”的前提。

## 方案地图

| 方案 | 改变模型函数？ | 对称性状态 | 无蒸馏可行？ | 真实速度证据 | 对 CHORUS 的判断 |
|---|---:|---|---:|---|---|
| FlashTP / OpenEquivariance / 当前 cuEquivariance 对照 | 否或仅数值舍入差 | 精确等变 kernel | 是，无需训练 | 强；TP kernel、端到端训练和 MD | **第一优先**；但性能强依赖版本、张量布局、l_max 和体系大小 |
| 图编译、算子融合、custom derivative | 否或仅数值顺序变化 | 由原算子保持 | 是，无需训练 | 强；NequIP/Allegro 报告最高 18 倍 MD 加速 | **第一优先**；必须包含保守力和应力路径计时 |
| 消息传递深度截断 | 是 | 仍可严格等变 | 是；原标签微调 | 强；MPDP 有 A100/大体系证据 | **强基线**；会缩短有效感受野 |
| irrep/channel/path/block 闭包裁剪 | 是 | 满足完整 m、parity、TP 与下游依赖闭包时可保持 | 是 | 相邻证据中等，直接 O(3)-MLIP 部署证据不足 | **主要研究方向**；必须物理压紧并重新编译 |
| l_max、correlation/body order、radial basis 联合缩减 | 是 | 架构内可保持 | 是；从头训练或原标签恢复 | MLIP 实践证据较强 | **强基线/联合搜索变量**；会改变表达能力，不只是删除冗余 |
| CP 分解 CG tensor | 是 | 通常近似等变 | 是；从头训练 | TDN 在等变模型上有 2.47-8.44 倍吞吐证据 | **高潜力、高风险**；必须测逐层等变误差与长时 MD |
| Gaunt/Fourier tensor product | 代数重参数化 | 论文给出等价表示；实现仍需数值验证 | 是；从头训练 | MACE-Gaunt 有速度/内存证据 | **高潜力架构分支**；不是后训练裁剪 |
| SO(3)→SO(2) 边对齐卷积 | 是 | 理论等变；离散非线性可能近似 | 是；从头训练 | eSCN 有大规模催化数据证据 | **架构对照**；不是 CHORUS 原权重直接压缩 |
| edge-to-node 等变计算 | 是 | 可设计为严格等变 | 是；从头训练 | NEMP 声称显存/计算下降 1-2 个数量级，证据仍新 | **值得关注**；核心是把昂贵计算从 O(E) 移至 O(N) |
| scalar-vector / Cartesian / local-frame 轻量架构 | 是 | 取决于构造；局部 frame 需检验退化点连续性 | 是；从头训练 | E²GNN、CACE、AlphaNet 有 MLIP/MD 证据 | **匹配小模型基线**；不能算“裁剪 CHORUS” |
| 空间自适应高低保真势 | 是，且包含两个势 | 平滑混合时可保持力连续性；需逐案验证 | 是；两个势可用同一原始标签独立拟合 | 铜纳米压痕 400 万原子、11.3 倍 | **场景型方案**；只适合难区局域化问题 |
| element-conditioned 静态容量/路由 | 是 | 原子种类是不变量；等变专家和闭包仍需保证 | 是；从头训练或原标签恢复 | 直接速度证据不足 | **较新颖**；优先静态 species/species-pair bucket，而非逐环境硬路由 |
| 稀疏 MoE | 是 | invariant gate + equivariant expert 可保持 | 是 | MLIP 中主要是容量/稳定性证据，缺少墙钟收益 | **不应称作压缩首选**；总参数和内存可能上升 |
| 混合精度 | 数值近似 | 结构等变仍在，数值等变和力导数可漂移 | 是，无需教师 | 有加速但结论依硬件/算子而变 | **工程实验**；TP/累加/导数不宜直接全半精度 |
| 图/边/三元组推理时裁剪 | 是，且可能不光滑 | 旋转不变的 gate 不保证力连续 | 是 | 直接 MLIP 证据很初步 | **高风险探索**；必须用平滑 gate、hysteresis 和长时 MD |
| graph-parallel / zero-redundancy partition | 否 | 不改变单模型对称性 | 是，无需训练 | 多 GPU 弱扩展证据 | **规模扩展，不是单卡压缩**；与消息深度、halo 通信联合优化 |
| 直接力输出头 | 是，目标也改变 | 可保持向量等变，但通常不保守 | 是 | 推理显著更快 | **不建议作为默认替换**；curl、能量漂移和应力一致性是硬门槛 |

## A. 先做：不改变模型语义的精确加速

### A1. FlashTP、OpenEquivariance 与当前 cuEquivariance 必须按版本重测

[FlashTP（ICML 2025）](https://proceedings.mlr.press/v267/lee25l.html)通过稀疏 CG 存储、path aggregation 和 kernel fusion 避免中间 outer-product 物化。论文在 A100 上报告 FP32 TP kernel 相对 e3nn 平均约 6.75 倍 forward、14.37 倍 backward、26.54 倍 double-backward；SevenNet-l3i5 在 4K 原子上端到端 MD 相对 e3nn 约 4.2 倍，并显著降低峰值内存。

但论文对 cuEquivariance 使用的是 0.2.0，不能把其中“相对 cuEq 6.2 倍”当作对当前 cuEq 的结论。[OpenEquivariance（SIAM ACDA 2025）](https://arxiv.org/abs/2501.13986)的全文恰好给出反例：它在 NequIP 中相对未修改实现达到 7.8 倍，但 MACE 上其实现只有最新 fused cuEq 的 0.73 倍；将其 fused convolution 与 cuEq 其他算子组合的 hybrid 才更接近最优。因此后端优劣不是全局排名，而是模型、版本、l_max、通道、图规模、数据布局和导数阶数的函数。

CHORUS 当前 MACE 路径已经是 MakeFX/Inductor + cuEq product + cuEq convolution，NequIP 路径已经使用 OpenEquivariance 0.6.8 AOTI。下一步不是拿 e3nn 旧基线重新证明融合核很快，而是对当前真实栈做：

- `cuEq-current`、`OpenEq-0.6.8/current`、`FlashTP`（若 instruction/layout 可兼容）的同语义矩阵；
- 128/512/2048/4096 原子以及实际邻居数分布；
- energy-only、energy+conservative-force、energy+force+stress、训练 double-backward；
- 编译时间、steady-state、峰值内存、数值偏差和 break-even MD 步数。

### A2. 编译、图融合与导数路径可能比裁剪收益更大

[High-performance training and inference for deep equivariant interatomic potentials](https://arxiv.org/abs/2504.16068)将 AOT 编译、Triton tensor-product kernel、图级并行和邻居处理组合，报告实际尺度 MD 最高约 18 倍加速。[2026 年 NequIP/Allegro 大规模研究](https://arxiv.org/abs/2607.28461)也表明图编译、等变专用 kernel 与谨慎混合精度可带来累计 5-10 倍训练加速；同时它提醒参数量只是不可靠的速度代理。

这类结果对论文叙事有一个不利含义：若 CHORUS 的主要收益来自换后端，贡献是系统/内核工程，不是因果压缩。相反，它也提供了必要前提：只有先锁定高效 backend，之后测出的裁剪增益才不是相对低效基线的幻觉。

## B. 最适合“因果 + 压缩”的方案：可编译闭包组选择

### B1. 处理单元应对应真实 kernel shape

建议的候选组按优先级为：

1. 完整 TP instruction/path bundle，包括 parity、全部 m 分量和下游输入维度；
2. irrep multiplicity block 或固定大小 channel tile；
3. 整个 interaction block / 后缀消息层；
4. correlation/body-order 分支和 radial basis group；
5. Hermitian/共轭配对、Final/Persistent 路径和残差闭包；
6. species 或 species-pair 的静态路径组。

这比“逐个 rank 分量打分”更难，但才可能同时满足严格对称、结构抽取和硬件加速。2026 年[表示论对称裁剪](https://www.mdpi.com/2073-8994/18/6/983)证明了有限群 G-CNN 中 irrep-bundle/orbit-wise closure 的必要性，并显示普通逐通道 L1 裁剪会严重破坏等变性；它不能直接证明 O(3) MLIP 的 CHORUS 闭包，只能作为设计原则和反例。

### B2. 因果评分要和联合选择、真实成本绑定

候选组的 knockout/机制替换效应应同时输出能量、力、应力、最坏环境损伤与实测毫秒变化。由于 TP path、rank 和消息层存在替代/协同关系，单组平均处理效应不足以决定联合子集。建议流程是：

- 随机分数因子 group masks，估计主效应和少量二阶交互；
- 用实测 latency/memory 作为成本，而非参数量或理论 FLOPs；
- 在 dependency closure 下做组合选择；
- 每次物理抽取后重新编译、重新估计；
- 与随机、幅值、Taylor/曲率、直接小模型和 MPDP 深度截断做相同恢复预算比较。

这个方向的论文新意不应表述为“因果评分第一次用于剪枝”，因为 C-SWAP、CMR 及 TP path pruning 已有相邻先例。更可辩护的贡献是：**面向等变/Hermitian 闭包和真实 kernel 成本的联合计算机制选择，并在保守力 MD 上验证。**

## C. 代数级替换：收益上限高于 rank-only，但风险更大

### C1. CP 分解 Clebsch-Gordan 张量

[Tensor Decomposition Networks（NeurIPS 2025）](https://arxiv.org/abs/2507.01131)对 CG tensor 做 CP 分解，并共享路径权重，将理论复杂度从约 L 的六次降为四次。全文 Table 4 中，相对 e3nn Equiformer 的吞吐分别为 L=1 的 2.47 倍、L=2 的 4.34 倍、L=3 的 8.44 倍，参数减少约 63%-92%；在 NequIP 变体中训练时间也由约 7.5 降到 2.0 min/epoch。

局限同样明确：CP 截断一般产生非零等变误差，rank 选择依赖经验；主要任务是 OC20/OC22 与分子性质，并未充分证明长时保守力 MD。它与 CHORUS 的“密度 rank”不是同一个秩，至少要分开报告 CG-operator rank、feature-density rank、channel-mixing rank 和 path tying。

### C2. Gaunt/Fourier 与 SO(3)→SO(2)

[Gaunt Tensor Products（ICLR 2024 Spotlight）](https://arxiv.org/abs/2401.10216)把 CG 计算改写到球函数/二维 Fourier 表示，理论复杂度从约 L 的六次降为三次。全文在 3BPA 的 MACE-Gaunt 设置中报告相对 e3nn 43.7 倍的操作级速度指标和 5.8% 的内存成本，同时保持与 MACE 接近的能量/力误差；但这个数字不能直接等价为 CHORUS 端到端能量-力吞吐。

[eSCN（ICML 2023）](https://arxiv.org/abs/2302.03655)通过边对齐把 SO(3) 卷积降为 SO(2) 操作，同样将高角动量计算从约 L 的六次降到三次。这两类方法是架构/代数替换，不是从现有 CHORUS checkpoint 无损抽取；更适合作为“如果 TP 是瓶颈，何种替代算子胜过剪枝”的研究分支。

### C3. edge-to-node 与轻量表示

[NEMP](https://arxiv.org/abs/2508.16086)把昂贵等变操作从 edge space 移到中心节点与聚合虚拟节点之间，目标是把主要成本从边数 E 的尺度降到节点数 N 的尺度；论文声称相对 edge-equivariant 模型降低 1-2 个数量级的显存/计算。该结论很新，尚不应视为成熟工程事实，但它指出了比“删少量通道”更根本的方向：改变昂贵计算发生的位置。

另一条路线直接限制表示阶数。[E²GNN（npj Comput. Mater. 2025）](https://www.nature.com/articles/s41524-025-01535-3)只用 scalar/vector，并在节点侧更新；[CACE（npj Comput. Mater. 2024）](https://www.nature.com/articles/s41524-024-01332-4)用 Cartesian ACE 式矩避免高维 CG；[AlphaNet（npj Comput. Mater. 2025）](https://www.nature.com/articles/s41524-025-01817-w)用局部 frame scalarization 避免显式高阶 tensor product。它们是“直接训练高效小架构”的强对照，不是对 CHORUS 的后训练压缩。局部 frame 还必须检查对称或近退化几何中的不唯一性和力连续性。

## D. 条件计算与空间自适应：有条件地值得做

### D1. species-conditioned 静态容量比逐环境硬路由更稳妥

不同元素对角向/径向分辨率的需求并不相同。可以令原子种类或 species-pair 选择预编译的 l_max、rank、path 集合。原子种类是离散不变量且在 MD 中不随位置变化，因此比依据瞬时局域环境做 hard routing 更容易保持力的平滑性和固定 kernel shape。

但这仍不是自动成立的：expert 必须是等变映射，gate 必须是 invariant scalar，species 间的信息流仍要满足 TP 与 Hermitian 闭包。若每个样本的动态形状不同，分支和 padding 可能吃掉理论收益。可先做 2-4 个静态 bucket，而不是每个原子一套动态网络。

### D2. 空间自适应保真度只适合难区局域化

[Adaptive-precision potentials（JCP 2025）](https://arxiv.org/abs/2411.03002)用局域结构检测器在 EAM 与 ACE 间平滑切换，并做负载均衡；在 400 万原子、100 ps 的铜纳米压痕中相对全 ACE 报告 11.3 倍 CPU 加速。它不需要教师蒸馏，但需要两个已拟合势以及能量/力匹配和切换校准。

该方法的适用前提很强：高精度区域必须只占小部分原子。均匀复杂的液体、非晶、反应体系或高缺陷密度下，高精度比例可能接近 100%，收益趋零。若使用同一 DFT 原始数据分别训练小/大 CHORUS，它满足“无蒸馏”；若用大模型生成小模型标签，则不满足。

### D3. MoE 不是天然压缩

最新 MLIP MoE 工作显示 invariant routing + shared equivariant experts 可以稳定工作，但某些全局 nonlinear MoE 出现数值不稳定，而且激活少数专家不等于总参数减少。若没有真实 wall-clock 和显存收益，就只能称为条件容量分配，不能称为模型压缩。

## E. 低精度、图裁剪和直接力：放在高风险实验区

### E1. 混合精度

[Low-precision MACE](https://arxiv.org/abs/2510.23621)说明 linear-only BF16/FP16 与 cuEq 可能叠加加速，但半精度训练会损害力误差。2026 年大规模 NequIP/Allegro 研究还发现低精度推理噪声会影响模拟收敛，因此其正式推理选择保留更高精度。更合理的策略是 representation-aware mixed precision：线性 mixing 用 BF16/FP16，TP 累加、能量归约、坐标导数和关键 residual 保留 FP32，之后测 NVE 能量漂移和数值等变误差。当前没有足够直接证据支持优先做 INT8 保守力 MLIP。

### E2. neighbor/edge/triplet 裁剪

图结构变化会直接改变能量对坐标的分段形式。即使 gate 对旋转不变，top-k 或硬阈值越界仍可能产生力不连续。2026 年 M3GNet 三元组推理时删除结果只来自短篇初步证据和短轨迹，尚不能证明长期稳定或真实 GPU 加速。若要做，应使用 smooth gate、cutoff envelope、hysteresis/Verlet reuse，并报告闭合路径功、力连续性、Hessian/声子和长时 MD，而不是只看静态 MAE/RDF。

### E3. 直接力输出

直接力可以绕过能量对坐标的反向自动微分。[E²GNN 全文](https://www.nature.com/articles/s41524-025-01535-3)也显示直接力模型显著更快，但作者明确承认其力不是能量守恒的，并需要 thermostat 调温。除非目标应用容忍非保守力，否则这不是“压缩现有 CHORUS”的等价替代。至少要报告力场 curl、NVE 漂移、应力一致性和 Newton 第三定律误差。

## 推荐给 CHORUS 的三阶段实验

### 阶段 0：先判定瓶颈，不训练

- 在 Final/Persistent R8/R16/R32 上分解 TP、symmetric contraction、conv-TP、scatter/gather、radial MLP、graph build、energy→force backward 和 stress。
- 记录不同原子数、实际邻居数、l_max、correlation、rank 的 kernel shape、时间和内存。
- 对每个结构只关一个机制，确认当前 rank 的 4% 吞吐差来自何处。

**停止条件**：若 CHORUS 特有 rank/Hermitian 路径不到端到端时间的 15%，继续精细 rank 因果排序大概率不经济，应先改 backend 或更大的结构轴。

### 阶段 1：精确后端与强结构基线

- 当前 cuEq、OpenEq/current、FlashTP-compatible、Inductor/AOT/custom derivative；
- depth 1/2/3、l_max 1/2/3、correlation 2/3、通道、R4/R8/R16/R32；
- 同预算原标签恢复和直接训练小模型；
- energy+force+stress 的端到端 Pareto，而非只报参数/FLOPs。

### 阶段 2：闭包组的因果/联合选择

- group treatment = TP path/irrep tile/Hermitian closure，而非标量；
- random factorial masks 估计主效应和少量交互；
- cost-aware subset selection，物理抽取后重编译；
- 随机、幅值、Taylor/曲率、MPDP、匹配小模型负对照；
- T1x/MS25/缺陷/高温/OOD 分层和长时 MD。

### 阶段 3：高收益架构分支

- 若 TP 占主导：CP-CG 与 Gaunt/SO(2)；
- 若 edge tensor product 占主导：NEMP 式 edge-to-node；
- 若元素复杂度差异显著：species-conditioned static buckets；
- 若复杂区域空间局域：同原始标签训练的高/低保真空间混合。

## 最终优先级

1. **后端/导数路径 profiling + 当前版本公平矩阵**：最可能立即得到真实速度，也能防止后续研究建立在错误成本代理上。
2. **TP path / irrep / Hermitian 闭包的硬件成本感知联合裁剪**：最符合“因果推断 + CHORUS + 无蒸馏”的可发表主线。
3. **MPDP 深度、l_max/correlation/channel、直接小模型**：必须有的强基线，可能直接击败复杂因果评分。
4. **CP/Gaunt/SO(2)/edge-to-node**：若 profiling 证实 TP/edge-space 是主瓶颈，优先级上升。
5. **species-conditioned static capacity**：有新颖性，但要先证明元素条件异质性和真实 kernel 可执行性。
6. **空间自适应精度**：只在缺陷、裂纹、界面等局域难区任务中优先。
7. **混合精度、图动态裁剪、MoE、直接力**：作为风险受控的副线，不应先于上述方案。

## 证据边界

- 各论文的倍数来自不同 GPU/CPU、模型、原子数、精度、邻居表边界和软件版本，不能横向直接相乘或排名。
- kernel 倍数不等于能量-保守力端到端倍数；训练还包含 double-backward，应单独计时。
- “等变架构”不等于数值上严格等变；近似分解、低精度、局部 frame 退化和抽取后的普通层都可能引入误差。
- “无蒸馏”不等于“零训练”；原标签微调、线性 refit、从头训练和两个独立势的成本必须分别报告。
- 2025-2026 的 NEMP、低精度、MoE、triplet pruning 等证据较新，部分仍为预印本或短论文，不能与成熟期刊/会议结果同等对待。
