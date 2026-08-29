# 因果干预驱动的网络裁剪与 MLIP 加速：全文检索初报

检索日期：2026-08-22  
范围：不限年份，重点覆盖 2021-2026；只把已打开并核查主论文全文的方法列为核心证据。正文所称“全文核查”不自动包含所有补充材料。

2026-08-23 的非蒸馏替代路线与算子级加速补充见 [alternative_schemes.md](./alternative_schemes.md)。

## 结论先行

这个方向可以做，但不能把“做过消融”直接等同于“完成因果推断”，也不能把“参数变少”直接等同于“推理加速”。目前最合理的判断是：

1. 直接使用组件干预进行裁剪的论文数量不多，主要集中在图像分类和 LLM；它们几乎没有覆盖 E(3)/O(3) 等变 MLIP。
2. 最强的相邻 MLIP 证据并不依赖因果推断：消息传递深度截断和成本感知基函数裁剪已经给出真实 MD step time 或 CPU 加速。因此，因果方法必须证明它优于这些简单而强的基线。
3. 不做蒸馏完全可行，但必须区分零恢复、精确机制折叠、原始能量/力/应力标签微调、线性系数重拟合和从头训练小模型。这些方案的信息预算与计算预算不同。
4. 对 CHORUS，逐个标量权重或单个磁量子数分量的裁剪不是合格方案。干预单元必须满足完整 irrep、多重度、张量积路径以及 Hermitian/共轭依赖闭包，并最终物理重建为更小的稠密算子。
5. CHORUS 当前的 rank 扫描已经给出一个不利但重要的信号：R8、R16、R32 的 A100 吞吐差异远小于参数差异。仅压低 rank 很可能只减少少量参数，无法解决主算子、邻居聚合或自动微分瓶颈。

## 核心全文证据

| 工作 | 因果/裁剪语义 | 无蒸馏状态 | 是否证明真实加速 | 对 CHORUS 的价值与主要局限 |
|---|---|---|---|---|
| [Ferianc et al., *On Causal Inference for Data-free Structured Pruning*](https://arxiv.org/abs/2112.10229)（2022 workshop） | 对输入施加高斯干预，再用相邻激活互信息给神经元评分；不是对待剪组件的平均处理效应 | 未使用蒸馏，实验基本为零恢复 | 否；小型全连接网络，没有墙钟延迟 | 可作“弱因果”基线。高斯输入会偏离分子结构流形，深层独立性假设也被作者承认会失效 |
| [Ahmad et al., *Causal Analysis for Robust Interpretability of Neural Networks*](https://arxiv.org/abs/2305.08950)（WACV 2024） | 切断节点输出路径，检验对预测的影响，生成因果解释图 | 不是压缩训练方案 | 否；mask/repair 不等于抽取小模型 | 是 C-SWAP 的方法前身，但主要证明解释与修复，不证明参数缩减或部署加速 |
| [Bauvin et al., *C-SWAP*](https://bmvc2025.bmva.org/proceedings/850/)（BMVC 2025） | 对通道/神经元做路径切断干预，用配对样本和显著性检验判断关键性，再逐层结构化删除 | 不微调、不蒸馏 | 主要报告参数/FLOPs；缺少端到端延迟 | 与“无需蒸馏的组件级干预”最接近，但只验证分类/分割；逐组件评估昂贵，遗漏联合交互和选择后多重检验 |
| [Asiaee, *Causal Mechanism Reduction*](https://arxiv.org/abs/2602.24266)（2026 preprint） | 将固定前馈网视作确定性计算 SCM，用常数或仿射机制替换单元并折叠到相邻权重，形成更小稠密网 | 可精确折叠；实验也使用匹配的原标签微调 | 报告 DeiT 吞吐，但不同评分间差异很小 | 概念上最严谨，强调重参数化不变性和交互干预验证；但因果性只属于模型内部机制，非对角曲率很大，随机裁剪经恢复后也接近最优 |
| [Sheth et al., *Pruning via Causal Attribution Preserves Reasoning Performance*](https://arxiv.org/abs/2606.19350)（ICLR 2026 workshop） | knockout 注意力头，以损失增量给头打分，再把分数下放到标量权重做全局稀疏 | 不微调、不蒸馏 | 否；输出是非结构化稀疏 | 中等稀疏度下部分任务优于 Wanda，但 50% 时可崩溃；头级因果分数向 MLP/标量权重传播缺乏严格依据，对加速价值有限 |
| [Li et al., *Revisiting Random Channel Pruning*](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Revisiting_Random_Channel_Pruning_for_Neural_Network_Compression_CVPR_2022_paper.html)（CVPR 2022） | 非因果负对照：随机层宽/通道配置加简单评分 | 部分流程含原标签训练；不是教师蒸馏必需条件 | 关注结构化压缩，部署证据仍弱于 MLIP 工作 | 关键反例：复杂重要性评分常与 L1/L2 或随机选择接近，恢复时长和最终架构常比“剪哪个通道”更重要 |
| [Kong et al., *Scalable Foundation Interatomic Potentials via Message-Passing Pruning and Graph Partitioning*](https://www.nature.com/articles/s41524-026-02001-4)（npj Comput. Mater. 2026） | 非因果：依据层间特征变化/CKA 的相关性诊断，直接保留前 K 个消息传递层 | 使用原始能量/力/应力标签微调，不使用教师 | 是；报告 A100 MD step time、大体系和物理观测量 | 目前最强的直接 MLIP 基线。只截断后层、搜索空间简单，但已经说明“简单深度裁剪 + 原标签恢复”能产生真实收益 |
| [Meng et al., *Accelerating Moment Tensor Potentials through Post-Training Pruning*](https://arxiv.org/abs/2510.19737)（2025 preprint） | 非因果：NSGA-II 在精度代理和 compute-tree 成本之间选择基函数子集，重新拟合线性系数 | 无新数据、无教师；需要系数重拟合/再训练 | 是；单核 AMD EPYC 上报告约 3.8-7.0 倍加速 | 强调参数数目不是成本代理：较多参数的结构也可能更快。对象是 MTP 基函数，不是等变神经网络；部分 MD 稳定性依赖初始化 |
| [Vorotnikov et al., *Low-rank Matrix and Tensor Approximations for Compression of MLIPs*](https://arxiv.org/abs/2509.04440)（JCP 2025） | 非因果：固定秩矩阵/张量分解与 rank augmentation | 不依赖教师 | 证明最高约 50% 参数压缩，但没有充分墙钟加速证据 | 与 CHORUS rank 压缩相邻；提醒“低秩可压缩”不等于“现有 GPU kernel 会更快” |

## “因果”强度不能混为一谈

按可辩护程度，现有工作至少分成四层：

1. **模型组件的随机/显式干预**：C-SWAP、Ahmad、CAP。它们回答“把这个计算组件切断会怎样”，但通常只估计单组件边际效应。
2. **确定性计算图 SCM 与机制替换**：CMR。它有明确的低层/高层机制映射和可编译替换，但不能外推为现实物理系统中的因果关系。
3. **对输入分布做干预、再用相关量评分**：Ferianc。它使用因果术语，但互信息仍是干预分布下的关联，不能识别“删除该神经元”的效应。
4. **纯相关性/结构诊断**：MPDP 的 CKA、过平滑分析和一般幅值/曲率评分。它们不是因果推断，但可能在工程上更有效。

因此，论文若采用这个方向，宜使用“**computational causal intervention / model-internal mechanism effect**”而不是“发现分子物理因果机制”。

## 对 CHORUS 的直接推论

### 1. rank-only 很可能不是主加速旋钮

仓库当前 A100 表中，CHORUS-Final 从 R32 降至 R8，参数由 888,454 降至 744,070，约减少 16%；但 2048 原子推理吞吐只从 25,945 提高到 26,964 atoms/s，约提高 4%。R8 与 R16 的吞吐差异也只有约 3%。这说明在当前 backend 与 workload 下，rank 参数并非主要耗时来源，或者较小 rank 没有触发更高效的 kernel。

所以先做“R16 中挑 8 个最重要分量”可能获得精度研究价值，却未必获得可发表的加速结果。应先做算子级 profiling，并让裁剪对象对齐实际 kernel 维度。

### 2. 合格的处理单元必须保持表示论闭包

建议的最小干预单元不是单个标量，而是：

- 完整 irrep multiplicity block，保留该表示的全部 m 分量；
- Hermitian/共轭配对和跨邻居项的闭包；
- 张量积 selection rule 对应的整条路径及其下游输入维度；
- Final/Persistent 路径、残差和归一化参数的同步重建。

只做 mask 而不物理抽取，既不能保证加速，也可能在抽取或微调阶段破坏等变性。

### 3. 估计量应是多目标、条件化的计算效应

单个平均验证损失不够。每个组干预至少要记录：能量误差变化、力误差变化、应力误差变化、数值等变误差、实测延迟/内存变化，以及按元素、相态、温压、密度、缺陷和稀有环境分层的最坏组效应。最终还需通过较长 MD rollout、能量漂移、RDF/ADF、扩散或任务相关物性验证。

### 4. 单组件排序不足以处理交互

CHORUS 的低秩分量、张量积路径和消息传递层会有协同或替代关系。CMR 已观察到很大的非对角曲率；C-SWAP 的逐组件路径切断也没有解决联合选择。较稳妥的设计是先用随机 group mask 的部分因子实验估计主效应和二阶交互，再做带硬件成本的组合选择，并在每轮结构删除后重新估计。

## 无蒸馏时必须设置的公平基线

| 基线 | 目的 |
|---|---|
| 直接训练 R4/R8/R12/R16、较低通道、较低 l_max 或更浅模型 | 检验“先训练大模型再裁剪”是否真的优于直接小模型 |
| 等规模随机 group/rank/irrep/path 裁剪 | 检验因果评分是否优于架构随机性 |
| 幅值、激活方差、Taylor/曲率、CMR 式重参数化不变评分 | 排除简单重要性指标已足够的解释 |
| MPDP 式消息传递深度截断 | 对齐当前最强直接 MLIP 裁剪先例 |
| 成本感知组合搜索 | 检验真实加速是否来自成本模型，而不是因果打分 |
| 仅换 backend/精度/融合 kernel | 判断剪枝相对 cuEquivariance/OpenEquivariance/混合精度还有多少增量 |

每个基线都要分别报告：裁剪后零恢复结果、相同原标签与相同 epoch 的微调结果、从头训练匹配小模型结果。禁止教师 logits/隐藏状态不代表可以忽略恢复预算。

## 可形成论文的问题，而不是仅做一个技巧

弱版本是“逐个置零 CHORUS rank，按能量/力损失排序，再微调”；它与 C-SWAP/CAP 的基本思路重复，也没有解决等变闭包、交互和真实加速。

更有研究价值的版本是：**等变与 Hermitian 闭包约束下的联合计算机制干预 + 硬件成本感知结构抽取**。其最低证据门槛包括：

1. 预注册的组干预定义和估计量；
2. 主效应与关键二阶交互，而非只做单组件排名；
3. 抽取后的稠密小模型与逐阶段等变误差测试；
4. 与随机、简单评分、直接小模型、MPDP、backend 优化的匹配比较；
5. A100/目标 CPU 上端到端能量-力吞吐、内存和 break-even 分析；
6. 静态精度之外的 MD/物性安全性验证。

在这些条件下，因果推断不是包装词，而是用于回答一个清晰问题：在给定结构分布和部署成本下，移除哪个**闭合计算机制集合**会以最小物理风险换取可兑现的运行时间收益。

## 第二层全文候选

后续若继续深入，应优先精读以下方法，并把它们用于实验设计而不是扩大引用数量：

- [OSSCAR: One-Shot Structured Pruning with Combinatorial Optimization](https://proceedings.mlr.press/v235/meng24a.html)：联合子集选择，针对单组件效应失效。
- [DepGraph: Towards Any Structural Pruning](https://openaccess.thecvf.com/content/CVPR2023/papers/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.pdf)：结构依赖分组与物理抽取。
- [No Free Prune](https://proceedings.mlr.press/v235/kumar24a.html)：解释无数据/初始化时剪枝为何缺少任务信息。
- [Pruning for GNNs](https://proceedings.mlr.press/v267/ma25e.html)：图模型剪枝后的表达能力边界。
- [A Unified Lottery Ticket Hypothesis for GNNs](https://proceedings.mlr.press/v139/chen21p.html)：联合图结构与权重稀疏；在 MLIP 中需额外处理坐标 cutoff 和力连续性。
- [Symmetry-Preserving Pruning of Group Equivariant CNNs](https://mdpi-res.com/d_attachment/symmetry/symmetry-18-00983/article_deploy/symmetry-18-00983-v2.pdf)：有限群等变 CNN 的表示论闭包；不能未经证明直接外推到 O(3) 张量积 MLIP。

## 当前证据边界

- 直接因果裁剪在等变 MLIP 上仍是明显空白，但“没有搜到”不是不存在的证明；后续应做引文追踪与数据库复核。
- 2025-2026 的几篇关键工作仍是预印本或 workshop，不能按成熟定论引用。
- 当前初报核查了主论文全文和关键表格/局限，尚未系统核查每篇补充材料、代码可复现性和所有引用链。
- CHORUS 吞吐表处于当前工作树状态；在将数值写入论文前应锁定 commit、复跑并统一 README 与结果表的内存数字。
