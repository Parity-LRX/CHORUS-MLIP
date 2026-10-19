# 高角动量训练优化记录

远端任务根：
`/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018`。
所有数值任务均由 `sbatch / GPU-A100 / XHPC-A100` 执行。

- `report_cached/`：source_v29 首轮 50 步正反配对，7 个架构/原子数测点。
- `report_selected/`：source_v35 按 scope 选择候选后的完整复测，包含更大原子数。
- `report_latest/`：source_v37 的最终 1024 原子、50 步正反配对。
- `report_candidates/`：全部通过核验的短筛选及速度/显存 Pareto 点。
- `parameter_matched_v1/`：固定同 lmax/深度，按可训练参数量选择的 DPA 通道、配置与扫描记录。
  这里只同步 JSON；固定初始化继续保存在远端，不从本地重新生成。
- `report_first_radial/`：source_v38/v40 的一阶调度与径向分块融合短测。
- `report_parameter_matched/`：source_v40，按 scope 匹配 DPA 参数预算，最新 8 个测点的
  50 步正反配对、同卡旧/新原生调度增益与独立原式资格。显存 GB = 10^9 字节。
- `report_parameter_matched_previous/`：前轮 12 测点；C128/C192 尚未选择第四卡的新一阶调度。
- `raw/highl_cached_pair_v2/`：首轮计时、显存、初始化/参数身份、完整命令与子进程记录。
- `raw/first_cached_*`：分量缓存一阶伴随和径向融合的原式核验、短测与独立 profiler。
- `raw/wide_cached_*`：32/64 分量路径缓存组的核验和短测。
- `raw/radial_tiles_*`：8k/32k/64k 边块的实时径向 gate–TP/VJP 融合。
- `raw/fused_jvp_*`：单遍混合 TP/输出 cotangent JVP 的核验和短筛选。
- `raw/group_pilot_*`：最大模型的 CG 路径组专门化，含实时一阶伴随候选。
- `raw/first_fusion_*`：一阶目标裁切、路径组专门化、磁分量即时消费及其组合。
- `raw/radial_first_*`：分块一阶 gate VJP 与实时径向投影，结合单遍混合 JVP。
- `raw/highl_selected_pair_v3/`、`raw/highl_latest_pair_v4/`：后两轮完整配对及原式核验。
- `raw/highl_parameter_matched_v5/`：近参数预算的正式计时、完整命令和 PID/Slurm/设备身份。
- `raw/highl_parameter_matched_v6/`：第四卡候选的 C128/C192、1024 原子正式配对。
- `raw/first_scale_*`：第四张 A100 上 C128/C192 一阶调度的资格与补充短测。
- `raw/parameter_matched_oracle_v1/`、`raw/parameter_matched_oracle_v2/`、
  `raw/parameter_matched_oracle_v3/`：最大模型的独立原式核验；第一轮 Final/512 通过，
  Persistent/512 在边分块、节点 SC 分块、节点＋通道 SC 分块三次参考尝试中均 OOM。
- `raw/bounded_oracle_tests_92400.xml`：A100 的原式分块/报告协议 11 项通过。
- `raw/reference_sc_unit_4090.xml`：相同 6 项分块数值单测在空闲 4090 上通过，
  与 A100 性能数字严格分开；远端根 `/home/ylzhang/lrx/chorus_reference_sc_validation_20261018`。
- `raw/large_oracle_v1/`：C128、1024 原子三批原式核验。仅未计时参考支路分块和 CPU 暂存。
- `raw/*regression*.xml`：各冻结源码版本的回归结果。
- `raw/radial_cached_qual_v4/`、`raw/radial_cached_qual_v5/` 与对应 XML：失败记录，未进入性能结论。
- `logs/`：命令、PID、Slurm、设备、异常及每次测量的原始日志。

大模型使用既有固定构造器初始化，保留其 U、全参数和可训练 TP 系数；并非等精度模型配对。
DPA 使用实际可微 compile 核心；Triton inference level 2 已配置，但上游训练分支不执行
该推理 value 核。两方使用相同的参数梯度目标、三批变化输入、E/F 损失和 AdamW。
不使用 CUDA Graph、AMP、TF32 或外部 TP 算子库。

短筛选和 profiler 与正式 50 步配对分开。未通过三批原式核验的候选不参与选择，
128 原子核验不标作 1024/4096 原子核验。报告保留 OOM 和失败，不用 OOM 推导速度比。
巨大 Chrome trace 未复制到本地；原文件继续保存在远端原结果目录。
