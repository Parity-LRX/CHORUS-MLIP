# A100 无 CUDA Graph 推理记录

远端任务根目录：
`/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018`。
所有数值计算、导出、编译、计时和汇总均通过 `sbatch` 在
`GPU-A100 / XHPC-A100` 执行。本地仅保存源码、日志及已生成报告。

当前 Persistent / Final 统一使用原生融合，不再继续 AOTI 导出或调优。
目录名保留旧实验身份；[当前正式表](native_report/INFERENCE_NATIVE_NO_GRAPH.md)
只比较原生融合与 DPA，不按尺寸选择 AOTI/原生的最小值。

## 正式配对

正式表中的原生融合与 DPA4 compile 来自同一作业、同一张 A100 的顺序执行。
较早的作业还保存了 AOTI 结果，现仅留档。网络与力为 FP32，TF32/AMP 关闭，virial 为 FP64。
完整输出为总能量、全部力分量、原子能量与 virial。
两侧均有 CUDA Graph API 禁用检查，Inductor cudagraphs 关闭。

| CHORUS 配置 | scope | 参数量 | DPA 通道 | DPA 参数量 | 实际 T2 value |
| --- | --- | ---: | ---: | ---: | --- |
| C128 / lmax 3 / 2 层 | Persistent | 1,938,724 | 48 | 1,752,128 | 不支持此宽度 |
| C128 / lmax 3 / 2 层 | Final | 1,559,448 | 44 | 1,583,031 | 不支持此宽度 |
| C256 / lmax 3 / 4 层 | Persistent | 15,441,468 | 108 | 15,373,021 | 不支持此宽度 |
| C256 / lmax 3 / 4 层 | Final | 12,384,408 | 96 | 12,152,174 | 已启用 |

DPA4 均设置 compile 和 Triton level 2。配置开关不等同于 value kernel 实际启用；
后者在每份 DPA JSON 中单独记录。C32/C64 的实际 T2 对照保存在
[独立复测](supported_repeat/SUPPORTED_T2_REPEAT.md)，其参数量与正式配对不同。
这些模型沿用固定的结构初始化，用于比较执行开销，不是已训练的等精度配对。

## 源码快照与作业

| 作业 | 不可变源码快照 | 内容 | 状态 |
| --- | --- | --- | --- |
| 92441 | source_v7 | C128 Final，atomic AOTI 与原生/DPA | COMPLETED |
| 92442 | source_v8 | C128 Persistent，atomic AOTI 与原生/DPA | COMPLETED |
| 92445 | source_v8 | C256 四层 Persistent，atomic AOTI 与原生/DPA | COMPLETED |
| 92446 | source_v9 | C128 Final，CSR AOTI、atomic 复测与原生/DPA | COMPLETED |
| 92447 | source_v9 | C128 Persistent，CSR AOTI 与原生/DPA | COMPLETED |
| 92448 | source_v10 | C256 四层 Final，分块 TP 伴随与 CSR AOTI、原生/DPA | COMPLETED |
| 92454 | source_v11 | C128 双 scope 独立复测，DPA C32/C64 实际 T2 | COMPLETED |
| 92456 | source_v12 | 完成记录的远端汇总 | COMPLETED |
| 92459 | source_v13 | 最终导出接口的两项动态 N/E 回归 | COMPLETED |
| 92460 | source_v13 | C256 四层 Persistent，CSR AOTI 与原生/DPA | COMPLETED |
| 92461 | source_v14 | 更新四项正式配对的历史汇总 | COMPLETED |
| 92462 | source_v14 | 最大 Persistent 对实际启用 T2 的 DPA C96，同卡原生补测 | COMPLETED |
| 92463 | source_v15 | 历史汇总及 Persistent 实际 T2 补测 | COMPLETED |
| 92467 | source_v16 | 无导出包依赖的原生入口核验及 DPA 对照 | COMPLETED |
| 92468 | source_v16 | 只采用原生路线的正式报告 | COMPLETED |

每个快照包含 `chorus/` 与 `ictc_o3_hp/` 两个源码根目录。
数值核心由 ICTC-O3-HP 的原生 Triton TP/伴随及 CHORUS 相位、Hermitian
算子组成，不使用 cuEq/OpenEq 等外部 TP 算子库。
`source_v13` 增加缺失相位输出梯度处理和未覆盖 density-pairs 的显式拒绝；
其动态导出回归作业 92459 为 2 passed。完整 prepared 回归作业 92448 为 14 passed。

## 工件位置与资格

- [原生正式结果](native_report/INFERENCE_NATIVE_NO_GRAPH.md)与相邻 JSON 保存各尺寸及显存。
- [历史导出实验](report/INFERENCE_AOTI_NO_GRAPH.md)保存旧 AOTI 与原生独立计时，不作为当前路线选择。
- `raw/prepared_csr_t211_v9/l3/`：C128 Persistent / Final 的导出、独立加载、原生和 DPA 记录。
- `raw/prepared_csr_t211_v10/l3c256d4/`：最大 Final 的相同记录。
- `raw/prepared_csr_t211_v13/l3c256d4/`：最大 Persistent 的最新原生/DPA 记录与旧 CSR 导出实验。
- `raw/prepared_t211_v7/l3c256d4/`：最大 Persistent 的初始 atomic AOTI 记录。
- `persistent_actual_t2/`：最大 Persistent 对更小、实际启用 T2 的 C96，同卡原始记录。
- `native_only_smoke/`：直接加载原初始化、无导出包依赖的原生入口与 DPA 对照均为 PASS；
  原生记录为 `package_reference=null`、`requires_aoti=false`，50/32 邻居均核验完整原式。
- `logs/` 保存正式作业日志；`raw/` 中保留相应 JUnit XML 和展开后的 FX 源码。
- `.pt2`、原始初始化及验证张量留在远端，不为展示报告而复制大工件。

历史动态包验证 N=128/256/1024，以及 50/32 邻居、变化的坐标与拓扑。
与完整原始表达式核验后，由不导入 CHORUS/ICTC Python 算子的独立进程加载 `.pt2`。
声明的动态范围不是对所有范围内规模的数值或性能资格承诺。
当前原生基准可直接从原初始化核对完整原式，无需导出包，
入口为 `profile_inference_native_nograph.py --case ... --scope ... --initialization ...`。
5 次预热后进行 30 次同步计时，保留每次样本；编译、邻居构建和原式参照不计入稳态。
CHORUS 计时包含周期边矢量、力组装和 virial；DPA lower 路线接收已整理的邻居矢量，
完整 DPA full 路线也单独保留。显存同时记录 allocated 和 reserved，报告 GB 使用十进制。

较早的失败导出与诊断日志完整保留在远端任务根目录，没有用它们替代通过资格检查的正式结果。
4090 只用于早期接口诊断，其时间不进入 A100 性能结论。现有训练与正式 MD 工件未改动。
