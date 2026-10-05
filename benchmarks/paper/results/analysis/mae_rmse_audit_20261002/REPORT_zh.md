# CHORUS 的 MAE 与 RMSE 联合报告分析

结论：适合共同报告。保留 MAE 为主指标，RMSE 用于检查平方误差意义下的大误差敏感性。

## 数据与口径

240 个训练模型，2 个骨干、4 个数据集、5 个配对种子。Final 为 Host、Self、Aligned、Full 四种配置，Persistent 为 Aligned、Full 两种配置。
全部使用原定验证集 Force MAE 选择的 checkpoint；没有为了 RMSE 重选模型。力误差单位为 meV/Å，能量误差为 meV/atom。960 组验证/测试指标均有限、为正，满足 RMSE ≥ MAE。
输入来自 publication_dag_20260829/trained_runs.csv 与 trained_effect_summary.csv。原始表保持不变。本次只在指定 4090 远端进行 CPU 统计。

## 力误差：两项指标的提升幅度

| 骨干 | Scope | 比较 | MAE 降低（95% CI） | RMSE 降低（95% CI） | MAE/RMSE 改善数 | 方向相反数 |
|---|---|---|---:|---:|---:|---:|
| MACE-ICTC | Final | Self / Host | 9.07% [4.87, 11.90] | 6.28% [1.48, 10.75] | 20/20；18/20 | 2 |
| MACE-ICTC | Final | Aligned / Self | 1.57% [-2.41, 5.16] | 3.12% [0.26, 5.84] | 14/20；17/20 | 3 |
| MACE-ICTC | Final | Full / Aligned | 0.61% [-0.12, 1.31] | 0.13% [-0.82, 1.09] | 16/20；11/20 | 5 |
| MACE-ICTC | Final | Full / Host | 11.05% [5.78, 15.77] | 9.32% [2.58, 15.70] | 20/20；18/20 | 2 |
| MACE-ICTC | Persistent | Full / Aligned | 0.91% [0.08, 1.81] | 0.34% [-1.07, 1.61] | 11/20；10/20 | 5 |
| NequIP-SH | Final | Self / Host | 18.33% [9.32, 26.52] | 15.52% [1.61, 26.60] | 20/20；16/20 | 4 |
| NequIP-SH | Final | Aligned / Self | 20.44% [6.12, 30.32] | 18.67% [4.64, 29.66] | 17/20；14/20 | 3 |
| NequIP-SH | Final | Full / Aligned | 0.42% [-0.42, 1.26] | 0.42% [-0.64, 2.16] | 13/20；10/20 | 5 |
| NequIP-SH | Final | Full / Host | 35.30% [18.49, 48.55] | 31.57% [9.42, 48.17] | 20/20；19/20 | 1 |
| NequIP-SH | Persistent | Full / Aligned | 2.43% [1.15, 3.83] | 2.94% [0.91, 5.09] | 18/20；16/20 | 2 |

幅度先计算每个种子的 log(control/treatment)，在数据集内平均，再对四个数据集等权平均，转换为误差降低百分比。95% CI 沿用原报告的 100,000 次分层 bootstrap，并已从原始模型表复现全部点估计。

## 排序与方向一致性

力：10 项既定对比共 200 次配对比较，32 次 MAE/RMSE 方向相反。其中 MAE 改善而 RMSE 恶化 26 次，反向 6 次。
Final：26/40 个同骨干、同数据集、同种子组选择相同的最低误差配置；组内 Spearman 的描述性平均为 0.745；全部配置对之间出现 34/240 次排序反转。
Persistent：33/40 个同骨干、同数据集、同种子组选择相同的最低误差配置；组内 Spearman 的描述性平均为 0.650；全部配置对之间出现 7/40 次排序反转。

能量：10 项既定对比共 200 次配对比较，19 次 MAE/RMSE 方向相反。其中 MAE 改善而 RMSE 恶化 10 次，反向 9 次。
Final：34/40 个同骨干、同数据集、同种子组选择相同的最低误差配置；组内 Spearman 的描述性平均为 0.827；全部配置对之间出现 27/240 次排序反转。
Persistent：40/40 个同骨干、同数据集、同种子组选择相同的最低误差配置；组内 Spearman 的描述性平均为 1.000；全部配置对之间出现 0/40 次排序反转。

上述比较复用了模型，不能作为 200 个独立样本进行显著性检验。相关性在同数据集、同骨干、同 Scope、同种子组内计算，不以跨数据集的误差尺度制造相关性。

## MAE 改善而 RMSE 恶化的力误差实例

| 骨干 | Scope | 数据集 | Seed | 比较 | MAE 降低 | RMSE 降低 |
|---|---|---|---|---|---:|---:|
| MACE-ICTC | Final | mal | 20260617 | diagonal/MACE-ICTC | 8.15% | -10.77% |
| MACE-ICTC | Final | mal | 20260620 | diagonal/MACE-ICTC | 12.03% | -0.41% |
| MACE-ICTC | Final | bucky | 20260620 | full/aligned | 0.26% | -0.15% |
| MACE-ICTC | Final | 3bpa | 20260619 | full/aligned | 1.31% | -0.10% |
| MACE-ICTC | Final | zro | 20260617 | full/aligned | 2.80% | -1.02% |
| MACE-ICTC | Final | zro | 20260618 | full/aligned | 0.39% | -0.62% |
| MACE-ICTC | Final | zro | 20260619 | full/aligned | 0.77% | -0.48% |
| MACE-ICTC | Final | mal | 20260617 | full/MACE-ICTC | 6.80% | -7.67% |
| MACE-ICTC | Final | zro | 20260617 | full/MACE-ICTC | 3.92% | -0.03% |
| MACE-ICTC | Persistent | mal | 20260616 | full/aligned | 0.45% | -1.27% |
| MACE-ICTC | Persistent | mal | 20260617 | full/aligned | 2.83% | -3.87% |
| MACE-ICTC | Persistent | mal | 20260620 | full/aligned | 2.41% | -1.60% |
| NequIP-SH | Final | mal | 20260616 | diagonal/NequIP-SH | 7.62% | -4.49% |
| NequIP-SH | Final | mal | 20260617 | diagonal/NequIP-SH | 3.64% | -13.02% |
| NequIP-SH | Final | mal | 20260618 | diagonal/NequIP-SH | 8.80% | -1.09% |
| NequIP-SH | Final | mal | 20260619 | diagonal/NequIP-SH | 3.91% | -12.92% |
| NequIP-SH | Final | mal | 20260616 | aligned/diagonal | 11.27% | -3.27% |
| NequIP-SH | Final | zro | 20260617 | aligned/diagonal | 1.09% | -2.93% |
| NequIP-SH | Final | zro | 20260618 | aligned/diagonal | 2.06% | -1.44% |
| NequIP-SH | Final | mal | 20260618 | full/aligned | 0.64% | -4.28% |
| NequIP-SH | Final | mal | 20260619 | full/aligned | 0.44% | -0.30% |
| NequIP-SH | Final | 3bpa | 20260618 | full/aligned | 0.16% | -0.51% |
| NequIP-SH | Final | zro | 20260617 | full/aligned | 0.45% | -0.35% |
| NequIP-SH | Final | mal | 20260616 | full/NequIP-SH | 19.19% | -5.60% |
| NequIP-SH | Persistent | 3bpa | 20260617 | full/aligned | 0.68% | -0.14% |
| NequIP-SH | Persistent | 3bpa | 20260618 | full/aligned | 0.77% | -0.04% |

## 在手稿中的安排

正文继续以 Force MAE 为主，RMSE 在相同 checkpoint 下配对补充，并在同一表中显示或放入补充材料。两项指标同时下降支持平均绝对误差与平方误差均改善；方向相反时分别说明，不平均成一个分数。
MAE 不是中位数误差，RMSE 也不是最大误差。RMSE/MAE 反映绝对残差的相对离散度，不能从这两个汇总数恢复 P95/P99 或定位具体异常构型。MD 轨迹连续性与稳定性仍使用轨迹本身的观测。
能量结果保留全部数据集的同 checkpoint 配对表，但不从跨数据集宏平均推出所有体系都改善。ZrO 的能量沿用原报告声明的 train-only calibration。

定义参考：[scikit-learn 回归指标](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)。
