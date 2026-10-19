# 原生融合推理：无 CUDA Graph

Persistent 与 Final 统一采用 ICTC-O3-HP 原生算子融合。所有数字均来自
同卡顺序计时的完整 E/F/virial；不采用 AOTI，也不选择不同执行路线的最小值。
网络与力 FP32，TF32/AMP off，virial FP64；两侧禁用 CUDA Graph。

模型使用固定结构初始化，比较执行开销；同 lmax=3、同层数，DPA 通道按参数量选择。

## 1024 原子、50 邻居

| CHORUS | scope | DPA C | 原生 ms | DPA lower ms | 加速 | reserved GB 原生/DPA | 实际 T2 value |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| C128 / 2层 | persistent | 48 | 26.731 | 42.212 | 1.579× | 2.051/3.259 | unsupported width |
| C128 / 2层 | final | 44 | 24.215 | 41.461 | 1.712× | 2.022/3.110 | unsupported width |
| C256 / 4层 | persistent | 108 | 126.745 | 223.979 | 1.767× | 7.336/10.815 | unsupported width |
| C256 / 4层 | final | 96 | 94.189 | 165.986 | 1.762× | 7.063/7.699 | active |

DPA4 均设置 compile 与 Triton level 2；实际内核支持情况单列，不能用开关代替。
DPA lower 不含邻居整理；CHORUS 包含周期几何、力组装与 virial。DPA full 原始计时也保留。
GB 为十进制峰值 reserved；原始记录另存 allocated 和逐次计时。

## 全部已测尺寸

| 配置 | scope | N | 原生 ms | DPA lower ms | 加速 |
| --- | --- | ---: | ---: | ---: | ---: |
| l3 | persistent | 128 | 17.932 | 8.932 | 0.498× |
| l3 | persistent | 256 | 18.300 | 13.544 | 0.740× |
| l3 | persistent | 1024 | 26.731 | 42.212 | 1.579× |
| l3 | final | 128 | 15.414 | 8.998 | 0.584× |
| l3 | final | 256 | 16.344 | 13.426 | 0.821× |
| l3 | final | 1024 | 24.215 | 41.461 | 1.712× |
| l3c256d4 | persistent | 128 | 41.288 | 36.487 | 0.884× |
| l3c256d4 | persistent | 256 | 49.779 | 63.135 | 1.268× |
| l3c256d4 | persistent | 1024 | 126.745 | 223.979 | 1.767× |
| l3c256d4 | final | 128 | 29.998 | 28.707 | 0.957× |
| l3c256d4 | final | 256 | 36.107 | 47.710 | 1.321× |
| l3c256d4 | final | 1024 | 94.189 | 165.986 | 1.762× |

## 最大 Persistent：实际启用 T2 的补测

同卡原生 129.368 ms，DPA C96 166.132 ms，加速 1.284×。
CHORUS 15,441,468 参数，DPA 12,152,174 参数（差 -21.30%）；DPA 实际 T2 value=True。
reserved 为 7.336/7.644 GB。
此项仍未达到 1.5×；近参数量主配对为 C108，不用较大或较小 DPA 替换其结论。

## C128 双 scope 独立复测与实际 T2 参考

| scope | DPA C | 实际 T2 | 原生 ms | DPA ms | 加速 |
| --- | ---: | --- | ---: | ---: | ---: |
| persistent | 48 | False | 27.254 | 42.289 | 1.552× |
| persistent | 32 | True | 27.254 | 24.404 | 0.895× |
| persistent | 64 | True | 27.254 | 54.462 | 1.998× |
| final | 44 | False | 23.787 | 41.303 | 1.736× |
| final | 32 | True | 23.787 | 24.404 | 1.026× |
| final | 64 | True | 23.787 | 54.462 | 2.290× |

C32/C64 为不同参数量的实际 T2 参考，不是近参数量主配对。
原生方案在小体系尚有 Python/核调度开销，并非全部已测尺寸都达到吞吐目标。
