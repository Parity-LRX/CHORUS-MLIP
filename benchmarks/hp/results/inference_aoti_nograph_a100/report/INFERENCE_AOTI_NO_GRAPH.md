# 无 CUDA Graph 的 AOTI 推理对照

同卡顺序比较完整 E/F/virial。CHORUS 为实际编译、独立进程加载的动态 N/E `.pt2`；
DPA4 开启 compile 与 Triton 2，实际 value path 单独记录。两侧禁止 capture/replay。
网络与力保持 FP32/TF32 off，virial 为 FP64；邻居构建与编译不进入稳态计时。

本表使用固定结构初始化，评估执行开销，不是已训练的等精度模型配对。
原子数/边数变化均与对应原始公式核验；不是固定输入缓存。

| 配置 | scope | N | DPA C | AOTI ms | 原生 eager ms | DPA lower ms | 最快路线/DPA 加速 | reserved GB 最快/DPA | 实际 T2 value |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| l3 | persistent | 128 | 48 | 7.695 | 17.932 | 8.932 | AOTI: 1.161× | 0.367/0.474 | unsupported width |
| l3 | persistent | 256 | 48 | 10.796 | 18.300 | 13.544 | AOTI: 1.255× | 0.698/0.856 | unsupported width |
| l3 | persistent | 1024 | 48 | 34.541 | 26.731 | 42.212 | native_eager: 1.579× | 2.051/3.259 | unsupported width |
| l3 | final | 128 | 44 | 5.221 | 15.414 | 8.998 | AOTI: 1.723× | 0.321/0.449 | unsupported width |
| l3 | final | 256 | 44 | 7.753 | 16.344 | 13.426 | AOTI: 1.732× | 0.633/0.812 | unsupported width |
| l3 | final | 1024 | 44 | 24.141 | 24.215 | 41.461 | AOTI: 1.717× | 2.458/3.110 | unsupported width |
| l3c256d4 | persistent | 128 | 108 | 34.990 | 41.288 | 36.487 | AOTI: 1.043× | 1.495/1.594 | unsupported width |
| l3c256d4 | persistent | 256 | 108 | 61.293 | 49.779 | 63.135 | native_eager: 1.268× | 1.967/2.970 | unsupported width |
| l3c256d4 | persistent | 1024 | 108 | 226.081 | 126.745 | 223.979 | native_eager: 1.767× | 7.336/10.815 | unsupported width |
| l3c256d4 | final | 128 | 96 | 19.213 | 29.998 | 28.707 | AOTI: 1.494× | 1.286/1.162 | active |
| l3c256d4 | final | 256 | 96 | 32.058 | 36.107 | 47.710 | AOTI: 1.488× | 2.561/2.141 | active |
| l3c256d4 | final | 1024 | 96 | 116.085 | 94.189 | 165.986 | native_eager: 1.762× | 7.063/7.699 | active |

各行的原始计时、参数差、完整输出资格、运行命令和作业身份保存在相邻 JSON 中。
显存 GB=10^9 bytes；显存与吞吐分别判断，不用一个指标代替另一个。

## 本轮尝试

- l3/persistent: export=PASS, load=PASS, native=PASS, DPA=PASS; stage=DONE
- l3/final: export=PASS, load=PASS, native=PASS, DPA=PASS; stage=DONE
- l3c256d4/persistent: export=PASS, load=PASS, native=PASS, DPA=PASS; stage=DONE
- l3c256d4/final: export=PASS, load=PASS, native=PASS, DPA=PASS; stage=DONE

## 独立同卡复测与实际 T2 参考

# 同卡独立复测与实际 Triton 2 参考

1024 原子、50 邻居，完整 E/F/virial；双方禁止 CUDA Graph。

| CHORUS | DPA C | DPA params | 实际 T2 | AOTI ms | 原生 ms | DPA ms | DPA/AOTI |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| persistent | 48 | 1752128 | False | 34.657 | 27.254 | 42.289 | 1.220× |
| persistent | 32 | 863356 | True | 34.657 | 27.254 | 24.404 | 0.704× |
| persistent | 64 | 3459812 | True | 34.657 | 27.254 | 54.462 | 1.571× |
| final | 44 | 1583031 | False | 24.137 | 23.787 | 41.303 | 1.711× |
| final | 32 | 863356 | True | 24.137 | 23.787 | 24.404 | 1.011× |
| final | 64 | 3459812 | True | 24.137 | 23.787 | 54.462 | 2.256× |

主配对为近参数量；C32/C64 是实际支持 T2 value 的上下宽度参考，其参数量不同。
完整原式核验、逐次计时、参数量、显存和命令保存在相邻 JSON/日志中。


## 最大 Persistent 的实际 T2 补测

同卡、1024 原子、50 邻居、完整 E/F/virial、无 CUDA Graph。
CHORUS C256 四层 Persistent：15,441,468 参数，原生 129.368 ms，reserved 7.336 GB。
DPA C96：12,152,174 参数，实际 T2 value=True，166.132 ms，reserved 7.644 GB。
加速 1.284×；DPA 参数量差 -21.30%。
此项保留为较小 DPA 模型的实际 T2 参考，主配对仍是近参数量的 C108。
