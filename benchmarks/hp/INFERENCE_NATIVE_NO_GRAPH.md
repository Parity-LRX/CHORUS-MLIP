# 原生融合推理

Persistent 与 Final 均采用 ICTC-O3-HP 原生算子融合，不使用 CUDA Graph，
不再运行 AOTI 导出/调优，也不按尺寸选择 AOTI 与原生计时的最小值。

- [正式同卡结果](results/inference_aoti_nograph_a100/native_report/INFERENCE_NATIVE_NO_GRAPH.md)
- [原始记录与作业身份](results/inference_aoti_nograph_a100/ARTIFACTS.md)

当前 1024 原子、50 邻居、lmax=3 的近参数量配对：

| CHORUS | 原生 ms | DPA4 compile ms | 加速 | reserved GB 原生/DPA |
| --- | ---: | ---: | ---: | --- |
| Persistent C256 / 4层 | 126.745 | 223.979 | 1.767× | 7.336/10.815 |
| Final C256 / 4层 | 94.189 | 165.986 | 1.762× | 7.063/7.699 |

Final 的 DPA C96 实际启用 Triton level 2 value；Persistent 的近参数量 C108
不支持该内核。另行同卡测试实际启用 T2 的较小 C96 时，Persistent 加速为 1.284×，
DPA 参数量少 21.30%。这项和小体系结果保留在完整报告中。
这里是固定初始化结构的执行开销比较，不是已训练模型的等精度比较。

## 直接执行

```python
from chorus.hp import configure_model, energy_forces_virial

# Load the original state, then select CUDA FP32, frozen parameters and eval.
configure_model(model, policy="generated", checkpoint_contraction=True,
                topology_backend="device", capture=False)
out = energy_forces_virial(model, pos, Z, batch, src, dst, shifts, cell)
```

基准入口 `profile_inference_native_nograph.py --case ... --scope ... --initialization ...`
直接与原始公式核对完整 E/F/virial，无导出包依赖。完整 `sbatch` 入口为
`run_inference_native_nograph.sbatch`，两侧都有 CUDA Graph 运行期禁用检查。
网络与力为 FP32，TF32/AMP 关闭，virial 使用 FP64。compile 保留给 DPA4；
CHORUS 的计时对象是原生融合算子本身。

旧 [AOTI 实验](INFERENCE_AOTI_NO_GRAPH.md)仅留档，不作为当前推荐路线。
