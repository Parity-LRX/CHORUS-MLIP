# 无 CUDA Graph 的推理执行

此页为已结束的 AOTI 实验记录。当前 Persistent / Final 统一采用
[原生融合](INFERENCE_NATIVE_NO_GRAPH.md)，不再继续 AOTI 导出和调优。

本轮同时比较原生融合与真实 AOTInductor `.pt2`，DPA4 开启 compile 和 Triton 2。
两侧均禁止 CUDA Graph capture/replay，包括编译器内部路径。
网络及力为 FP32、TF32/AMP off，virial 为 FP64。

结构族复用训练扩展实验的原初始化：同 lmax=3、同 interaction 深度，DPA 通道按
既定参数量扫描选定。所有同卡配对保留 AOTI、原生、DPA full/lower 的独立计时，
不预设 AOTI 是最快路线。DPA 的实际 T2 value path 与配置开关分别记录。

- [正式同卡表与逐点结论](results/inference_aoti_nograph_a100/report/INFERENCE_AOTI_NO_GRAPH.md)
- [实际支持 T2 的 C32/C64 参考及独立复测](results/inference_aoti_nograph_a100/supported_repeat/SUPPORTED_T2_REPEAT.md)
- [运行、源码快照与资格记录](results/inference_aoti_nograph_a100/ARTIFACTS.md)

## 历史实验接口

```python
from e3nn import set_optimization_defaults
from chorus.hp import prepare_for_export

# Set before reconstructing/loading the model for dynamic N/E.
set_optimization_defaults(jit_script_fx=False)
# Load the original state, including its actual U buffers.
model.eval().requires_grad_(False)
plan = prepare_for_export(model, message_schedule="csr")
# Continue with first-derivative FX flattening, torch.export and AOTInductor.
```

数值资格覆盖完整能量、全部力分量与 virial。单个动态包验证 N=128/256/1024、
50/32 邻居及变化的几何和拓扑；独立进程仅加载 `.pt2`，不导入 CHORUS/ICTC
Python 算子。声明 N/E 范围与实际测试尺寸单独记录。编译、原式参照和邻居构建
不进入稳态计时；周期几何、力组装及 virial 在 CHORUS 计时边界内。

本轮 AOTI runtime 为 PyTorch 2.11；导出环境隔离旧 ABI 的可选邻居/scatter 扩展。
新 benchmark 包返回 E/F/atomic-E/virial 四项，现有 LAMMPS 部署协议及正式 MD 不变。
