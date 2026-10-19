# 训练主线融合复测

完整 E/F、混合反传与 AdamW；每步重新构建 autograd，CUDA Graph 禁用。
同卡正反顺序各一轮，每轮 20 次，三批变化输入；FP32，AMP/TF32 off。

| 原子数 / 50 邻居 | CHORUS ms | DPA compile ms | DPA / CHORUS | CHORUS reserved GB | DPA reserved GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2048 | 64.659 | 131.681 | 2.037× | 4.230 | 10.383 |
| 4096 | 115.831 | 254.508 | 2.197× | 8.380 | 20.728 |

原始输出、逐张量梯度范数、有限值及优化器更新检查保留。
原式 oracle 的 checkpoint/CPU 卸载仅用于未计时校验，不进入 GPU 训练 step。
小的逐元素舍入异常单独记录，不改变强制验收阈值。
