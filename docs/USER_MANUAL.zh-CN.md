# MACE-ICTC 使用说明书

本文档说明 MACE-ICTC 代码库的整体结构、常用模式、训练、MACE 原生模型转换、AOTInductor 导出、LAMMPS 部署、benchmark 和测试方法。

English version: [USER_MANUAL.md](USER_MANUAL.md)

## 1. 这个仓库包含什么

MACE-ICTC 是一个独立的 MACE 实现：它把 MACE 的等变特征从原生 e3nn 球谐基表示改写到 ICTC（Irreducible Cartesian Tensor Decomposition，不可约笛卡尔张量分解）基中，并带有训练、转换、导出和 LAMMPS 部署栈。

核心技术点是 ICTC basis construction：用固定的 `Q`/`U` 算子把 MACE/e3nn 的角向代数重写成不可约笛卡尔张量块，同时保留原始 MACE 的 interaction/readout 语义。这个仓库重点解决 basis conversion、原生 `mace-torch` 数值对应和部署。

主要目录：

- `mace_ictc/models/`：`PureCartesianICTDFix` 主模型、ICTC irreps、tensor product、MACE-compatible symmetric contraction、radial basis、ZBL、long-range 等。
- `mace_ictc/training/`：energy/force/stress trainer，以及 `make_fx` + Inductor 的 force step 编译。
- `mace_ictc/data/`：extended XYZ 解析、H5 dataset、padding、bucket sampler、collate。
- `mace_ictc/cli/`：训练、MACE 转换、AOTI 导出、TorchScript 导出、LAMMPS helper。
- `mace_ictc/interfaces/`：checkpoint 加载、LAMMPS MLIAP wrapper、部署兼容逻辑。
- `mace_ictc/evaluation/`：ASE calculator。
- `mace_ictc/bench/`：MACE-ICTC 和原生 `mace-torch` 的 benchmark。
- `mace_ictc/test/`：数值一致性和 smoke tests。
- `lammps_user_mfftorch/`：LAMMPS `USER-MFFTORCH` C++ package。

核心模型 forward 接口：

```python
model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
```

含义：

- `pos`：原子坐标，形状 `[N, 3]`。
- `A`：原子序数，不是 species index。
- `batch`：每个原子属于哪个 graph，形状 `[N]`。
- `edge_src`, `edge_dst`：有向边索引。
- `edge_shifts`：周期性 image shift，形状 `[E, 3]`。
- `cell`：晶胞，形状 `[B, 3, 3]`。
- 返回值：逐原子 interaction energy，通常形状 `[N, 1]`。

E0 原子参考能通常由训练/导出 wrapper 在模型外部处理。

## 2. 安装

最小安装：

```bash
cd /path/to/MACE-ICTC
pip install -e .
```

可选依赖：

```bash
pip install -e ".[pyg]"   # torch-scatter / torch-cluster 加速
pip install -e ".[cue]"   # cuEquivariance backend
pip install -e ".[e0]"    # 读取 fitted E0 CSV 所需 pandas
pip install -e ".[full]"  # 所有可选依赖
```

版本建议：

- Python >= 3.9。
- PyTorch >= 2.4；`make_fx` 和 AOTInductor 路径建议 PyTorch >= 2.7。
- `e3nn < 0.6`，用于兼容当前 `mace-torch`。
- cuEq、AOTI benchmark 和真实 GPU 训练需要 CUDA。

可选 ICTC tensor-product C++/CUDA extension：

```bash
MFF_BUILD_ICTD_TP_EXT=1 pip install -e .
```

CUDA extension：

```bash
MFF_BUILD_ICTD_TP_EXT=1 MFF_BUILD_ICTD_TP_CUDA=1 pip install -e .
```

不编译 extension 时，Python fallback 仍然可用。

## 3. 命令行工具

安装后会有这些 console scripts：

| 命令 | Python 入口 | 用途 |
|---|---|---|
| `mff-convert-mace` | `mace_ictc.cli.convert_mace` | 把原生 `mace-torch` `ScaleShiftMACE` 转成 MACE-ICTC checkpoint。 |
| `mff-export-aoti` | `mace_ictc.cli.export_aoti_core` | 导出 AOTInductor `.pt2` core，用于 Python/C++/LAMMPS 部署。 |
| `mff-export-core` | `mace_ictc.cli.export_libtorch_core` | 导出 TorchScript core，主要用于旧的 LibTorch 部署。 |
| `mff-lammps` | `mace_ictc.cli.lammps_interface` | 生成 LAMMPS 部署 helper。 |

也可以直接用 module 方式：

```bash
python -m mace_ictc.cli.train --help
python -m mace_ictc.cli.convert_mace --help
python -m mace_ictc.cli.export_aoti_core --help
```

## 4. 核心概念

### 4.1 ICTC 基和 e3nn/MACE 基

原生 MACE 使用 e3nn 球谐基。MACE-ICTC 在内部把等变特征存成 ICTC 笛卡尔基。两者之间由固定的正交矩阵 `Q_l` 相连。

能量、力、virial 这类物理不变量不应该依赖基选择。等变中间特征如果需要转回原生 MACE/e3nn convention，可以用：

```python
model.to_mace_basis(x)
model.to_ictd_basis(x)
```

或者直接使用 `mace_ictc.mace_basis` 里的底层函数。

### 4.2 `angular_basis`

`angular_basis` 控制模型内部用哪个角向基计算：

| 值 | 含义 | 适用场景 |
|---|---|---|
| `ictd` | 默认 ICTC 内部基。 | parity 基准、bridge-U、最保守路径。 |
| `e3nn` | 把固定角向算子一次性 fold 到 e3nn/MACE 基。 | cuEq product 性能路径；AOTI 导出配合 `--cueq-product`。 |

限制：

- `ictd-bridge-u` 没有 e3nn fold path。直接对 bridge-U 请求 `angular_basis=e3nn` 时，导出逻辑会保留 `ictd` 并打印 warning。
- `cueq` product 支持 `angular_basis=e3nn`。
- 训练时如果使用 `angular_basis=e3nn`，固定 buffer 会在第一次 forward 后处于已 fold 状态；checkpoint reload 会恢复 Q blocks 和 product runtime flag，避免 double-fold。

### 4.3 Product backend

| Backend | 含义 | 推荐用途 |
|---|---|---|
| `ictd-bridge-u` | 使用 MACE/e3nn symmetric-contraction U，并把 ICTC/e3nn basis bridge fold 到 U 里。 | MACE parity、原生 MACE 转换、高 `max_ell` 稳妥路径。 |
| `cueq` | 用 cuEquivariance 做 product/symmetric contraction。 | 性能训练和推理，尤其配合 `--angular-basis e3nn`。 |
| `native-mace` | product block 内部直接调用 MACE 的 native symmetric contraction。 | debug/reference。 |
| `ictd-pure-u` | 直接用 ICTC 生成的 U。 | 诊断路径，不是当前高 `max_ell` 主生产路径。 |

### 4.4 `use_reduced_cg`

`--use-reduced-cg` 是 product/symmetric contraction 的结构选项，会改变 CG/path layout 和权重形状。

规则：

- 转换已有原生 MACE checkpoint 时，必须跟随原模型的 `mace_model.use_reduced_cg`，不要手动猜。
- 原生 `mace-torch` 训练也有这个选项，名字是 `--use_reduced_cg`。
- 从头训练 MACE-ICTC 时，只有在你明确要 reduced-CG 架构时才开。
- 它不是稳定训练吞吐的主要加速开关。最近 4090 上 `cueq + angular_basis=e3nn + make_fx` 的完整训练 step 测试里，稳定 step time 变化大约在 -1% 到 +2%，compile time 改善更明显。

## 5. 应该选择哪种模式

| 目标 | 推荐模式 |
|---|---|
| 原生 MACE 转换 / 数值 parity 基准 | `ictd-bridge-u`，`angular_basis=ictd`，通常 `dtype=float64`。 |
| 从头训练但希望模型语义贴近原生 MACE | `ictd-bridge-u`，`function-type=bessel`，保留 MACE-style ScaleShift。 |
| 高吞吐训练 | `cueq`，`angular_basis=e3nn`，`--train-makefx-compile`，配合 bucket。 |
| 从 ICTC checkpoint 导出高吞吐 AOTI | `mff-export-aoti --cueq-product --angular-basis e3nn`，前提是部署环境能加载 cuEq custom op。 |
| 最保守部署 | 不做 cuEq product replacement，保留 checkpoint 原本 basis 或 `ictd`。 |

完整 parity 训练命令：

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend ictd-bridge-u \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --max-steps 200000 \
  --batch-size 4 \
  --loss smooth_l1 \
  --loss-beta 0.5 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 0.001 \
  --min-lr 0.000001 \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr-scheduler plateau \
  --lr-factor 0.8 \
  --scheduler-patience 50 \
  --warmup-batches 1000 \
  --warmup-start-ratio 0.1 \
  --swa \
  --start-swa 225 \
  --swa-lr 0.0001 \
  --swa-energy-weight 1000.0 \
  --swa-force-weight 100.0 \
  --swa-stress-weight 0.0 \
  --ema-decay 0.0 \
  --checkpoint-state-source swa \
  --device cuda \
  --dtype float64 \
  --checkpoint model_bridge_u.pth
```

完整高性能训练命令：

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --makefx-max-slots 8 \
  --pad-nodes-to-max \
  --pad-edges-to-max \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --max-steps 200000 \
  --batch-size 8 \
  --loss smooth_l1 \
  --loss-beta 0.5 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 0.001 \
  --min-lr 0.000001 \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr-scheduler cosine \
  --warmup-batches 1000 \
  --warmup-start-ratio 0.1 \
  --swa \
  --start-swa 225 \
  --swa-lr 0.0001 \
  --swa-energy-weight 1000.0 \
  --swa-force-weight 100.0 \
  --swa-stress-weight 0.0 \
  --ema-decay 0.999 \
  --ema-start-step 1000 \
  --checkpoint-state-source swa \
  --device cuda \
  --dtype float32 \
  --checkpoint model_cueq_e3nn_makefx.pth
```

上面的数值是安全起点，不是对所有化学体系都最优的超参。要和原生 MACE
训练做严格对比，需要同时匹配 dataset split、seed、loss weight、optimizer、
scheduler、batch 构造、dtype、ScaleShift 和 E0 设置。

注意：两个 trainer 从随机初始化开始独立训练，不能因为函数形式一致就保证每一步 bitwise 一样；初始化、数据顺序、optimizer、CUDA kernel、dtype 和 loss 累加顺序都会影响训练轨迹。

### 5.1 多卡训练

MACE-ICTC 训练 CLI 支持 PyTorch `DistributedDataParallel`。默认 `--ddp auto`
会在环境里存在 `WORLD_SIZE>1` 时自动启用 DDP，所以标准 `torchrun` 启动即可：

```bash
torchrun --standalone --nproc_per_node=2 \
  -m mace_ictc.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --pad-nodes-to-max \
  --pad-edges-to-max \
  --batch-size 4 \
  --device cuda \
  --ddp auto \
  --checkpoint model_ddp.pth
```

在 Slurm 上，必须通过调度器申请 GPU，并让 `srun` 或 `torchrun` 启动每卡一个进程。例如：

```bash
#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

source /path/to/conda.sh
conda activate mff

srun python -m mace_ictc.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --pad-nodes-to-max \
  --pad-edges-to-max \
  --batch-size 4 \
  --device cuda \
  --ddp auto \
  --checkpoint model_ddp.pth
```

DDP 行为说明：

- 每个进程会使用 `cuda:$LOCAL_RANK`；不要手动让所有 rank 都用同一个
  `--device cuda:0`。
- `--batch-size` 是每个 rank 的 batch size；全局有效 batch 约等于
  `batch_size * world_size`。
- rank 0 负责 validation log 和 checkpoint 写入；保存出来的 checkpoint key
  不带 `module.` 前缀。
- 使用 `--makefx-buckets` 时，bucket sampler 是 DDP-aware 的。配合
  `--train-makefx-compile` 时建议继续打开 bucket 和 padding，否则每个原始 graph
  shape 都可能触发单独 compile。
- `--train-makefx-compile` 可以在 DDP 下使用，但每个 rank 第一次遇到 bucket 时都有
  compile 开销。它适合长跑和稳定 bucketed shapes，不适合用很短的 smoke test 判断总耗时。
- 如果希望命令在没启用 DDP 时直接失败，用 `--ddp on`；单进程 debug 时用 `--ddp off`。

## 6. 数据流程

训练读取预处理后的 H5：

```text
DATA/
  processed_train.h5
  processed_val.h5       # 可选
  processed_train.counts.npz / bucket sidecar
```

`mace_ictc.data.preprocessing` 支持 extended XYZ 风格数据：

- species / atomic number，
- 坐标，
- 力，
- 总能量，
- cell 和 PBC，
- 可选 stress 或 virial。

预处理是 Python API：

```python
from mace_ictc.data.preprocessing import save_to_h5_parallel

save_to_h5_parallel(
    prefix="train",
    max_radius=5.0,
    num_workers=8,
    data_dir="DATA",
)
```

训练 dataset 是 `mace_ictc.data.datasets.H5Dataset`，batching 使用 `mace_ictc.data.collate.collate_fn_h5`。

使用 `make_fx` 训练时推荐 bucket：

```bash
--train-makefx-compile --makefx-buckets 6
```

这样相近的 atom/edge count 会共用 compile，避免每个 shape 编译一次。

## 7. 训练细节

训练 CLI：

```bash
python -m mace_ictc.cli.train --help
```

关键结构参数：

| 参数 | 含义 |
|---|---|
| `--channels` | hidden channel 数。 |
| `--lmax` | hidden feature 的最高角动量。 |
| `--max-ell` | edge spherical harmonics cutoff；默认等于 `--lmax`。 |
| `--num-interaction` | MACE interaction/product block 数。 |
| `--correlation` | MACE product correlation order，也就是 `save_contraction_order`。 |
| `--function-type` | radial basis；要贴近 MACE 通常用 `bessel`。 |
| `--product-backend` | product backend，常用 `ictd-bridge-u` 或 `cueq`。 |
| `--angular-basis` | `ictd` 或 `e3nn`；性能路径中 `cueq` 可用 `e3nn`。 |
| `--use-reduced-cg` | reduced-CG product layout；转换原生 MACE 时必须跟原模型一致。 |
| `--long-range-mode` | 可选 learned scalar reciprocal correction；用 `reciprocal-spectral-v1` 打开，默认 `none`。 |

优化、step 和 seed：

| 参数 | 含义 |
|---|---|
| `--seed` | 设置 Python、NumPy、PyTorch、DataLoader shuffle 和 bucket sampler 的随机种子；这不等于强制 CUDA kernel 完全 deterministic。 |
| `--epochs` | 最大 epoch 数。 |
| `--max-steps` | 可选的 optimizer step 上限；达到后即使在 epoch 中间也会停止。 |
| `--batch-size` | 每个 batch 或 bucketed batch 的 graph 数。 |
| `--optimizer` | `adamw` 或 `adam`。 |
| `--lr`、`--min-lr`、`--weight-decay` | 初始/最大学习率、scheduler 的 LR 下限、AdamW weight decay。LR 会限制在 `[--min-lr, --lr]`。 |
| `--adam-beta1`、`--adam-beta2`、`--adam-eps`、`--amsgrad` | Adam/AdamW 的数值参数。 |
| `--lr-scheduler` | `plateau`、`exp`、`cosine`、`step` 或 `none`；也接受 `ReduceLROnPlateau` 和 `ExponentialLR` 别名。 |
| `--warmup-batches`、`--warmup-start-ratio` | linear warmup 长度和初始 LR 倍率。 |
| `--lr-factor`、`--scheduler-patience` | plateau scheduler 的 factor 和 patience。 |
| `--lr-scheduler-gamma` | exp scheduler 的 gamma。 |
| `--lr-decay-step`、`--lr-decay-factor` | legacy step scheduler 参数。 |
| `--max-grad-norm` | 可选 gradient clipping 阈值。 |

Loss：

```text
total = energy_weight * loss(E)
      + force_weight  * loss(F)
      + stress_weight * loss(stress)
```

| 参数 | 含义 |
|---|---|
| `--loss` | `smooth_l1` 默认，或者 `mse`。 |
| `--loss-beta` | `--loss smooth_l1` 时 energy/force/stress 共用的 SmoothL1 beta。 |
| `--energy-weight`、`--force-weight`、`--stress-weight` | 总 loss 里的权重；默认 `--stress-weight 0`，即不训练 stress。 |
| `--force-shift-value` | 进入 force loss 前乘到参考 force 上；除非复现 legacy run，否则保持 `1.0`。 |

打开 stress 后，stress 通过 strain derivative 计算。

MACE-style Stage Two / SWA：

| 参数 | 含义 |
|---|---|
| `--swa`、`--stage-two` | 开启 mace-torch 风格 Stage Two：切换 loss weights、降低 LR、保存平均权重。 |
| `--start-swa`、`--start-stage-two`、`--swa-start-epoch` | 从哪个 epoch 开始 Stage Two。如果只开 `--swa` 但不指定 start，默认 `max(1, epochs * 3 // 4)`，和 mace-torch 一致。 |
| `--swa-start-step` | 可选的 global step 触发。 |
| `--swa-lr`、`--stage-two-lr` | Stage Two/SWA LR；必须满足 `--min-lr <= --swa-lr <= --lr`。 |
| `--swa-energy-weight`、`--swa-force-weight`、`--swa-stress-weight` | Stage Two loss 权重。MACE-like energy/force 默认是 `1000` 和 `100`；stress 默认在 stress 关闭时为 `0`，stress 打开时为 `10`。 |
| `--swa-anneal-epochs`、`--swa-anneal-strategy` | SWALR annealing 控制。 |
| `--ema-decay` | 大于 `0` 时开启 EMA，例如 `0.999`；保存为 `e3trans_ema_state_dict`。 |
| `--ema-start-step` | 从哪个 global optimizer step 开始更新 EMA。 |
| `--checkpoint-state-source` | `auto`、`raw`、`ema` 或 `swa`。部署加载器读取 `default_state_source`；`auto` 优先 EMA，其次 SWA，最后 raw。 |

现在这里复刻的是和本 trainer 相关的 mace-torch Stage Two 行为：到 Stage Two
边界后切换 loss weights，暂停主 LR scheduler，把 optimizer LR 切到 `--swa-lr`，
并把平均权重保存为 `e3trans_swa_state_dict`。

ScaleShift：

- 默认 `--scaling rms_forces_scaling`。
- 也支持 `std_scaling`、`no_scaling`。
- 可用 `--atomic-inter-scale` 和 `--atomic-inter-shift` 显式覆盖。
- `--no-atomic-inter-shift` 表示保留 scale，但 interaction energy shift 设为 0。

E0：

- 用 `--atomic-energy-keys` 和 `--atomic-energy-values` 显式传入。
- 如果省略，训练 CLI 使用内置 H/C/N/O 默认值。
- 部署时如果需要绝对能量，AOTI export 用 `--embed-e0`。

Long-range correction：

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 64 --lmax 2 --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-boundary periodic \
  --long-range-reciprocal-backend direct_kspace \
  --long-range-kmax 4 \
  --long-range-source-channels 1 \
  --checkpoint model_lr.pth
```

这会打开当前支持的 long-range 模块 `reciprocal-spectral-v1`。它从最终逐原子的 invariant
descriptor 预测 learned latent scalar sources，并添加 reciprocal-space energy term。贡献项接近
zero-init，所以开启后初始行为接近 short-range baseline，再通过同一个 energy/force/stress loss 学到修正。
它不是固定电荷的解析 Ewald 项，也不需要显式 charge label。

常用选项：

| 参数 | 含义 |
|---|---|
| `--long-range-boundary periodic` | 完全周期 reciprocal solve；`direct_kspace` 必须用这个。 |
| `--long-range-boundary slab` | slab 边界；需要配合 `--long-range-reciprocal-backend mesh_fft`。 |
| `--long-range-reciprocal-backend direct_kspace` | 直接 k-space sum，保留在模型/导出 core 内；适合小 `kmax` 测试。 |
| `--long-range-reciprocal-backend mesh_fft` | FFT mesh 路径，适合更大 periodic/slab 系统。 |
| `--long-range-kmax` | `direct_kspace` 的整数 k-lattice cutoff。 |
| `--long-range-mesh-size` | `mesh_fft` 的 mesh resolution。 |
| `--long-range-source-channels` | latent scalar source channel 数。 |
| `--no-long-range-neutralize` | 关闭每个 graph 的 source neutralization；通常保持默认开启。 |
| `--long-range-include-k0` | 包含 k=0 mode；neutralized source 下通常保持关闭。 |
| `--long-range-green-mode` | `poisson` 或 `learned_poisson`。 |

checkpoint 会把 long-range 超参数写进 `model_hyperparameters`，所以
`LAMMPS_MLIAP_MFF.from_checkpoint` 和 `mff-export-aoti --checkpoint model_lr.pth ...` 会重建同一个架构。
原生 MACE 转换不会给已经训练好的 MACE checkpoint 自动加 long-range module；需要这个修正时，应在
MACE-ICTC 里用 `--long-range-mode reciprocal-spectral-v1` 训练或 fine-tune。

## 8. 原生 MACE 模型转换

用原生 `mace-torch` 训练好模型后，走这个转换路径。输入必须是 torch 保存的
`ScaleShiftMACE` 对象，不只是 raw `state_dict`：

```bash
mff-convert-mace \
  --mace-model mace.model \
  --out mace_ictc.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

推荐的数值对应路径：

- 转换时使用 `--product-backend ictd-bridge-u`。
- 用 `--dtype float64` 做最严格的 energy/force 对比。
- 导出时不替换 product backend。
- 如果部署的 `.pt2` 需要直接返回绝对能量，用 `--embed-e0`。

```bash
mff-export-aoti \
  --checkpoint mace_ictc.pth \
  --elements H,C,N,O \
  --out mace_ictc.pt2 \
  --dynamic \
  --embed-e0
```

如果要在推理端替换成 cuEq product 并做 e3nn fold：

```bash
mff-export-aoti \
  --checkpoint mace_ictc.pth \
  --elements H,C,N,O \
  --out mace_ictc_cueq_e3nn.pt2 \
  --dynamic \
  --cueq-product \
  --angular-basis e3nn
```

cuEq 导出路径会替换 product block，并把兼容的角向算子 fold 到 e3nn convention。部署时需要注册
cuEquivariance custom ops。如果当前 exporter/runtime 不能把 `--embed-e0` 和 `--cueq-product`
同时使用，就把 E0 留在 core 外部处理。

支持的版本和模型变体：

- converter 面向本仓库测试和 benchmark 使用的 `mace-torch` `ScaleShiftMACE` 对象结构；验证环境是
  `mace==0.3.16` 和 `e3nn<0.6`。
- 更新的 `mace-torch` 版本如果保存对象结构和 `extract_config_mace_model` 输出仍兼容，可能可以工作，
  但这里不自动承诺覆盖。
- checkpoint 加载本身受 Python pickle 兼容性影响。很老的 MACE 预训练模型可能需要对应历史版本的
  `mace-torch`/`e3nn` 环境先成功 load；只要模型对象能 load，后续是否能转换由下面的结构检查决定。
- 这不是任意 native-MACE-like 实现、raw state dict 或自定义 research fork 的通用转换器；它要求保存出来的是兼容的 `ScaleShiftMACE`。

转换支持范围是故意收紧的；不支持的原生 MACE 变体会报错，而不是静默转换成错误模型。当前主要假设包括：

- `ScaleShiftMACE`，
- Bessel radial basis，
- `radial_MLP=[64,64,64]`，
- hidden irreps 使用 MACE parity，`l=0..L` 连续，且各 `l` channel 数一致，
- 每层 correlation 一致，
- 无 pair repulsion / distance transform，
- SiLU MACE-style scalar readout，`MLP_irreps=16x0e`，
- 第一层 interaction 是 `RealAgnosticInteractionBlock` 或 `RealAgnosticResidualInteractionBlock`，
- 后续 interaction 是 `RealAgnosticResidualInteractionBlock`，
- `max_ell >= hidden_irreps.lmax`，
- `num_interactions >= 2`。

backend 差异：

- `ictd-bridge-u`：推荐转换 backend；通过 ICTC/e3nn basis bridge fold MACE/e3nn 的 U convention，是主数值对应路径。
- `native-mace`：debug/reference backend，用于诊断 MACE 侧 contraction 行为。
- `cueq`：性能 product backend，尤其适合导出时配合 `--cueq-product --angular-basis e3nn`。
- `ictd-pure-u`：直接用 ICTC 生成 U 的诊断路径，不是原生 MACE exact conversion 主路径。

converter 会读取 `mace_model.use_reduced_cg` 并用同样配置重建 MACE-ICTC。用户转换原生 MACE 模型时不应该手动猜这个选项。
转换会保留原生 MACE 架构，不会自动添加 learned long-range module。

### 8.1 OFF23 预训练模型转换和 `mff/torch` 烟测

公开的 MACE 预训练模型经常是 pickled Python model object。converter 只有在 Python 能成功
load 原始对象后才能开始工作。比较老的 OFF23 checkpoint 可能需要历史版本
`mace-torch`/`e3nn` 环境来完成 load 和 conversion；转换后的 ICTC checkpoint 可以再交给当前
MACE-ICTC runtime 加载、导出和部署。

**搭建这个历史加载环境。** 当前 `e3nn`（0.5.x / 0.6.x）无法反序列化 OFF23 checkpoint：`torch.load`
会从 `e3nn/util/codegen/_mixin.py` 抛 `ValueError: too many values to unpack (expected 2)`（e3nn 各版本间
compiled-module 的 pickle 格式变了）。用 OFF23 序列化时的版本——**`e3nn==0.4.4` + `mace-torch==0.3.16`**——
装到隔离目录、再用 `PYTHONPATH` 前置，让它们 shadow 掉环境里更新的 `e3nn`/`mace`，而 `torch` 仍由环境提供
（反序列化对 torch 版本不敏感）：

```bash
# 一次性：把这两个 legacy 反序列化依赖装到任意持久目录
pip install --target=$HOME/compat_e3nn044/e3nn_0_4_4        "e3nn==0.4.4"
pip install --target=$HOME/compat_e3nn044/mace_torch_0_3_16 "mace-torch==0.3.16"

# 前置它们来 load + convert（mace_ictc 也要可导入：PYTHONPATH 或已安装）：
PYTHONPATH=$HOME/compat_e3nn044/mace_torch_0_3_16:$HOME/compat_e3nn044/e3nn_0_4_4 \
  python -m mace_ictc.cli.convert_mace \
    --mace-model /path/to/MACE-OFF23_small.model \
    --out MACE-OFF23_small_ictd_bridge_u_f64.pth \
    --product-backend ictd-bridge-u --dtype float64
```

转换出的 `.pth` 是普通 state_dict，可在正常（当前 `e3nn`）环境里加载用于训练和导出——只有*加载 pickled
源模型*这一步需要 legacy 依赖。注意：fresh-build 的 converter parity 测试
（`mace_ictc/test/test_mace_converter.py`）在进程内现建 MACE，不受影响；这个 legacy 环境只用于加载*已存档的*
foundation checkpoint。

OFF23 small 模型转换示例：

```bash
mff-convert-mace \
  --mace-model /path/to/MACE-OFF23_small.model \
  --out MACE-OFF23_small_ictd_bridge_u_f64.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

float64 适合做 native-MACE 数值对应审计。LAMMPS 部署通常导出 float32 AOTInductor core；
可以用同一条转换命令把 `--dtype` 改成 `float32` 生成部署 checkpoint。如果目标 MD
体系原子数固定，可以使用 static-N 导出：

```bash
mff-export-aoti \
  --checkpoint MACE-OFF23_small_ictd_bridge_u_f32.pth \
  --elements H,C,N,O,F,P,S,Cl,Br,I \
  --atoms 6 \
  --degree 5 \
  --static-n \
  --dtype float32 \
  --device cuda \
  --embed-e0 \
  --out MACE-OFF23_small_ictd_bridge_u_f32_static6.pt2
```

最小 LAMMPS input：

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch 4.5 cuda
pair_coeff * * MACE-OFF23_small_ictd_bridge_u_f32_static6.pt2 H C N O

thermo 1
thermo_style custom step temp pe etotal fmax
run 0
```

在 4090 验证环境中，`build-mfftorch` 和 `build-mfftorch-kk` 都能编译并加载 OFF23 `.pt2`
core。普通 build 和 Kokkos build 输出的 LAMMPS 能量在打印精度内一致。fresh static-6
导出的 `run 0` 对比如下：

| 量 | LAMMPS `mff/torch` | Python checkpoint |
|---|---:|---:|
| energy (eV) | `-6633.036` | `-6633.03613281` |
| 最大绝对力分量 (eV/A) | `11.767612` | `11.76760674` |

这里 LAMMPS thermo 里的 `fmax` 是最大绝对力分量，不是逐原子力向量范数最大值。因此应与
Python 中的 `max(abs(forces))` 比较，而不是和 `max(norm(forces_i))` 比较。

同一个 converted float64 checkpoint 也和原生 `mace-torch` 做了苯 same-frame 轨迹对比：
最大能量绝对差 `2.73e-12 eV`，最大力分量差 `4.44e-15 eV/A`。这个测试验证 conversion
bridge；AOTI 导出仍应单独做 eager-vs-compiled 数值检查，因为 compiler lowering 会改变浮点运算顺序。

## 9. AOTInductor 导出

基础导出：

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model.pt2 \
  --dynamic \
  --embed-e0
```

性能导出：

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model_cueq_e3nn.pt2 \
  --dynamic \
  --cueq-product \
  --angular-basis e3nn \
  --assume-cutoff-edges \
  --preserve-edge-order \
  --fuse-selector-message-linear \
  --inductor-max-autotune
```

性能导出的 cuEq 路径里，如果当前 exporter/runtime 不能把 `--embed-e0` 和 product replacement
组合使用，就把 atomic E0 留在导出 core 外处理。需要 E0-embedded 绝对能量 `.pt2` 时，最简单稳妥的是使用上面的 bridge-U 保守导出。

常用导出选项：

| 选项 | 含义 |
|---|---|
| `--dynamic` | 尽量导出动态 atom/edge 维度。 |
| `--static-n` | 固定 atom count，适合固定 N 的 MD 或 dynamic export 不稳定时。 |
| `--embed-e0` | 把 E0 原子参考能嵌进导出 core。 |
| `--cueq-product` | 导出时把 product block 替换成 cuEq product。 |
| `--angular-basis e3nn` | 对可 fold 的 product backend 做 e3nn basis fold。 |
| `--assume-cutoff-edges` | 假设调用方已经筛好 cutoff 内边，跳过模型内部 edge mask。 |
| `--preserve-edge-order` | 假设调用方 edge order 稳定，跳过模型内部按 `edge_dst` 排序。 |
| `--fuse-selector-message-linear` | 在支持的位置融合 selector/message linear。 |
| `--inductor-max-autotune` | 编译更慢，可能得到更快 kernel；必须 benchmark 后再用于生产。 |

关于 `strict=False`：

- exporter 会优先尝试 strict export。
- 如果 strict 失败但属于 exporter 限制，会 fallback 到 non-strict。
- non-strict 不等于默认正确；导出流程仍然会 compile/load `.pt2` 并做数值比较。

## 10. ASE 和 Python 推理

ASE wrapper 是 `mace_ictc.evaluation.calculator.MyE3NNCalculator`。

示例：

```python
import torch
from mace_ictc.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from mace_ictc.evaluation.calculator import MyE3NNCalculator

wrapper = LAMMPS_MLIAP_MFF.from_checkpoint(
    "model.pth",
    element_types=["H", "C", "N", "O"],
    device="cuda",
)

atoms.calc = MyE3NNCalculator(
    model=wrapper.wrapper.model,
    atomic_energies_dict={1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
    device=torch.device("cuda"),
    max_radius=5.0,
)
```

真实 MD 生产建议先导出 AOTI/LAMMPS，并做数值验证后再跑长程模拟。

## 11. LAMMPS 部署

LAMMPS package 位于：

```text
lammps_user_mfftorch/
```

阅读：

- `lammps_user_mfftorch/README.md`
- `lammps_user_mfftorch/docs/BUILD_AND_RUN.md`

提供：

- `pair_style mff/torch`
- `pair_style mff/torch/kk`

当前 `.pt2` LAMMPS 部署支持 energy 和 force。Physical tensor outputs 还不是公开支持的
LAMMPS 接口。

一般流程：

1. 训练或转换 checkpoint。
2. 根据目标部署方式导出 AOTI `.pt2` 或 TorchScript core。
3. 编译带 `USER-MFFTORCH` 和 LibTorch 的 LAMMPS。
4. 在 LAMMPS input 中使用导出的模型。

简化示例：

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O

fix 1 all nve
run 100
```

`pair_coeff` 里的元素顺序必须和导出/加载时一致。

### 11.1 多卡 LAMMPS 运行

`USER-MFFTORCH` 是 MPI pair style。多卡运行通常是每张 GPU 一个 MPI rank，
不是一个 LAMMPS 进程同时控制所有 GPU。编译时需要 MPI、`USER-MFFTORCH`、
LibTorch；如果要用 Kokkos GPU 数据路径，还需要 Kokkos/CUDA。

普通非 Kokkos pair style：

```bash
export MFF_DEBUG_BUNDLE=1   # 可选：打印 requested/selected device
mpirun -np 2 /path/to/lmp -in in.mfftorch
```

LAMMPS input 中使用：

```lammps
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O
```

Kokkos GPU 数据路径：

```bash
export MFF_DEBUG_BUNDLE=1
mpirun -np 2 /path/to/lmp -k on g 2 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
```

input 里可以显式写 Kokkos style：

```lammps
pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O
```

也可以写 `pair_style mff/torch`，让 `-sf kk` 在支持的 build 中映射到 Kokkos 版本。

设备映射细节：

- 普通 `mff/torch` 会从 MPI/Slurm 的 local-rank 环境变量选择本地 CUDA 设备，
  例如 `SLURM_LOCALID`、`LOCAL_RANK`、`OMPI_COMM_WORLD_LOCAL_RANK` 或
  `MPI_LOCALRANKID`。
- `mff/torch/kk` 在单节点、每 GPU 一个 MPI rank 的情况下按 rank 映射到 Kokkos GPU。
  多节点 Kokkos 运行需要先显式验证 local-rank 映射，不建议未经验证直接生产。
- 设置 `MFF_DEBUG_BUNDLE=1` 后，engine 会打印 requested device 和 selected device；
  生产前应确认不同 local rank 选择了不同 GPU。
- 如果使用 static-N `.pt2`，导出时的 `--atoms` / `--degree` 容量必须覆盖每个 MPI rank
  上的 local atoms 加 ghosts。部署环境支持时，N-dynamic `.pt2` 更省心。
- 长 MD 前先比较 `-np 1` 和 `-np N` 的 `run 0` energy/force。fp32 下由于 edge order
  和归约顺序不同，可能有很小差异；大差异通常说明 domain decomposition、cutoff
  或导出容量有问题。

## 12. 长程与色散（训练 → 导出 → 部署）

MACE-ICTC 学习两类长程修正，都是网络内置的（不依赖外部 Ewald 或 libMBD），都能通过
`mff/torch` pair style 部署到 LAMMPS。本节是端到端参考：有哪些方法、怎么训练、怎么导出、
怎么在 LAMMPS 里跑。

统一机制：网络从描述符发射**每原子隐变量源**。训练时长程能量内联计算，从同样的
energy/force/stress 损失里学（无需电荷标签）。部署时把源导出，长程能量要么留在编译图里
（pairwise-C6 色散），要么**延后到专用 C++ 求解器**（倒空间静电、MBD），保持可分离 + 可扩展。

### 12.1 有哪些方法

| 类别 | 模式（训练 flag） | 物理 | 部署路径 |
|---|---|---|---|
| 静电 | `--long-range-mode reciprocal-spectral-v1` | 学习的隐变量标量电荷，倒空间 | C++ 倒空间求解器（`direct_kspace` 留在 core 内） |
| 静电（多极） | `… --long-range-reciprocal-backend mesh_fft --long-range-max-multipole-l {1,2}` | 学习的单极/偶极/四极，mesh-FFT | C++ mesh-FFT 倒空间求解器 |
| 色散（pairwise） | `--long-range-dispersion-mode pairwise-c6` | 学习的 C6 + Becke–Johnson 阻尼，r⁻⁶ | 图内（随 `.pt2` 一起） |
| 色散（多体） | `--long-range-dispersion-mode mbd-slq` | MBD@rsSCS 耦合偶极，无矩阵 Tr[√C] | C++ MBD 求解器 |
| 色散（稠密） | `--long-range-dispersion-mode mbd` | 稠密 QHO 本征分解 | 仅参考/验证 |

MBD-SLQ 还有两个维度：

- **算子后端** `--mbd-operator-backend edge_sparse|pme_fft`。`edge_sparse`（默认）在 cutoff 色散图
  上直接求和阻尼偶极张量（O(E)，中小体系最快）；`pme_fft` 是 reciprocal-only 的 FFT matvec（大
  周期盒子）。**训练和部署必须一致**——C++ 求解器跑对应的算子。
- **极化率阶数** `--mbd-anisotropic`。关 = 各向同性标量 α（发射 `[N,2]` 源 `(ω, α)`）；开 =
  **各向异性 l=2 张量** α（发射 `[N,8]` `(ω, α_iso, 6×B)`，耦合 W=ω·B）。张量从 l=2 节点块构造，
  需要 `--lmax ≥ 2`。这是 ICTC 独有的路径，且几乎免费（见 12.5）。

静电和色散相互独立，可在一个模型里组合（如 multipole l=2 + MBD）。

### 12.2 训练

MBD-SLQ，各向同性，默认 `edge_sparse` 后端：

```bash
python -m mace_ictc.cli.train \
  --data-dir DATA \
  --channels 128 --lmax 2 --num-interaction 2 \
  --long-range-dispersion-mode mbd-slq \
  --dispersion-cutoff 8.0 \
  --mbd-operator-backend edge_sparse \
  --checkpoint model_mbd.pth
```

各向异性（l=2 张量）MBD——加 `--mbd-anisotropic`（需要 `--lmax >= 2`）：

```bash
python -m mace_ictc.cli.train ... \
  --long-range-dispersion-mode mbd-slq --dispersion-cutoff 8.0 \
  --mbd-operator-backend edge_sparse --mbd-anisotropic \
  --checkpoint model_mbd_aniso.pth
```

Pairwise-C6 色散（最便宜，图内）：

```bash
python -m mace_ictc.cli.train ... --long-range-dispersion-mode pairwise-c6 --dispersion-cutoff 8.0 --checkpoint model_c6.pth
```

多极静电（mesh-FFT，偶极+四极）——mesh-FFT 多极路径要求 full-Ewald 屏蔽：

```bash
python -m mace_ictc.cli.train ... \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-reciprocal-backend mesh_fft --long-range-mesh-size 32 \
  --long-range-max-multipole-l 2 --long-range-mesh-fft-full-ewald \
  --long-range-assignment pcs --checkpoint model_mp.pth
```

主要色散/MBD flag：

| Flag | 默认 | 含义 |
|---|---|---|
| `--long-range-dispersion-mode` | `none` | `none` / `pairwise-c6` / `mbd` / `mbd-slq` |
| `--dispersion-cutoff` | `8.0` | 色散邻居 cutoff（Å）；~8 ≈ MBD 能量收敛到 ~5%；`0` 复用模型边表 |
| `--mbd-operator-backend` | `edge_sparse` | `edge_sparse`（直接）/ `pme_fft`（倒空间）；部署须一致 |
| `--mbd-anisotropic` | 关 | l=2 张量极化率（`[N,8]` 源）；需要 `lmax ≥ 2` |
| `--mbd-pme-mesh-size` | `32` | `pme_fft` 的 PME 网格 |
| `--dispersion-slq-num-probes` | `8` | Tr[√C] 的 Hutchinson 探针数 |
| `--dispersion-slq-lanczos-steps` | `16` | 每探针的 Lanczos 步数 |

checkpoint 把所有长程超参存在 `model_hyperparameters` 里，导出时重建完全相同的结构。

### 12.3 导出

两种 core，都是 parity 正确的——按目标选：

| | AOTI `.pt2`（生产） | TorchScript `.pt`（便携） |
|---|---|---|
| 工具 | `mace_ictc.cli.export_aoti_core` | `mace_ictc.cli.export_libtorch_core` |
| 速度 | Inductor 融合 fwd+bwd，快 ~3.7–5.4× | C++ autograd，不融合 |
| N | static-N 或 N-dynamic | N 灵活（一个 core 任意 N） |
| 长程 | 完整（倒空间/MBD 延后到 C++；C6 图内） | 极少 |

从训练好的 checkpoint 导出（继承模型的长程配置，见第 9 节）：

```bash
python -m mace_ictc.cli.export_aoti_core --checkpoint model_mbd_aniso.pth --out model.pt2
```

合成/组合导出（导出时现搭长程配置，用于测试）接受：

| Flag | 含义 |
|---|---|
| `--dispersion-mode mbd-slq` | 色散类别 |
| `--dispersion-cutoff 8.0` | 色散 cutoff（Å） |
| `--mbd-operator-backend edge_sparse\|pme_fft` | MBD 算子（与训练一致） |
| `--mbd-anisotropic` | 各向异性 l=2 张量（`[N,8]` 源） |
| `--lr-mesh-size 32` | mesh-FFT / `pme_fft` 的网格 |
| `--long-range-mode`, `--long-range-multipole-l` | 静电块 |

```bash
python -m mace_ictc.cli.export_aoti_core --route baseline --channels 128 --lmax 2 \
  --num-interaction 2 --dtype float32 --device cuda \
  --dispersion-mode mbd-slq --dispersion-cutoff 8.0 --mbd-operator-backend edge_sparse \
  --mbd-anisotropic --out model_aniso.pt2
```

导出会写一个 `model.pt2.json` 旁文件，含部署元数据；MBD 部分包含
`long_range_mbd_source_channels`（2 各向同性 / 8 各向异性）、`mbd_operator_backend`、
`long_range_mbd_beta`、`long_range_mbd_coupling_scale`，C++ engine 读这些。

> **延后部署**：MBD/倒空间模型的 AOTI forward 只发射每原子源，**长程能量延后到 C++ 求解器**，
> 所以无论用哪种长程方法，forward 时间 ~不变。长程开销只出现在 MD 步里（见 12.5）。

### 12.4 LAMMPS 部署

pair style 语法（`pair_mff_torch.cpp`）：

```
pair_style mff/torch <model_cutoff> [cpu|cuda] [dispersion <disp_cutoff>]
pair_coeff * * <model.pt2> <elem_1> [<elem_2> ...]
```

色散/MBD 模型**必须**带 `dispersion <disp_cutoff>` 关键字：它建 LAMMPS ghost 邻居表（到
`disp_cutoff`），C++ MBD/色散求解器复用这个表。用**和训练相同的 cutoff**。其余（算子后端、各向
异性源宽度、β、coupling scale、探针数）都烘焙进 `.pt2` 元数据，所以各向同性和各向异性 MBD 的
LAMMPS 输入完全相同（源宽度从元数据来）：

```lammps
units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data system.data
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

pair_style mff/torch 5.0 cuda dispersion 8.0
pair_coeff * * model_mbd_aniso.pt2 H C N O

fix 1 all nve
run 1000
```

- **Pairwise-C6** 也要 `dispersion <cutoff>`（用来提供邻居表）；能量本身随 `.pt2` 图内一起。
- **倒空间静电**不需要额外关键字——engine 从元数据和网格跑倒空间求解器。
- **网格分配（assignment）训练与部署自动对齐。** C++ 倒空间求解器从部署元数据读 `long_range_assignment`
  （`cic` / `pcs`）并跑对应的 spread + sinc 窗反卷积 stencil，所以用 `--long-range-assignment pcs` 训的
  多极模型在 LAMMPS 里能复现其 in-model 能量/力（OFF23-large l=2 实测相对 ~2e-6）。训练与部署通过元数据
  自动匹配，无需额外 flag。
- `pair_coeff` 的元素顺序必须和导出 `--elements` 一致；`NULL` 跳过某个 type。

匹配规则与约束：

- **后端必须一致** 训练↔部署（`edge_sparse` vs `pme_fft`）；C++ 跑元数据里写的那个。
- **单像 cutoff**（`edge_sparse`）：`2·dispersion_cutoff ≤` 最小盒面高度（最近像）；否则 pair
  style 报错——加大盒子或减小 cutoff。`pme_fft` 没有这个限制。
- 部署 MBD 能量用 C++ Chebyshev trace 估计器，训练用 Lanczos/Newton–Schulz：*算子*精确一致，但
  随机 Tr[√C] 估计器不同（非逐位一致；各向同性/各向异性都如此）。
- `pair_style mff/torch/kk` 是 Kokkos/GPU 变体，语法相同。

### 12.5 开销与如何选择

各向异性 vs 各向同性 MBD——512 原子，ch128，lmax2，`edge_sparse`，cutoff 8（RTX 4090）：

| 模式 | 各向同性 | 各向异性 | Δ |
|---|---|---|---|
| 训练（fwd + 力反传 + loss 反传） | 304.9 ms | 304.9 ms | +0.0% |
| 推理（fwd + 力） | 109.3 ms | 109.6 ms | +0.3% |
| MD（部署） | 35.3 ms/步 | 37.3 ms/步 | +5.7% |

各向异性多了一个 l=2 读出，加上每个 SLQ matvec 里每原子一次 3×3 W 矩阵乘：训练/推理基本免费
（MBD 是启动/autograd overhead-bound），MD ~6%。（eager 训练/推理的绝对值含 O(N²) 色散搜索，真
实训练用预算好的边表不会有；但 Δ 是干净的。）

选择规则：

- **色散**：不需要多体效应时，`pairwise-c6` 最便宜（图内，~免费）且是不错的默认；需要多体屏蔽/
  极化响应时用 `mbd-slq`。
- **MBD 后端**：中小体系用 `edge_sparse`（到 ~8k 原子都更快）；大周期盒子用 `pme_fft`。
- **各向异性**：当希望 l=2 表示驱动方向性极化率时开启（ICTC 独有路径）；开销很小。
- **Core**：要吞吐用 AOTI `.pt2`；要 N 灵活/便携用 TorchScript `.pt`。

### 12.6 训练稳定性与 warm-start（长程 head）

**Warm-start 是给模型加任意原生长程 head 的推荐方式**——MBD 色散、倒空间静电、多极静电都一样。先训
（或转换）一个短程 backbone，再用非严格加载（`--finetune`）在其上加长程物理：backbone 被 warm-start，
只有新 head 从头初始化。这对三种 head 都是收敛最快、最稳的路径。同一个 flag 适用于每种 head——唯一变的
是你传哪些 `--long-range-*` flag。

`--finetune` 以**非严格**方式加载权重（缺失的长程-head 键保持新初始化、多余的键忽略），用全新 optimizer
从 epoch 0 开始。架构 flag 仍要描述*完整*模型（backbone + 长程 head）。之后要继续训完整模型，正常 resume：
`--resume-checkpoint model.pth --resume-training-state`（严格加载、恢复 optimizer）。

**warm-start 为什么稳：每种 head 都温和起步。** 新加的 head 在初始化时必须贡献 ~0 能量，才不会破坏被
warm-start 的 backbone，同时还要可训练。每种 head 都内置温和起步：

- **静电**（`--long-range-mode reciprocal-spectral-v1`，默认 `direct_kspace` 后端）：能量被一个可学习的
  `energy_scale` 门控、初始化为 **0**，所以初始贡献恰好为零、随训练逐步上升（能 bootstrap 是因为隐变量源
  非零，故 `dE/d energy_scale ≠ 0`）。
- **多极静电**（`mesh_fft` + `--long-range-max-multipole-l {1,2}`）：倒空间能量关于 q/μ/Q 源是*二次*的，
  所以用一个可学习的**门控**（`--long-range-multipole-gate-init`，默认 `0.1`）缩放发射的源——head 以
  ~`gate²` 的强度起步（温和）但仍可训练。门控被折进导出的源里，所以训练与部署保持一致。backbone 特别"硬"
  时可调小（如 `0.02`）；设 `1.0` 则恢复未门控的从头训练行为。
- **MBD** 是在*学习到的*极化率上做耦合偶极求解，所以训练早期耦合矩阵 `C` 可能漂向极化灾变边缘。两个数值
  护栏内置且常开：**detached 谱再缩放**让 `C` 每一步都严格正定（`Tr[√C]` 估计器永远不会看到非 PD 算子，
  谱本身不会产生硬 NaN），各向异性 l=2 readout 用**光滑范数**让二阶（force-loss）梯度在初始化时保持有限。

**配方**（都从 `backbone.pth` = 一个不带长程、已训好的 MACE-ICTC checkpoint，或转换自 MACE 的 checkpoint 出发）：

```bash
# (a) MBD 色散
python -m mace_ictc.cli.train --data-dir DATA \
  --channels 128 --lmax 2 --num-interaction 2 \
  --long-range-dispersion-mode mbd-slq --mbd-anisotropic --dispersion-cutoff 8.0 \
  --resume-checkpoint backbone.pth --finetune \
  --max-grad-norm 10 --lr 1e-3 \
  --checkpoint model_mbd_aniso.pth

# (b) 静电（学习的隐变量标量电荷，倒空间）
python -m mace_ictc.cli.train --data-dir DATA \
  --channels 128 --lmax 2 --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --resume-checkpoint backbone.pth --finetune \
  --max-grad-norm 10 --lr 1e-3 \
  --checkpoint model_es.pth

# (c) 多极静电（学习的单极/偶极/四极，mesh-FFT）
python -m mace_ictc.cli.train --data-dir DATA \
  --channels 128 --lmax 2 --num-interaction 2 \
  --long-range-mode reciprocal-spectral-v1 \
  --long-range-reciprocal-backend mesh_fft --long-range-mesh-fft-full-ewald \
  --long-range-max-multipole-l 2 --long-range-assignment pcs \
  --long-range-multipole-gate-init 0.1 \
  --resume-checkpoint backbone.pth --finetune \
  --max-grad-norm 10 --lr 1e-3 \
  --checkpoint model_multipole.pth
```

静电与色散相互独立，可在一个模型里组合（如 multipole l=2 + MBD），同时传两组 flag 即可。
`--long-range-max-multipole-l` 不能超过模型 `lmax`。

**从头训 MBD** 也行，但更娇气：

- 始终保留 `--max-grad-norm 10`（项目标准）。loss 若 spike，降 LR 或降色散 cutoff——**不要**调高 clip。
- 两层交互（感受野 ≈ 2×cutoff）一般过了前 ~20 epoch 就稳；再缩放温和触发、收敛平滑。
- **单层交互**（`--num-interaction 1`）的 MBD 在 `lr 1e-3` 下敏感：粗感受野让 α/ω 在 batch 间剧烈摆动、
  再缩放硬触发、loss 会 spike（无 NaN 但毁收敛）。warm-start 它，或降 LR。

**MBD 的 eager vs 编译。** `--train-makefx-compile` 给 ~3× 单步吞吐，适合吞吐 benchmark 和中等长度（几十
epoch）的训练。**长训/生产 MBD 训练优先用 eager**（去掉 `--train-makefx-compile`）：编译路径每个原子数
bucket 持有一张图，一个深层的 `torch.compile` 多图二阶反传交互会在长的多 shape 训练后期冒出*随机* NaN
（靠关闭 AOTAutograd donated buffer 缓解、但未根除）。这是训练侧、多图特有的问题：**AOTI 部署是单图、
不受影响。**

## 13. Benchmark

主 benchmark：

```bash
python -m mace_ictc.bench.bench_mace_ictc_vs_mace \
  --device cuda \
  --dtype float32 \
  --channels 64 \
  --atoms-list 256,1024,4096 \
  --configs 1:1,2:2,2:3 \
  --train-iters 5 \
  --infer-iters 20 \
  --out-dir /tmp/mace_ictc_bench
```

这个 benchmark 是 backend/kernel throughput harness，不是化学精度验证。

建议分开比较：

- 原生 `mace-torch` e3nn backend。
- 原生 `mace-torch` cuEq backend。
- MACE-ICTC bridge-U eager/make_fx/AOTI。
- MACE-ICTC cuEq product eager/make_fx/AOTI。
- 可选 pure-U 诊断路径。

报告时不要混在一起：

- 首次 compile 时间，
- 稳态 step time，
- ASE/Python overhead，
- neighbor-list overhead，
- LAMMPS 端吞吐。

## 14. 测试和验证

基础 smoke tests：

```bash
python -m mace_ictc.test.test_training_smoke
python -m pytest mace_ictc/test/test_angular_basis.py -q
python -m pytest mace_ictc/test/test_export_aoti_core.py -q
```

MACE converter 验证：

```bash
python -m mace_ictc.test.test_mace_converter
```

cuEq product 测试：

```bash
python -m pytest mace_ictc/test/test_cueq_product_backend.py -q
python -m mace_ictc.test.test_cueq_makefx_training
```

cuEq 和 make_fx 测试需要 CUDA 环境。

数值精度预期：

- float64 bridge-U conversion 应该接近机器精度。
- float32 cuEq 路径按 float32 容差判断。
- AOTI 和 make_fx 路径必须和 eager 输出做 compile/load 后数值比较。

## 15. 常见坑

### Bridge-U 和 `angular_basis=e3nn`

Bridge-U 没有 e3nn fold path。bridge-U parity 路径应使用：

```bash
--product-backend ictd-bridge-u --angular-basis ictd
```

如果要 e3nn-folded product 推理，用：

```bash
--cueq-product --angular-basis e3nn
```

### cuEq product replacement

从 bridge-U product 替换到 cuEq product 时，只能拷 learnable MACE contraction weights，不能拷 bridge-U 的固定 `U_matrix_*` buffer。bridge-U 的 U 已经包含 ICTC/e3nn basis fold，直接拷到 cuEq 会错。当前 export path 已处理这个问题。

### `use_reduced_cg`

它是结构选项，不是无害的速度开关。转换原生 MACE 时跟随原 checkpoint；从头训练时要主动决定，并保证 checkpoint metadata 记录一致。

### ScaleShift 和 E0

MACE-style ScaleShift 作用在 interaction energy 上；E0 原子参考能单独加。要和原生 MACE 或部署绝对能量一致，必须检查：

- atomic energy keys/values，
- scale/shift，
- `avg_num_neighbors`，
- export 是否用了 `--embed-e0`。

### `max_ell` 和 `lmax`

- `lmax`：hidden feature angular cutoff。
- `max_ell`：edge spherical-harmonics cutoff。

原生 MACE 常见配置允许 `max_ell >= hidden_lmax`。更高 `max_ell` 会明显增加 product contraction 和 force training 成本。

### 动态 shape

Dynamic AOTI 和 make_fx 对 PyTorch/Inductor 版本敏感。如果 dynamic export 失败，可以尝试：

- `--static-n` 固定 atom count，
- 减少 dynamic 维度，
- 更小/更少的 bucket，
- PyTorch 2.7+，
- 关闭 fusion/autotune 等可选优化。

### MBD / 色散训练不稳定

MBD 训练若 spike 或 NaN，几乎总是以下之一：(1) 从头训而非 warm-start（在已训 backbone 上用
`--resume-checkpoint … --finetune` 加 MBD——见 12.6）；(2) 缺失或调高了梯度裁剪（保持
`--max-grad-norm 10`，要稳就降 LR 或 cutoff）；(3) 单层交互 MBD 在 `lr 1e-3`（warm-start 或降 LR）；
(4) 长的 `--train-makefx-compile` 运行（长训/生产 MBD 用 eager；AOTI 部署不受影响）。极化率求解每步都
保持正定，所以硬 NaN 很少见，出现时通常就是上面几条之一。见 12.6。

## 16. 开发说明

建议遵守：

- 修改角向 basis 或 product backend 前，先跑 MACE parity 测试。
- smoke test 通过不等于 MACE parity 已证明；parity 要跑 MACE-vs-ICTC converter test。
- 修改 checkpoint metadata 后，必须验证 `LAMMPS_MLIAP_MFF.from_checkpoint` strict reload。
- 修改 `angular_basis=e3nn` 后，必须验证 eager forward 和 checkpoint reload，避免 fixed buffer double-fold。

重要文件：

| 文件 | 作用 |
|---|---|
| `mace_ictc/models/pure_cartesian_ictd_fix.py` | 主模型和 product backends。 |
| `mace_ictc/mace_basis.py` | ICTC/e3nn 正交 basis 转换。 |
| `mace_ictc/interfaces/mace_converter.py` | 原生 MACE 到 MACE-ICTC 的权重转换。 |
| `mace_ictc/cli/export_aoti_core.py` | AOTI 导出、cuEq product replacement、export-time angular basis 逻辑。 |
| `mace_ictc/training/makefx_compile.py` | `make_fx` force-step 编译。 |
| `mace_ictc/training/train_loop.py` | Trainer、checkpoint metadata、ScaleShift/E0 loss 逻辑。 |
| `mace_ictc/interfaces/lammps_mliap.py` | 部署端 checkpoint reload 和 wrapper 逻辑。 |
| `mace_ictc/test/test_mace_converter.py` | 原生 MACE 到 MACE-ICTC 的转换 parity 测试。 |
