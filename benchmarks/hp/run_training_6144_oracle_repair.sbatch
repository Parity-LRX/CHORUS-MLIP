#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-6144-oracle-repair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/oracle_repair_6144_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v37"
export SOURCE="$ROOT/source_v20/chorus" HP_SOURCE="$ROOT/source_v20/ictc_o3_hp"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
CACHE="$ROOT/cache/native"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/profile_training_nograph.py" \
  --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
  --atoms 6144 --neighbors 50 --rounds 1 --warmup 1 --repeats 1 \
  --training-species-fusion --radial-coefficient-fusion --phase-coefficient-mixed-fusion \
  --training-mixed-destination --reference-tp-chunk-edges 1024 --reference-save-on-cpu \
  --reference-empty-cache --tag qualification_repaired \
  --out "$ROOT/results/selected_training_6144/n6144/qualification_repaired.json"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/report_selected_training.py" \
  --root "$ROOT/results/selected_training_6144" --sizes 6144 \
  --qualification-name qualification_repaired --out "$ROOT/reports_selected_training_6144"
printf 'DONE\n'
