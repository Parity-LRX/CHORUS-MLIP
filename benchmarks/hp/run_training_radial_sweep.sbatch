#!/usr/bin/env bash
#SBATCH --job-name=chorus-live-radial
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/live_radial_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v45"
export SOURCE="$ROOT/source_v25/chorus" HP_SOURCE="$ROOT/source_v25/ictc_o3_hp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TORCH_EXTENSIONS_DIR="$ROOT/cache/native/extensions" FSCETP_ICTD_CACHE_DIR="$ROOT/cache/native/ictd"
export TRITON_CACHE_DIR="$ROOT/cache/native/triton" XDG_CACHE_HOME="$ROOT/cache/native/xdg"
export TMPDIR="$ROOT/cache/tmp" CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest "$SOURCE/chorus/test/test_hp_backend.py" \
  -q -k 'live_radial_tp_all_mixed_targets or high_l_compact_force_training' --maxfail=1
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_mixed_sweep.py" \
  --root "$ROOT" --tools "$TOOLS" --case l3 --atoms 128 --result-set live_radial_qual_v1 \
  --qualify --variants radial32 --scopes persistent final --repeats 1 --stop-on-error
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_mixed_sweep.py" \
  --root "$ROOT" --tools "$TOOLS" --case l3 --atoms 1024 --result-set live_radial_screen_v1 \
  --variants radial radial32 radialg32 --repeats 12 --profile --stop-on-error
printf 'DONE\n'
