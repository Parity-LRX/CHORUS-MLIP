#!/usr/bin/env bash
#SBATCH --job-name=chorus-large-training-profile
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/large_profile_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v40"
export SOURCE="$ROOT/source_v23/chorus" HP_SOURCE="$ROOT/source_v23/ictc_o3_hp"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
CACHE="$ROOT/cache/native"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
CASES=(l3 d4 l3c256d4)
SIZES=(4096 4096 1024)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
N="${SIZES[$SLURM_ARRAY_TASK_ID]}"
INITIAL=/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations
OUT="$ROOT/results/large_training_profiles_v1/$CASE"
mkdir -p "$OUT"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$TOOLS/profile_training_architecture_nograph.py" --architecture-case "$CASE" --scope persistent \
  --checkpoint "$INITIAL/$CASE/chorus_persistent_${CASE}_initialization.pt" \
  --atoms "$N" --neighbors 50 --rounds 1 --warmup 2 --repeats 5 --measurement-only --profile \
  --training-species-fusion --radial-coefficient-fusion --phase-coefficient-mixed-fusion \
  --training-mixed-destination --training-destination-first 1 \
  --tag "${CASE}_profile" --out "$OUT/persistent_n${N}.json"
printf 'DONE\n'
