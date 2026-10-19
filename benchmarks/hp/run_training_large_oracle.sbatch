#!/usr/bin/env bash
#SBATCH --job-name=chorus-large-oracle
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/large_oracle_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v51"
export SOURCE="$ROOT/source_v29/chorus" HP_SOURCE="$ROOT/source_v29/ictc_o3_hp"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TORCH_EXTENSIONS_DIR="$ROOT/cache/native/extensions" FSCETP_ICTD_CACHE_DIR="$ROOT/cache/native/ictd"
export TRITON_CACHE_DIR="$ROOT/cache/native/triton" XDG_CACHE_HOME="$ROOT/cache/native/xdg"
export TMPDIR="$ROOT/cache/tmp" CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CASES=(l3 l3c192d3 l3c256d4)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"
OUT="$ROOT/results/large_oracle_v1/$CASE/n1024"
mkdir -p "$OUT"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
status=0
for scope in persistent final; do
  test ! -e "$OUT/$scope.json"
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/profile_training_architecture_nograph.py" \
    --architecture-case "$CASE" --scope "$scope" \
    --checkpoint "/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations/$CASE/chorus_${scope}_${CASE}_initialization.pt" \
    --atoms 1024 --neighbors 50 --rounds 1 --warmup 1 --repeats 1 \
    --training-species-fusion --radial-coefficient-fusion --phase-coefficient-mixed-fusion \
    --training-mixed-destination --training-destination-first 1 --training-mixed-cache 16 \
    --parameter-targets-only --training-terminal-fusion \
    --reference-tp-chunk-edges 1024 --reference-save-on-cpu --reference-empty-cache \
    --tag "large_oracle_${CASE}_${scope}" --out "$OUT/$scope.json" || status=1
done
if [[ "$status" == 0 ]]; then printf 'DONE\n'; fi
exit "$status"
