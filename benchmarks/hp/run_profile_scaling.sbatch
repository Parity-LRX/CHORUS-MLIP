#!/usr/bin/env bash
#SBATCH --job-name=chorus-hp-profile
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/profile_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:-$ROOT/source/chorus}"
HP_SOURCE="${HP_SOURCE:-$ROOT/source/ictc_o3_hp}"
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
TAG="${CACHE_SET:-${RESULT_SET:-profile}_$SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="$ROOT/cache/$TAG/xdg"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/$TAG/extensions"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/$TAG/ictd"
export TRITON_CACHE_DIR="$ROOT/cache/$TAG/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/$TAG/pycache"
export TMPDIR="$ROOT/cache/$TAG/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
POLICY=reference
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then POLICY=generated; fi
set -x
for N in ${SIZES:-128 1024 4096}; do
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
    "$SOURCE/benchmarks/hp/profile_scaling.py" \
    --checkpoint "${CHECKPOINT:-/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth}" \
    --atoms "$N" --policy "$POLICY" --sc on --checkpoint-sc --profile \
    --topology-backend device ${CAPTURE_FLAG:-} ${STREAM_FLAG:-} \
    --out "$ROOT/results/${RESULT_SET:-profile}/chorus_${POLICY}_n$N.json"
done
printf 'DONE\n'
