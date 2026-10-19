#!/usr/bin/env bash
#SBATCH --job-name=chorus-radial-ab
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/radial_ab_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set immutable CHORUS snapshot}"
HP_SOURCE="${HP_SOURCE:?set immutable ICTC-O3-HP snapshot}"
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
CACHE="$ROOT/cache/native_v1_1"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
set -x
for N in ${SIZES:-4096}; do
  for TILE in ${TILES:-off 4096 8192 16384}; do
    EXTRA=(--no-radial-fusion)
    if [[ "$TILE" != off ]]; then
      EXTRA=(--radial-fusion --radial-tile-edges "$TILE" ${GATE_PIPELINE_FLAG:-} ${RADIAL_JVP_FLAG:-})
    fi
    /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
      "$SOURCE/benchmarks/hp/profile_scaling.py" \
      --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
      --atoms "$N" --policy generated --sc on --checkpoint-sc --profile \
      --topology-backend device --capture --virial "${EXTRA[@]}" \
      --out "$ROOT/results/${RESULT_SET:-radial_tiles_v23}/persistent_tile${TILE}_n${N}.json"
  done
done
printf 'DONE\n'
