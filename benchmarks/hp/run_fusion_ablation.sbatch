#!/usr/bin/env bash
#SBATCH --job-name=chorus-fusion-ab
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/fusion_ab_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
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
for TAG in ${SNAPSHOTS:-native_v10 native_v11}; do
  export PYTHONPATH="$ROOT/$TAG/chorus:$ROOT/$TAG/ictc_o3_hp"
  for N in ${SIZES:-4096}; do
    /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
      "$ROOT/$TAG/chorus/benchmarks/hp/profile_scaling.py" \
      --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
      --atoms "$N" --policy generated --sc on --checkpoint-sc --profile \
      --topology-backend device --capture --virial \
      ${RADIAL_FLAG:-} \
      --out "$ROOT/results/${RESULT_SET:-fusion_angular_v11}/persistent_${TAG}_n$N.json"
  done
done
printf 'DONE\n'
