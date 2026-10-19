#!/usr/bin/env bash
#SBATCH --job-name=dpa4-compile-training
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/training_dpa_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set benchmark snapshot}"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
export PYTHONPATH="$OVERLAY:$SOURCE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"
done < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
CACHE="$ROOT/cache/dpa_training_v1"
mkdir -p "$CACHE/tmp" "$CACHE/inductor" "$CACHE/triton" "$CACHE/pycache" "$CACHE/xdg"
export XDG_CACHE_HOME="$CACHE/xdg" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export TRITON_CACHE_DIR="$CACHE/triton" PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
for N in ${TEST_SIZES:-128 512}; do
  /public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python \
    "$SOURCE/benchmarks/hp/profile_training_dpa.py" --atoms "$N" --neighbors 50 --boundary lower \
    --checkpoint "$ASSETS/dpa4/c32/model.pt" --config "$ASSETS/dpa4/c32/input.json" \
    --out "$ROOT/results/${RESULT_SET:-training_dpa_v34}/dpa4_lower_n${N}_k50.json"
done
printf 'DONE\n'
