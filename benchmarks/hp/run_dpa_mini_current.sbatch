#!/usr/bin/env bash
#SBATCH --job-name=chorus-hp-dpa-mini
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/dpa_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
export PYTHONPATH="$OVERLAY:$ROOT/source/chorus"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export DP_COMPILE_INFER=1 DP_TRITON_INFER=2 DP_TF32_INFER=0 DP_CUTE_INFER=0
export XDG_CACHE_HOME="$ROOT/cache/dpa/xdg"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/dpa/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/dpa/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/dpa/pycache"
export TMPDIR="$ROOT/cache/dpa/tmp"
mkdir -p "$TMPDIR"
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"
done < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
set -x
date -Is
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
for N in 128 256 512 1024 2048 4096; do
  /public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python \
    "$ROOT/source/chorus/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py" \
    --engine dpa4 --dpa-amp off --tasks inference --sizes "$N" \
    --config "$ASSETS/dpa4/c32/input.json" --checkpoint "$ASSETS/dpa4/c32/model.pt" \
    --output "$ROOT/results/dpa4_mini_n$N.json"
done
printf 'DONE\n'
