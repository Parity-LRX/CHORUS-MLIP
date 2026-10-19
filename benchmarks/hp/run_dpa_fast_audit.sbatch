#!/usr/bin/env bash
#SBATCH --job-name=dpa4-fast-audit
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=1-2%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/dpa_fast_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:-$ROOT/native_v7/chorus}"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
export PYTHONPATH="$OVERLAY:$SOURCE"
export DP_COMPILE_INFER=1 DP_TRITON_INFER="$SLURM_ARRAY_TASK_ID" DP_TF32_INFER=0 DP_CUTE_INFER=0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export XDG_CACHE_HOME="$ROOT/cache/dpa_fast_t$SLURM_ARRAY_TASK_ID/xdg"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/dpa_fast_t$SLURM_ARRAY_TASK_ID/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/dpa_fast_t$SLURM_ARRAY_TASK_ID/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/dpa_fast_t$SLURM_ARRAY_TASK_ID/pycache"
export TMPDIR="$ROOT/cache/dpa_fast_t$SLURM_ARRAY_TASK_ID/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME/torch/kernels"
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"
done < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
scontrol show job -d "$SLURM_JOB_ID"
set -x
for N in ${SIZES:-128 4096}; do
  for BOUNDARY in ${BOUNDARIES:-full lower}; do
    /public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python \
      "$SOURCE/benchmarks/hp/profile_dpa_boundary.py" --atoms "$N" --boundary "$BOUNDARY" \
      --checkpoint "$ASSETS/dpa4/c32/model.pt" --config "$ASSETS/dpa4/c32/input.json" \
      --profile ${CAPTURE_FLAG:-} \
      --out "$ROOT/results/${RESULT_SET:-dpa_fast}_t$SLURM_ARRAY_TASK_ID/dpa4_${BOUNDARY}_n$N.json"
  done
done
printf 'DONE\n'
