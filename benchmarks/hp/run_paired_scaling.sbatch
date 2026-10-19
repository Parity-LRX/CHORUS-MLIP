#!/usr/bin/env bash
#SBATCH --job-name=chorus-hp-paired
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/paired_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:-$ROOT/native_v4/chorus}"
HP_SOURCE="${HP_SOURCE:-$ROOT/native_v4/ictc_o3_hp}"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export DP_COMPILE_INFER=1 DP_TRITON_INFER="${TRITON_LEVEL:-2}" DP_TF32_INFER=0 DP_CUTE_INFER=0
RESULT_SET="${RESULT_SET:-paired_v4}"
RESULT="$ROOT/results/$RESULT_SET"
mkdir -p "$RESULT"
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"
done < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
DPA_LIBS="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
NATIVE_LIBS="${LD_LIBRARY_PATH:-}"
SIZES=(128 256 512 1024 2048 4096)
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
set -x
if [[ "${ONE_SIZE_PER_TASK:-0}" == 1 ]]; then
  QUALIFIED_SIZES=(128 1024 4096)
  SELECTED="${QUALIFIED_SIZES[$SLURM_ARRAY_TASK_ID]}"
else
  SELECTED="${TEST_SIZES:-${SIZES[$((2*SLURM_ARRAY_TASK_ID))]} ${SIZES[$((2*SLURM_ARRAY_TASK_ID+1))]}}"
fi
for N in $SELECTED; do
  for BOUNDARY in full lower; do
    export PYTHONPATH="$OVERLAY:$SOURCE"
    export LD_LIBRARY_PATH="$DPA_LIBS"
    DPA_CACHE="$ROOT/cache/dpa_fast_t$DP_TRITON_INFER"
    export XDG_CACHE_HOME="$DPA_CACHE/xdg"
    export TORCHINDUCTOR_CACHE_DIR="$DPA_CACHE/inductor"
    export TRITON_CACHE_DIR="$DPA_CACHE/triton"
    export PYTHONPYCACHEPREFIX="$DPA_CACHE/pycache"
    export TMPDIR="$DPA_CACHE/tmp"
    /public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python \
      "$SOURCE/benchmarks/hp/profile_dpa_boundary.py" --atoms "$N" --boundary "$BOUNDARY" \
      --checkpoint "$ASSETS/dpa4/c32/model.pt" --config "$ASSETS/dpa4/c32/input.json" \
      --out "$RESULT/dpa4_${BOUNDARY}_n$N.json"
  done
  export PYTHONPATH="$SOURCE:$HP_SOURCE"
  export LD_LIBRARY_PATH="$NATIVE_LIBS"
  export XDG_CACHE_HOME="$ROOT/cache/native_v1_1/xdg"
  export TORCH_EXTENSIONS_DIR="$ROOT/cache/native_v1_1/extensions"
  export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/native_v1_1/ictd"
  export TRITON_CACHE_DIR="$ROOT/cache/native_v1_1/triton"
  export PYTHONPYCACHEPREFIX="$ROOT/cache/native_v1_1/pycache"
  export TMPDIR="$ROOT/cache/native_v1_1/tmp"
  for SCOPE in final persistent; do
    CHECKPOINT=/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth
    if [[ "$SCOPE" == persistent ]]; then CHECKPOINT="$ASSETS/persistent_t1x.pth"; fi
    /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
      "$SOURCE/benchmarks/hp/profile_scaling.py" --atoms "$N" --checkpoint "$CHECKPOINT" \
      --policy generated --sc on --checkpoint-sc --topology-backend device --capture --virial \
      ${RADIAL_FLAGS:-} \
      --out "$RESULT/chorus_${SCOPE}_n$N.json"
  done
  if [[ -n "${BASELINE_SOURCE:-}" ]]; then
    export PYTHONPATH="$BASELINE_SOURCE:${BASELINE_HP_SOURCE:?set prior ICTC snapshot}"
    /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
      "$BASELINE_SOURCE/benchmarks/hp/profile_scaling.py" --atoms "$N" \
      --checkpoint "$ASSETS/persistent_t1x.pth" \
      --policy generated --sc on --checkpoint-sc --topology-backend device --capture --virial \
      --out "$RESULT/chorus_previous_persistent_n$N.json"
  fi
done
printf 'DONE\n'
