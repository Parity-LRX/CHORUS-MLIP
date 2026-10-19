#!/usr/bin/env bash
#SBATCH --job-name=chorus-final-large50
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/final_large50_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set immutable CHORUS snapshot}"
HP_SOURCE="${HP_SOURCE:?set immutable ICTC-O3-HP snapshot}"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
CHECKPOINT="${CHECKPOINT:-/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth}"
MODEL_LABEL="${MODEL_LABEL:-final}"
RESULT="$ROOT/results/${RESULT_SET:-final_large50_v51}"
mkdir -p "$RESULT"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export DP_COMPILE_INFER=1 DP_TRITON_INFER=2 DP_TF32_INFER=0 DP_CUTE_INFER=0
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"
done < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
NATIVE_LIBS="${LD_LIBRARY_PATH:-}"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
for BOUNDARY in full lower; do
  CACHE="$ROOT/cache/dpa_fast_t2"
  export PYTHONPATH="$OVERLAY:$SOURCE"
  export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:$NATIVE_LIBS"
  export XDG_CACHE_HOME="$CACHE/xdg" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
  export TRITON_CACHE_DIR="$CACHE/triton" PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
  /public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python \
    "$SOURCE/benchmarks/hp/profile_dpa_boundary.py" --atoms 10000 --neighbors 50 \
    --boundary "$BOUNDARY" --checkpoint "$ASSETS/dpa4/c32/model.pt" \
    --config "$ASSETS/dpa4/c32/input.json" --out "$RESULT/dpa4_${BOUNDARY}_n10000_k50.json"
done
CACHE="$ROOT/cache/native_v1_1"
export PYTHONPATH="$SOURCE:$HP_SOURCE" LD_LIBRARY_PATH="$NATIVE_LIBS"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
FAILURES=0
for MODE in ${MODES:-fastest}; do
  FLAGS=()
  if [[ "$MODE" == radial_jvp ]]; then FLAGS=(--radial-jvp --radial-tile-edges 65536); fi
  if [[ -n "${EXECUTION_EXPERIMENT:-}" ]]; then FLAGS+=(--execution-experiment "$EXECUTION_EXPERIMENT"); fi
  if [[ -n "${REFERENCE_CACHE:-}" ]]; then FLAGS+=(--reference-cache "$REFERENCE_CACHE"); fi
  if [[ "${USE_MODE_EXPERIMENT:-0}" == 1 ]]; then FLAGS+=(--execution-experiment "$MODE"); fi
  if ! /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
    "$SOURCE/benchmarks/hp/profile_scaling.py" --atoms 10000 --neighbors 50 \
    --checkpoint "$CHECKPOINT" --policy generated --sc on --checkpoint-sc \
    --topology-backend device --capture --virial --allow-reference-oom --profile "${FLAGS[@]}" \
    --out "$RESULT/chorus_${MODEL_LABEL}_${MODE}_n10000_k50.json"; then
    FAILURES=$((FAILURES+1))
  fi
done
if (( FAILURES )); then printf 'FAILED qualification cases: %s\n' "$FAILURES"; exit 1; fi
printf 'DONE\n'
