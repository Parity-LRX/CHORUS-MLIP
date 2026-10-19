#!/usr/bin/env bash
#SBATCH --job-name=chorus-native-nograph
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018/logs/native_%j.log
set -euo pipefail
ulimit -c 0
ROOT=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
cd "$ROOT"
SOURCE="${SOURCE:?immutable CHORUS source required}"
HP_SOURCE="${HP_SOURCE:?immutable ICTC source required}"
CASE="${CASE:-l3}"
SCOPE="${SCOPE:-persistent}"
RUN="${RUN:-native_only_v16}"
read -r -a SIZES <<< "${ATOMS:-128 256 1024}"
TOOLS="$SOURCE/benchmarks/hp"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
export SOURCE HP_SOURCE CHORUS_DISABLE_ABI_EXTENSIONS=1
export PYTHONPATH="$TOOLS/runtime_import_guard:$TOOLS:$OVERLAY:$SOURCE:$HP_SOURCE"
CUDA_LIBS=""
while IFS= read -r libdir; do CUDA_LIBS="${CUDA_LIBS:+$CUDA_LIBS:}$libdir"; done < <(
  find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE="$ROOT/cache/$RUN/$CASE/$SCOPE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" PYTHONPYCACHEPREFIX="$CACHE/pycache"
export TMPDIR="$CACHE/tmp" CUDA_CACHE_PATH="$CACHE/cuda_driver"
OUT="$ROOT/native_only/$RUN/$CASE"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME/torch/kernels" "$OUT"
PY=/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python
INITIAL=/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations
MATCH=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/parameter_matched_v1
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
set -x
"$PY" "$TOOLS/profile_inference_native_nograph.py" --case "$CASE" --scope "$SCOPE" \
  --initialization "$INITIAL/$CASE/chorus_${SCOPE}_${CASE}_initialization.pt" \
  --atoms "${SIZES[@]}" --out "$OUT/$SCOPE.native.json"
"$PY" "$TOOLS/profile_inference_dpa_nograph.py" --config "$MATCH/$CASE/$SCOPE/input.json" \
  --checkpoint "$MATCH/$CASE/$SCOPE/initialization.pt" --atoms "${SIZES[@]}" \
  --out "$OUT/$SCOPE.dpa.json"
printf 'DONE\n'
