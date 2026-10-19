#!/usr/bin/env bash
#SBATCH --job-name=chorus-aoti-nograph
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018/logs/export_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
cd "$ROOT"
ulimit -c 0
SOURCE="${SOURCE:?set immutable CHORUS source snapshot}"
HP_SOURCE="${HP_SOURCE:?set immutable ICTC-O3-HP source snapshot}"
TOOLS="$SOURCE/benchmarks/hp"
RUN="${RUN:-prepared_v1}"
CASE="${CASE:-l3}"
SCOPE="${SCOPE:-persistent}"
export SOURCE HP_SOURCE PYTHONPATH="$TOOLS:$SOURCE:$HP_SOURCE"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE="$ROOT/cache/$RUN/$CASE/$SCOPE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" PYTHONPYCACHEPREFIX="$CACHE/pycache"
export TMPDIR="$CACHE/tmp" CUDA_CACHE_PATH="$CACHE/cuda_driver"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME/torch/kernels" "$ROOT/packages/$RUN/$CASE"
PY=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
if [[ "${TORCH211:-0}" == 1 ]]; then
  ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
  OVERLAY="$ASSETS/dpa4_torch211_overlay"
  CUDA_LIBS=""
  while IFS= read -r libdir; do
    CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"
  done < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
  export CHORUS_DISABLE_ABI_EXTENSIONS=1
  export PYTHONPATH="$TOOLS/runtime_import_guard:$TOOLS:$OVERLAY:$SOURCE:$HP_SOURCE"
  export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
  PY=/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python
fi
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
set -x
if [[ "${UNIT:-0}" == 1 ]]; then
  "$PY" -m pytest "$SOURCE/chorus/test/test_hp_export.py" -q -x \
    -k "${UNIT_SELECTION:-prepared}" \
    --junitxml="$ROOT/packages/$RUN/unit_$SLURM_JOB_ID.xml"
  if [[ "${UNIT_ONLY:-0}" == 1 ]]; then exit 0; fi
fi
PACKAGE="$ROOT/packages/$RUN/$CASE/$SCOPE.pt2"
"$PY" "$TOOLS/export_inference_aoti_nograph.py" --case "$CASE" --scope "$SCOPE" \
  --initialization "/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations/$CASE/chorus_${SCOPE}_${CASE}_initialization.pt" \
  --backend "${BACKEND:-prepared}" --message-schedule "${MESSAGE_SCHEDULE:-atomic}" --out "$PACKAGE"
"$PY" "$TOOLS/load_inference_aoti_nograph.py" --package "$PACKAGE" \
  --out "$PACKAGE.standalone.json"
if [[ -n "${BASELINE_RUN:-}" ]]; then
  "$PY" "$TOOLS/load_inference_aoti_nograph.py" \
    --package "$ROOT/packages/$BASELINE_RUN/$CASE/$SCOPE.pt2" \
    --out "$PACKAGE.aoti_baseline.json" --profile
fi
if [[ "${NATIVE:-0}" == 1 ]]; then
  "$PY" "$TOOLS/profile_inference_native_nograph.py" --package "$PACKAGE" \
    --out "$PACKAGE.native.json"
fi
if [[ "${ATEN_CONTROL:-0}" == 1 ]]; then
  "$PY" "$TOOLS/load_inference_aoti_nograph.py" \
    --package "$ROOT/packages/aten_v3/l3/final.pt2" --out "$PACKAGE.aten_control.json"
fi
if [[ "${DPA:-0}" == 1 ]]; then
  ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
  OVERLAY="$ASSETS/dpa4_torch211_overlay"
  CUDA_LIBS=""
  while IFS= read -r libdir; do
    CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"
  done < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
  export PYTHONPATH="$TOOLS:$OVERLAY:$SOURCE"
  export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
  export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
  MATCH=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/parameter_matched_v1/$CASE/$SCOPE
  /public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python \
    "$TOOLS/profile_inference_dpa_nograph.py" --config "$MATCH/input.json" \
    --checkpoint "$MATCH/initialization.pt" --out "$PACKAGE.dpa.json"
fi
printf 'DONE\n'
