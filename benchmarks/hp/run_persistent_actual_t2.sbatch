#!/usr/bin/env bash
#SBATCH --job-name=chorus-persistent-actual-t2
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018/logs/persistent_t2_%j.log
set -euo pipefail
ulimit -c 0
ROOT=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
SOURCE="${SOURCE:?immutable CHORUS source required}"
HP_SOURCE="${HP_SOURCE:?immutable ICTC source required}"
TOOLS="$SOURCE/benchmarks/hp"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
export SOURCE HP_SOURCE CHORUS_DISABLE_ABI_EXTENSIONS=1
export PYTHONPATH="$TOOLS/runtime_import_guard:$TOOLS:$OVERLAY:$SOURCE:$HP_SOURCE"
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+$CUDA_LIBS:}$libdir"
done < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE="$ROOT/cache/persistent_actual_t2_v14"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" PYTHONPYCACHEPREFIX="$CACHE/pycache"
export TMPDIR="$CACHE/tmp" CUDA_CACHE_PATH="$CACHE/cuda_driver"
OUT="$ROOT/followup/persistent_actual_t2_v14"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME/torch/kernels"
mkdir "$OUT"
PY=/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python
PACKAGE="$ROOT/packages/prepared_t211_v7/l3c256d4/persistent.pt2"
MATCH=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/parameter_matched_v1/l3c256d4/final
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
set -x
# The package supplies original-formula reference tensors only. Native execution
# uses the current fused backend and preserves the Persistent checkpoint.
"$PY" "$TOOLS/profile_inference_native_nograph.py" --package "$PACKAGE" \
  --out "$OUT/persistent_native.json"
# This existing C96 checkpoint has the same lmax and four interaction layers.
# Its smaller parameter count is reported explicitly, not treated as matched.
"$PY" "$TOOLS/profile_inference_dpa_nograph.py" --config "$MATCH/input.json" \
  --checkpoint "$MATCH/initialization.pt" --atoms 1024 --out "$OUT/dpa_c96.json"
printf 'DONE\n'
