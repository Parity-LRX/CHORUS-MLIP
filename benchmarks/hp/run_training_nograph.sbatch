#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-nograph
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/paired_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="${TOOLS:-$ROOT/tools_v5}"
CANDIDATE="${CANDIDATE:-source_v5}"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
PY=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
NATIVE_LIBS="${LD_LIBRARY_PATH:-}"
CUDA_LIBS=""
while IFS= read -r libdir; do CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"; done \
  < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
N=128
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then N=512; fi
RESULT="$ROOT/results/${RESULT_SET:-no_graph_v5}/n${N}"
mkdir -p "$RESULT"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv
set -x
for ROUND in 0 1; do
  ORDER="source_v1 $CANDIDATE"
  if [[ "$ROUND" == 1 ]]; then ORDER="$CANDIDATE source_v1"; fi
  for VERSION in $ORDER; do
    FLAGS=()
    if [[ "$VERSION" == "$CANDIDATE" ]]; then FLAGS=(--phase-context-mixed-fusion); fi
    export SOURCE="$ROOT/$VERSION/chorus" HP_SOURCE="$ROOT/$VERSION/ictc_o3_hp"
    export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
    export LD_LIBRARY_PATH="$NATIVE_LIBS"
    CACHE="$ROOT/cache/native"
    export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
    export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
    export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
    "$PY" "$TOOLS/profile_training_nograph.py" --checkpoint "$ASSETS/persistent_t1x.pth" \
      --atoms "$N" --neighbors 50 --rounds 1 --warmup 2 --repeats 12 --profile "${FLAGS[@]}" \
      --tag "${VERSION}_r${ROUND}" --out "$RESULT/${VERSION}_r${ROUND}.json"
  done
  export SOURCE="$ROOT/source_v1/chorus"
  export PYTHONPATH="$TOOLS:$OVERLAY:$SOURCE"
  export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:$NATIVE_LIBS"
  CACHE="$ROOT/cache/dpa"
  export XDG_CACHE_HOME="$CACHE/xdg" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
  export TRITON_CACHE_DIR="$CACHE/triton" PYTHONPYCACHEPREFIX="$CACHE/pycache"
  /public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python \
    "$TOOLS/profile_training_dpa_nograph.py" --atoms "$N" --neighbors 50 \
    --checkpoint "$ASSETS/dpa4/c32/model.pt" --config "$ASSETS/dpa4/c32/input.json" \
    --out "$RESULT/dpa_r${ROUND}.json"
done
if [[ "$N" == 512 ]]; then
  export SOURCE="$ROOT/$CANDIDATE/chorus" HP_SOURCE="$ROOT/$CANDIDATE/ictc_o3_hp"
  export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE" LD_LIBRARY_PATH="$NATIVE_LIBS"
  CACHE="$ROOT/cache/native"
  export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
  export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
  export PYTHONPYCACHEPREFIX="$CACHE/pycache"
  "$PY" "$TOOLS/profile_training_nograph.py" --checkpoint "$ASSETS/persistent_t1x.pth" \
    --variable-atoms 128 256 512 --neighbors 50 --rounds 2 --warmup 2 --repeats 12 \
    --phase-context-mixed-fusion \
    --tag variable_n --out "$RESULT/variable_n.json"
fi
printf 'DONE\n'
