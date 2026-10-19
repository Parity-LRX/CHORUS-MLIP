#!/usr/bin/env bash
#SBATCH --job-name=chorus-native-training-release
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/native_release_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v13"
export SOURCE="$ROOT/source_v10/chorus" HP_SOURCE="$ROOT/source_v10/ictc_o3_hp"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
PY=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
NATIVE_LIBS="${LD_LIBRARY_PATH:-}"
CUDA_LIBS=""
while IFS= read -r libdir; do CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"; done \
  < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
SIZES=(128 512 2048)
N="${SIZES[$SLURM_ARRAY_TASK_ID]}"
RESULT="$ROOT/results/native_release_v1/n${N}"
mkdir -p "$RESULT"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
native_env() {
  export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE" LD_LIBRARY_PATH="$NATIVE_LIBS"
  CACHE="$ROOT/cache/native"
  export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
  export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
  export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
}
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv
set -x
native_env
"$PY" -m pytest -q "$SOURCE/benchmarks/hp/test_species_training.py" \
  "$SOURCE/benchmarks/hp/test_gate_coefficients.py" "$SOURCE/benchmarks/hp/test_silu_literal.py" \
  "$SOURCE/chorus/test/test_hp_backend.py" "$SOURCE/benchmarks/hp/test_no_cuda_graph.py" \
  --maxfail=1 --junitxml="$RESULT/targeted_tests.xml"
for ROUND in 0 1; do
  ORDER="control fused"
  if [[ "$ROUND" == 1 ]]; then ORDER="fused control"; fi
  for VERSION in $ORDER; do
    native_env
    FLAGS=()
    if [[ "$VERSION" == fused ]]; then FLAGS=(--training-species-fusion --radial-coefficient-fusion); fi
    "$PY" "$TOOLS/profile_training_nograph.py" --checkpoint "$ASSETS/persistent_t1x.pth" \
      --atoms "$N" --neighbors 50 --rounds 1 --warmup 2 --repeats 12 --profile "${FLAGS[@]}" \
      --tag "${VERSION}_r${ROUND}" --out "$RESULT/${VERSION}_r${ROUND}.json"
  done
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
printf 'DONE\n'
