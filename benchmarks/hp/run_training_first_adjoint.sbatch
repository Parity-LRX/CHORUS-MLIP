#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-first-adjoint
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/first_adjoint_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v26"
RESULT="$ROOT/results/first_adjoint_v1/n2048"
mkdir -p "$RESULT"
CACHE="$ROOT/cache/native"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export SOURCE="$ROOT/source_v16/chorus" HP_SOURCE="$ROOT/source_v16/ictc_o3_hp"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  "$SOURCE/benchmarks/hp/test_coherent_training.py" --maxfail=1 --junitxml="$RESULT/targeted_tests.xml"
for TAG in old_r0 empty split split_phase old_r1; do
  VERSION=source_v16
  FLAGS=()
  if [[ "$TAG" == old* ]]; then VERSION=source_v15; fi
  if [[ "$TAG" == split* ]]; then FLAGS+=(--training-split-first); fi
  if [[ "$TAG" == split_phase ]]; then FLAGS+=(--phase-coefficient-mixed-fusion); fi
  export SOURCE="$ROOT/$VERSION/chorus" HP_SOURCE="$ROOT/$VERSION/ictc_o3_hp"
  export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/profile_training_nograph.py" \
    --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
    --atoms 2048 --neighbors 50 --rounds 1 --warmup 2 --repeats 12 \
    --training-species-fusion --radial-coefficient-fusion "${FLAGS[@]}" \
    --tag "$TAG" --out "$RESULT/${TAG}.json"
done
printf 'DONE\n'
