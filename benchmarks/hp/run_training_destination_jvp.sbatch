#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-destination-jvp
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/destination_jvp_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v29"
RESULT="$ROOT/results/destination_jvp_v1/n2048"
mkdir -p "$RESULT"
CACHE="$ROOT/cache/native"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver"
export SOURCE="$ROOT/source_v19/chorus" HP_SOURCE="$ROOT/source_v19/ictc_o3_hp"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  "$SOURCE/benchmarks/hp/test_coherent_training.py" --maxfail=1 --junitxml="$RESULT/targeted_tests.xml"
for TAG in base_r0 destination destination_phase destination_grouped base_r1; do
  FLAGS=()
  if [[ "$TAG" == destination* ]]; then FLAGS+=(--training-mixed-destination); fi
  if [[ "$TAG" == destination_phase ]]; then FLAGS+=(--phase-coefficient-mixed-fusion); fi
  if [[ "$TAG" == destination_grouped ]]; then FLAGS+=(--training-mixed-grouped); fi
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/profile_training_nograph.py" \
    --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
    --atoms 2048 --neighbors 50 --rounds 1 --warmup 2 --repeats 12 \
    --training-species-fusion --radial-coefficient-fusion "${FLAGS[@]}" \
    --tag "$TAG" --out "$RESULT/${TAG}.json"
done
printf 'DONE\n'
