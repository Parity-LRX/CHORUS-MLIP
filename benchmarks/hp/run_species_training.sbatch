#!/usr/bin/env bash
#SBATCH --job-name=chorus-live-species-training
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/species_training_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v12"
export SOURCE="$ROOT/source_v9/chorus" HP_SOURCE="$ROOT/source_v9/ictc_o3_hp"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
RESULT="$ROOT/results/species_training_v1/n512"
mkdir -p "$RESULT"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
CACHE="$ROOT/cache/native"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  "$SOURCE/benchmarks/hp/test_species_training.py" "$SOURCE/benchmarks/hp/test_gate_coefficients.py" \
  "$SOURCE/benchmarks/hp/test_silu_literal.py" "$SOURCE/chorus/test/test_hp_backend.py" \
  "$SOURCE/benchmarks/hp/test_no_cuda_graph.py" --maxfail=1 --junitxml="$RESULT/targeted_tests.xml"
for MODE in control species species_radial; do
  FLAGS=()
  if [[ "$MODE" != control ]]; then FLAGS+=(--training-species-fusion); fi
  if [[ "$MODE" == species_radial ]]; then FLAGS+=(--radial-coefficient-fusion); fi
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/profile_training_nograph.py" \
    --checkpoint "$ASSETS/persistent_t1x.pth" --atoms 512 --neighbors 50 \
    --rounds 2 --warmup 2 --repeats 12 --profile "${FLAGS[@]}" \
    --tag "$MODE" --out "$RESULT/${MODE}.json"
done
printf 'DONE\n'
