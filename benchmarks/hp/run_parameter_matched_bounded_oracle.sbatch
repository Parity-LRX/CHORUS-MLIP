#!/usr/bin/env bash
#SBATCH --job-name=chorus-bounded-oracle
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/bounded_oracle_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v77"
export SOURCE="$ROOT/source_v40/chorus" HP_SOURCE="$ROOT/source_v40/ictc_o3_hp"
CACHE="$ROOT/cache/native"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  "$TOOLS/test_checkpoint_reference_tp.py" "$TOOLS/test_parameter_match_report.py" \
  --maxfail=1 --junitxml="$ROOT/results/bounded_oracle_tests_${SLURM_JOB_ID}.xml"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/qualify_parameter_matched_max.py" \
  --root "$ROOT" --tools "$TOOLS" --scopes persistent --atoms 512 \
  --result-set parameter_matched_oracle_v3 \
  --reference-tp-chunk-edges 256 --reference-sc-chunk-nodes 128 \
  --reference-sc-chunk-channels 64
printf 'DONE\n'
