#!/usr/bin/env bash
#SBATCH --job-name=chorus-literal-activation-gradients
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/gradient_followup_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v7"
export SOURCE="$ROOT/source_v6/chorus" HP_SOURCE="$ROOT/source_v6/ictc_o3_hp"
CACHE="$ROOT/cache/native"
export PYTHONPATH="$TOOLS:$ROOT/tools_v5:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
N=128
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then N=512; fi
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  "$SOURCE/benchmarks/hp/test_silu_literal.py" --maxfail=1 \
  --junitxml="$ROOT/results/silu_literal_${SLURM_JOB_ID}.xml"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/audit_training_gradients.py" \
  --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
  --atoms "$N" --neighbors 50 --reference-repeats 3 --candidate-repeats 5 \
  --modes generated stable_csr original_activation original_harmonics original_phase original_sc \
    original_maps stable_original_maps stable_original_activation original_activation_harmonics \
    stable_original_activation_harmonics \
  --out "$ROOT/results/gradient_audit_literal_v2/n${N}.json"
printf 'DONE\n'
