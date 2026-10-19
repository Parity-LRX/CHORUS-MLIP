#!/usr/bin/env bash
#SBATCH --job-name=chorus-repeat-gradient-diagnostic
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/repeat_diagnostic_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v9"
export SOURCE="$ROOT/source_v6/chorus" HP_SOURCE="$ROOT/source_v6/ictc_o3_hp"
CACHE="$ROOT/cache/native"
export PYTHONPATH="$TOOLS:$ROOT/tools_v5:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
extra=()
tag=ordinary
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then
  extra=(--deterministic-diagnostic)
  tag=deterministic
fi
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/audit_training_gradients.py" \
  --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
  --atoms 128 --neighbors 50 --reference-repeats 16 --candidate-repeats 5 \
  --modes reference generated stable_csr stable_original_activation_harmonics \
    stable_original_activation_harmonics_sc stable_original_activation_harmonics_sc_phase \
    only_messages only_sc only_density only_phase only_channel \
  "${extra[@]}" --out "$ROOT/results/gradient_repeat_diagnostic/${tag}.json"
printf 'DONE\n'
