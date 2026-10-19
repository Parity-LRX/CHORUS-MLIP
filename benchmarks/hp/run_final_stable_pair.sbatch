#!/usr/bin/env bash
#SBATCH --job-name=chorus-final-stable-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/final_stable_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
SOURCE="${SOURCE:?immutable source required}"
HP_SOURCE="${HP_SOURCE:?immutable mother source required}"
TOOLS="${TOOLS:?versioned benchmark tools required}"
export SOURCE HP_SOURCE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$TOOLS/run_capacity_pair.py" --root "$ROOT" --source "$SOURCE" --hp-source "$HP_SOURCE" \
  --scope final --atoms 32768 --dpa-reference-atoms 1024 \
  --tiles 65536 --tag "${RESULT_TAG:-final_stable_selected_v19}" --experiments none_forcecsr \
  --native-wrapper "$TOOLS/profile_final_reference_maps.py"
printf 'DONE\n'
