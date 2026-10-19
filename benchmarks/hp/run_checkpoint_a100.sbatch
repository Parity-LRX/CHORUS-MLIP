#!/usr/bin/env bash
#SBATCH --job-name=chorus-hp-ckpt
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_o3_hp_20261017/logs/checkpoint_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_o3_hp_20261017
export PYTHONPATH="$ROOT/source_v3/chorus:$ROOT/source/ictc_o3_hp"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XDG_CACHE_HOME="$ROOT/cache/checkpoint_$SLURM_ARRAY_TASK_ID/xdg"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/checkpoint_$SLURM_ARRAY_TASK_ID/ictd"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/checkpoint_$SLURM_ARRAY_TASK_ID/extensions"
export TRITON_CACHE_DIR="$ROOT/cache/checkpoint_$SLURM_ARRAY_TASK_ID/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/checkpoint_$SLURM_ARRAY_TASK_ID/pycache"
export TMPDIR="$ROOT/cache/checkpoint_$SLURM_ARRAY_TASK_ID/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
case "$SLURM_ARRAY_TASK_ID" in
  0) CKPT=/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth; TAG=final ;;
  1) CKPT=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth; TAG=persistent ;;
esac
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$ROOT/tools/qualify_checkpoint.py" \
  --checkpoint "$CKPT" \
  --data-h5 /public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728/transition1x/chorus_reaction_id_50k_seed20260616/processed_val.h5 \
  --frames 16 --out "$ROOT/results/a100_checkpoint_heldout_$TAG.json"
printf 'DONE\n'
