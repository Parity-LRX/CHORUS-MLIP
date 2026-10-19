#!/usr/bin/env bash
#SBATCH --job-name=chorus-hp-native-ckpt
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/native_qualification_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="$ROOT/native_v8/chorus"
export PYTHONPATH="$SOURCE:$ROOT/native_v8/ictc_o3_hp"
export OMP_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XDG_CACHE_HOME="$ROOT/cache/native_v1_1/xdg"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/native_v1_1/extensions"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/native_v1_1/ictd"
export TRITON_CACHE_DIR="$ROOT/cache/native_v1_1/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native_v1_1/pycache"
export TMPDIR="$ROOT/cache/native_v1_1/tmp"
case "$SLURM_ARRAY_TASK_ID" in
  0) CKPT=/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth; TAG=final ;;
  1) CKPT=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth; TAG=persistent ;;
esac
cd "$ROOT"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$SOURCE/benchmarks/hp/qualify_checkpoint.py" \
  --checkpoint "$CKPT" --checkpoint-sc \
  --data-h5 /public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728/transition1x/chorus_reaction_id_50k_seed20260616/processed_val.h5 \
  --frames 16 --out "$ROOT/results/native_heldout_v8_$TAG.json"
printf 'DONE\n'
