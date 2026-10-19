#!/usr/bin/env bash
#SBATCH --job-name=chorus-kernel-screen
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/kernel_screen_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set immutable CHORUS snapshot}"
HP_SOURCE="${HP_SOURCE:?set immutable ICTC-O3-HP snapshot}"
CACHE="$ROOT/cache/native_v1_1"
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
CANDIDATES=(fw4 fw8 owned4 streams16 streams8 owned2 owned8 dest1 dest2 dest4 phase dest1phase rank dest1rank dest1rankphase geom dest1rankphasegeom dest1atomicphasegeom readonly readonlyphasegeom readonlyfactorphasegeom dest1atomicfactorphasegeom neighbor2phasegeom neighbor4phasegeom network dest1atomicfactornetworkgeom dest1atomicfactorcoalescenetworkgeom dest1atomicfactorcoalescephasegeom dest1atomicfactorcoalescenetworkpreprojectgeom dest1atomicfactorcoalescenetworknativeharmgeom)
CANDIDATE="${CANDIDATES[$SLURM_ARRAY_TASK_ID]}"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/profile_kernel_candidates.py" \
  --checkpoint "${CHECKPOINT:-/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth}" \
  --atoms 10000 --candidate "$CANDIDATE" \
  --out "$ROOT/results/${RESULT_SET:-final_kernel_v53}/${CANDIDATE}.json"
printf 'DONE\n'
