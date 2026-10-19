#!/usr/bin/env bash
#SBATCH --job-name=chorus-arch-full-oracle
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/logs/offload_oracle_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017
SOURCE="${ORACLE_SOURCE:-$ROOT/oracle_source_v1/chorus}"
HP_SOURCE="${ORACLE_HP_SOURCE:-$ROOT/oracle_source_v1/ictc_o3_hp}"
CACHE=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/cache/native_v1_1
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/qualify_architecture_offload.py" --root "$ROOT" \
  --case "${CASE:-l3c256d4}" --scope "${SCOPE:-persistent}" --atoms "${ATOMS:-1024}" \
  --out "$ROOT/${QUALIFICATION_SET:-qualifications}/${CASE:-l3c256d4}/n${ATOMS:-1024}/${SCOPE:-persistent}.json" \
  --product-chunk "${PRODUCT_CHUNK:-0}"
printf 'DONE\n'
