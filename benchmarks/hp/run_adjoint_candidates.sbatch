#!/usr/bin/env bash
#SBATCH --job-name=chorus-adjoint
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/adjoint_%A_%a.log
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
SCHEDULES=(segmented edge_dx phase_context csr1 csr2 csr4 csr8 split)
SCHEDULE="${SCHEDULES[$SLURM_ARRAY_TASK_ID]}"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
if [[ "$SCHEDULE" == phase_context ]]; then
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
    "$SOURCE/benchmarks/hp/test_phase_trunk.py"
fi
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/profile_adjoint_candidates.py" \
  --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
  --atoms "${ATOMS:-10000}" --schedule "$SCHEDULE" \
  --out "$ROOT/results/${RESULT_SET:-adjoint_v43}/${SCHEDULE}.json"
printf 'DONE\n'
