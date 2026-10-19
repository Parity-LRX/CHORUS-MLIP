#!/usr/bin/env bash
#SBATCH --job-name=chorus-hp-training
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/training_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set immutable CHORUS snapshot}"
HP_SOURCE="${HP_SOURCE:?set immutable ICTC-O3-HP snapshot}"
export PYTHONPATH="$SOURCE:$HP_SOURCE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
CACHE="$ROOT/cache/native_v1_1"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
PY=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv
set -x
"$PY" -m pytest -q "$SOURCE/benchmarks/hp/test_coherent.py" \
  "$SOURCE/benchmarks/hp/test_coherent_training.py" \
  "$SOURCE/chorus/test/test_hp_backend.py::test_force_training_preserves_live_tp_coefficients" \
  "$SOURCE/chorus/test/test_hp_backend.py::test_whole_force_training_and_checkpoint_identity" --tb=short -x
for N in ${TEST_SIZES:-128 512}; do
  "$PY" "$SOURCE/benchmarks/hp/profile_training.py" \
    --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
    --atoms "$N" --neighbors 50 --profile ${TRAINING_FLAGS:-} \
    --out "$ROOT/results/${RESULT_SET:-training_v31}/persistent_n${N}_k50.json"
done
printf 'DONE\n'
