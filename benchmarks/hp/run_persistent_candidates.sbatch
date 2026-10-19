#!/usr/bin/env bash
#SBATCH --job-name=persistent-candidate
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/persistent_candidate_%j.log
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
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
if [[ "${TRUNK_MICROBENCH:-0}" == 1 ]]; then
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$SOURCE/benchmarks/hp/profile_radial_trunk.py" \
    --out "$ROOT/results/${RESULT_SET:-radial_trunk_tiles_v70}/tiles.json"
  printf 'DONE\n'
  exit 0
fi
if [[ "${RUN_RADIAL_TESTS:-0}" == 1 ]]; then
  /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
    "$SOURCE/benchmarks/hp/test_radial_trunk.py" --tb=short -x
fi
FAILURES=0
for EXPERIMENT in ${EXPERIMENTS:-persistent_dest_trunk}; do
  if ! /public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
    "$SOURCE/benchmarks/hp/profile_scaling.py" --atoms 10000 --neighbors 50 \
    --checkpoint /public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth \
    --policy generated --sc on --checkpoint-sc --topology-backend device \
    --capture --virial --allow-reference-oom --profile --execution-experiment "$EXPERIMENT" \
    --reference-cache "$ROOT/cache/verification/persistent_n10000_k50_fp32_v67.pt" \
    --out "$ROOT/results/${RESULT_SET:-persistent_candidates_v69}/${EXPERIMENT}.json"; then
    FAILURES=$((FAILURES+1))
  fi
done
if (( FAILURES )); then printf 'FAILED qualification cases: %s\n' "$FAILURES"; exit 1; fi
printf 'DONE\n'
