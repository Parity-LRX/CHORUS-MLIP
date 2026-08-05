#!/usr/bin/env bash
#SBATCH --job-name=tece_a100_scale
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs/tece-%A_%a.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
ASSETS="${ASSETS:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804}"
TACE_REPO="${TACE_REPO:-${ASSETS}/tace_chorus_benchmark}"
TACE_SITE="${TACE_SITE:-${ASSETS}/tace_chorus_venv_overlay/site-packages}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/tece_family}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
SIZES="${SIZES:-128,256,512,1024,2048,4096}"

mkdir -p "${OUT}" /public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "${SLURM_ARRAY_TASK_ID}" in
  0) label="tece_c36_openeq"; checkpoint="${ASSETS}/tece_c36.ckpt" ;;
  1) label="tece_c48_openeq"; checkpoint="${ASSETS}/tece_c48.ckpt" ;;
  *) exit 2 ;;
esac

PYTHONPATH="${TACE_SITE}:${TACE_REPO}:${REPO}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${BENCH}" --engine tece --tece-backend openeq \
  --checkpoint "${checkpoint}" --sizes "${SIZES}" \
  --output "${OUT}/${label}.json"
touch "${OUT}/${label}.DONE"
