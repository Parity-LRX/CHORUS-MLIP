#!/usr/bin/env bash
#SBATCH --job-name=chorus_rank_a100
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs/chorus-rank-%A_%a.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
ASSETS="${ASSETS:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/mace_family}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
SIZES="${SIZES:-128,256,512,1024,2048,4096}"

mkdir -p "${OUT}" /public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "${SLURM_ARRAY_TASK_ID}" in
  0) engine="chorus-r8"; label="chorus_final_c128_r8"; checkpoint="${ASSETS}/chorus_rank8_t1x.pth" ;;
  1) engine="chorus-r32"; label="chorus_final_c128_r32"; checkpoint="${ASSETS}/chorus_rank32_t1x.pth" ;;
  *) exit 2 ;;
esac

PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${BENCH}" --engine "${engine}" \
  --checkpoint "${checkpoint}" --sizes "${SIZES}" \
  --output "${OUT}/${label}.json"
touch "${OUT}/${label}.DONE"
