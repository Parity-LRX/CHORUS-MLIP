#!/usr/bin/env bash
#SBATCH --job-name=chorus_a100_scale
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs/mace-%A_%a.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/mace_family}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
SIZES="${SIZES:-128,256,512,1024,2048,4096}"

OFF="/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/phase_off.pth"
FINAL="/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth"
PERSISTENT="/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth"

mkdir -p "${OUT}" /public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "${SLURM_ARRAY_TASK_ID}" in
  0) engine="native-mace"; label="mace_sh_c128"; checkpoint="" ;;
  1) engine="ictc-baseline"; label="mace_ictc_c128"; checkpoint="${OFF}" ;;
  2) engine="chorus-final-r16"; label="chorus_final_c128_r16"; checkpoint="${FINAL}" ;;
  3) engine="chorus-persistent-r16"; label="chorus_persistent_c128_r16"; checkpoint="${PERSISTENT}" ;;
  *) exit 2 ;;
esac

args=(
  "${BENCH}" --engine "${engine}" --sizes "${SIZES}"
  --output "${OUT}/${label}.json"
)
if [[ -n "${checkpoint}" ]]; then
  args+=(--checkpoint "${checkpoint}")
fi

PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${args[@]}"
touch "${OUT}/${label}.DONE"
