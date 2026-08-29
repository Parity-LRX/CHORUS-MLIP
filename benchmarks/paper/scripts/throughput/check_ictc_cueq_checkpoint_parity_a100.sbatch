#!/usr/bin/env bash
#SBATCH --job-name=cueq_ckpt_parity
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=1-3%1

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_cueq_conv_20260817/parity}"

case "${SLURM_ARRAY_TASK_ID}" in
  1) label="mace_ictc_c128"; checkpoint="/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/phase_off.pth" ;;
  2) label="chorus_final_c128_r16"; checkpoint="/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth" ;;
  3) label="chorus_persistent_c128_r16"; checkpoint="/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth" ;;
  *) exit 2 ;;
esac

mkdir -p "${OUT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/throughput/check_ictc_cueq_checkpoint_parity.py" \
  --checkpoint "${checkpoint}" --label "${label}" --output "${OUT}/${label}.json"
