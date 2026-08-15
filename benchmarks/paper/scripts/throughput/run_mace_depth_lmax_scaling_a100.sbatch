#!/usr/bin/env bash
#SBATCH --job-name=ch-depth-lmax-scale
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/mace_depth_lmax_scaling_a100_20260815/slurm-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-table4-l3-20260814}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/mace_depth_lmax_scaling_a100_20260815}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"

L2_OFF="/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/phase_off.pth"
L2_FINAL="/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth"
L2_PERSISTENT="/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/persistent_t1x.pth"
LMAX3_OFF="/public/home/sps-xia/rxlin/a100_scaling_assets_20260815/lmax3_off_t1x.pth"
DEPTH3_OFF="/public/home/sps-xia/rxlin/chorus_depth3_scope_20260801/off_c128_l2_corr3_depth3_seed20260616/t1x/checkpoints/t1x_phaseoff_c128_l2_corr3_rank16_final.pth"

mkdir -p "${OUT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "${L2_OFF}" "${L2_FINAL}" "${L2_PERSISTENT}" "${LMAX3_OFF}" "${DEPTH3_OFF}"; do
  [[ -f "${path}" ]] || { echo "missing checkpoint: ${path}" >&2; exit 2; }
done

run_one() {
  local label="$1"
  local engine="$2"
  local checkpoint="$3"
  local configuration="$4"

  echo "START ${label} $(date --iso-8601=seconds)"
  PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${BENCH}" \
      --engine "${engine}" \
      --checkpoint "${checkpoint}" \
      --configuration-label "${configuration}" \
      --tasks inference \
      --sizes 128,256,512,1024,2048,4096 \
      --output "${OUT}/${label}_inference.json"

  PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${BENCH}" \
      --engine "${engine}" \
      --checkpoint "${checkpoint}" \
      --configuration-label "${configuration}" \
      --tasks train \
      --sizes 128,256,512,1024,2048 \
      --output "${OUT}/${label}_training.json"
  touch "${OUT}/${label}.DONE"
  echo "DONE ${label} $(date --iso-8601=seconds)"
}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
run_one l2_off ictc-baseline "${L2_OFF}" \
  "MACE-ICTC C128, lmax=2, two interactions, correlation=3, CHORUS-Off"
run_one l2_final_r16 chorus-final-r16 "${L2_FINAL}" \
  "MACE-ICTC C128, lmax=2, two interactions, correlation=3, CHORUS-Final R16"
run_one l2_persistent_r16 chorus-persistent-r16 "${L2_PERSISTENT}" \
  "MACE-ICTC C128, lmax=2, two interactions, correlation=3, CHORUS-Persistent R16"
run_one lmax3_off ictc-baseline "${LMAX3_OFF}" \
  "MACE-ICTC C128, lmax=3, two interactions, correlation=3, CHORUS-Off"
run_one depth3_off ictc-baseline "${DEPTH3_OFF}" \
  "MACE-ICTC C128, lmax=2, three interactions, correlation=3, CHORUS-Off"

touch "${OUT}/ALL_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)"
