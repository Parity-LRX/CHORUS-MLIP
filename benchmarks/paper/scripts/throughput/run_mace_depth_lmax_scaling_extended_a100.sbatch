#!/usr/bin/env bash
#SBATCH --job-name=ch-depth-lmax-ext
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/mace_depth_lmax_scaling_extended_a100_20260818/slurm-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-table4-l3-20260814}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/mace_depth_lmax_scaling_extended_a100_20260818}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"

ASSETS="/public/home/sps-xia/rxlin/chorus_lmax_depth_extended_assets_20260818"
L1_OFF="${ASSETS}/l1_off_t1x.pth"
L1_FINAL="${ASSETS}/l1_final_t1x.pth"
L1_PERSISTENT="${ASSETS}/l1_persistent_t1x.pth"
LMAX3_FINAL="${ASSETS}/lmax3_final_t1x.pth"
LMAX3_PERSISTENT="/public/home/sps-xia/rxlin/chorus_table4_lmax_controls_c128_2layer_r16_20260814/L3/persistent/t1x/checkpoints/t1x_persistent_c128_l3_L3_corr3_rank16_final.pth"
DEPTH3_FINAL="/public/home/sps-xia/rxlin/chorus_depth3_scope_20260801/final_c128_l2_corr3_r16_depth3_seed20260617/t1x/checkpoints/t1x_chorus_c128_l2_corr3_rank16_final.pth"
DEPTH3_PERSISTENT="/public/home/sps-xia/rxlin/chorus_depth3_scope_20260801/persistent_c128_l2_corr3_r16_depth3_seed20260617/t1x/checkpoints/t1x_persistent_c128_l2_corr3_rank16_final.pth"
WIDENED_C140_OFF="${ASSETS}/widened_c140_off_t1x.pth"

mkdir -p "${OUT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "${L1_OFF}" "${L1_FINAL}" "${L1_PERSISTENT}" "${LMAX3_FINAL}" \
  "${LMAX3_PERSISTENT}" "${DEPTH3_FINAL}" "${DEPTH3_PERSISTENT}" "${WIDENED_C140_OFF}"; do
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
run_one l1_off ictc-baseline "${L1_OFF}" \
  "MACE-ICTC C128, lmax=1, two interactions, correlation=3, CHORUS-Off"
run_one l1_final_r16 chorus-final-r16 "${L1_FINAL}" \
  "MACE-ICTC C128, lmax=1, two interactions, correlation=3, CHORUS-Final R16"
run_one l1_persistent_r16 chorus-persistent-r16 "${L1_PERSISTENT}" \
  "MACE-ICTC C128, lmax=1, two interactions, correlation=3, CHORUS-Persistent R16"
run_one lmax3_final_r16 chorus-final-r16 "${LMAX3_FINAL}" \
  "MACE-ICTC C128, lmax=3, two interactions, correlation=3, CHORUS-Final R16"
run_one lmax3_persistent_r16 chorus-persistent-r16 "${LMAX3_PERSISTENT}" \
  "MACE-ICTC C128, lmax=3, two interactions, correlation=3, CHORUS-Persistent R16"
run_one depth3_final_r16 chorus-final-r16 "${DEPTH3_FINAL}" \
  "MACE-ICTC C128, lmax=2, three interactions, correlation=3, CHORUS-Final R16"
run_one depth3_persistent_r16 chorus-persistent-r16 "${DEPTH3_PERSISTENT}" \
  "MACE-ICTC C128, lmax=2, three interactions, correlation=3, CHORUS-Persistent R16"
run_one widened_c140_off ictc-baseline "${WIDENED_C140_OFF}" \
  "MACE-ICTC C140, lmax=2, two interactions, correlation=3, CHORUS-Off"

touch "${OUT}/ALL_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)"
