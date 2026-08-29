#!/usr/bin/env bash
#SBATCH --job-name=ch-lmax3-train-retry
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/mace_lmax3_train2048_retry_20260819/slurm-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-table4-l3-20260814}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/mace_lmax3_train2048_retry_20260819}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"

ASSETS="/public/home/sps-xia/rxlin/chorus_lmax_depth_extended_assets_20260818"
LMAX3_FINAL="${ASSETS}/lmax3_final_t1x.pth"
LMAX3_PERSISTENT="/public/home/sps-xia/rxlin/chorus_table4_lmax_controls_c128_2layer_r16_20260814/L3/persistent/t1x/checkpoints/t1x_persistent_c128_l3_L3_corr3_rank16_final.pth"

mkdir -p "${OUT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for path in "${LMAX3_FINAL}" "${LMAX3_PERSISTENT}"; do
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
      --tasks train \
      --sizes 2048 \
      --output "${OUT}/${label}_training.json"
  touch "${OUT}/${label}.DONE"
  echo "DONE ${label} $(date --iso-8601=seconds)"
}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
run_one lmax3_final_r16 chorus-final-r16 "${LMAX3_FINAL}" \
  "MACE-ICTC C128, lmax=3, two interactions, correlation=3, CHORUS-Final R16"
run_one lmax3_persistent_r16 chorus-persistent-r16 "${LMAX3_PERSISTENT}" \
  "MACE-ICTC C128, lmax=3, two interactions, correlation=3, CHORUS-Persistent R16"

touch "${OUT}/ALL_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)"
