#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/lrx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ylzhang/chorus_runs/phase_interference_final_20260804}"
SCRIPT="${REPO}/benchmarks/paper/scripts/training/diagnose_phase_interference.py"

mkdir -p "${OUTPUT_ROOT}/logs"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

run_one() {
  local name="$1"
  local label="$2"
  local elements="$3"
  local data_dir="$4"
  local checkpoint="$5"
  local intervention_batches="${6:-16}"

  local done_file="${OUTPUT_ROOT}/${name}/DONE"
  local log_file="${OUTPUT_ROOT}/logs/${name}.log"
  if [[ -f "${done_file}" ]]; then
    echo "SKIP ${name}: ${done_file} exists"
    return 0
  fi
  test -f "${checkpoint}"
  test -f "${data_dir}/processed_val.h5"
  mkdir -p "${OUTPUT_ROOT}/${name}"
  {
    echo "START $(date --iso-8601=seconds)"
    echo "DATASET=${label}"
    echo "CHECKPOINT=${checkpoint}"
    echo "DATA_DIR=${data_dir}"
    echo "STRICT_FP32=true TF32=false"
    "${PYTHON_BIN}" "${SCRIPT}" \
      --checkpoint "${checkpoint}" \
      --data-dir "${data_dir}" \
      --elements "${elements}" \
      --split val \
      --dataset-label "${label}" \
      --batch-size 16 \
      --max-batches 1000000 \
      --intervention-batches "${intervention_batches}" \
      --bins 72 \
      --device cuda \
      --output-dir "${OUTPUT_ROOT}/${name}"
    touch "${done_file}"
    echo "DONE $(date --iso-8601=seconds)"
  } 2>&1 | tee "${log_file}"
}

run_one \
  buckyball "MD22 Buckyball" "H,C" \
  "${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed" \
  "/home/ylzhang/chorus_runs/buckyball_fair_r5_noema_20260724/chorus_c128_l2_corr3_rank16/checkpoints/md22_buckyball_catcher_c128_l2_corr3_rank16_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs300.pth"

run_one \
  mal "xxMD MAL" "H,C,O" \
  "${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal" \
  "/home/ylzhang/chorus_runs/large_scale_main_20260724/xxmd/mal/chorus_c128_l2_corr3_rank16/checkpoints/xxmd_mal_c128_l2_corr3_rank16_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs52.e48s42875.pth"

run_one \
  sti "xxMD STI" "H,C" \
  "${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti" \
  "/home/ylzhang/chorus_runs/large_scale_main_20260724/xxmd/sti/chorus_c128_l2_corr3_rank16/checkpoints/xxmd_sti_c128_l2_corr3_rank16_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs57.e16s13600.pth"

run_one \
  3bpa "3BPA 300 K" "H,C,N,O" \
  "${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5" \
  "/home/ylzhang/chorus_runs/3bpa_rank16_20260727/3bpa_chorus_c128_l2_corr3_rank16/checkpoints/3bpa_chorus_c128_l2_corr3_rank16_final.e1363s38192.pth"

run_one \
  t1x "Transition1x subset" "H,C,N,O" \
  "${DATA_ROOT}/transition1x/chorus_reaction_id_50k_seed20260616" \
  "/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x/chorus_c128_l2_corr3_rank16_mae_ckpts_rerun/checkpoints/t1x_c128_l2_corr3_rank16_mae_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs32.e26s84348.pth"

echo "ALL_DONE $(date --iso-8601=seconds)"
