#!/usr/bin/env bash
set -euo pipefail

CHORUS_REPO="${CHORUS_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
TACE_REPO="${TACE_REPO:-/home/ylzhang/tace_chorus_benchmark}"
FSCETP_PYTHON="${FSCETP_PYTHON:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
TACE_PYTHON="${TACE_PYTHON:-/home/ylzhang/tace_chorus_venv/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
OUT_ROOT="${OUT_ROOT:-/home/ylzhang/chorus_runs/atom_scaling_20260724}"
SIDES="${SIDES:-4,5,6,7,8,9,10,12,14,16}"

CHORUS_CHECKPOINT="${CHORUS_CHECKPOINT:-${CHORUS_REPO}/benchmarks/paper/results/phase/transition1x_chorus_enhanced_20260723/elemcal_c48_l2_h1_rh64_tece_scale_rank8_steps100000_seed20260616/checkpoints/transition1x_ictc_phase_full-nonlinear_c48_l2_rank8_seed20260616_steps100000.calibrated.pth}"
TECE_CHECKPOINT="${TECE_CHECKPOINT:-/home/ylzhang/tace_chorus_runs/transition1x_tece_s_seed20260616/calibrated/TECE-030-96875.train-calibrated.ckpt}"
BENCH="${CHORUS_REPO}/benchmarks/paper/external/tece/bench_atom_scaling.py"

mkdir -p "${OUT_ROOT}"
status="${OUT_ROOT}/status.log"

echo "START chorus $(date -Is)" | tee -a "${status}"
env PYTHONPATH="${CHORUS_REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
  "${FSCETP_PYTHON}" "${BENCH}" \
    --engine chorus \
    --checkpoint "${CHORUS_CHECKPOINT}" \
    --sides "${SIDES}" \
    --output "${OUT_ROOT}/chorus.json" \
    > "${OUT_ROOT}/chorus.log" 2>&1
echo "END chorus $(date -Is)" | tee -a "${status}"

echo "START tece-eager $(date -Is)" | tee -a "${status}"
env PYTHONPATH="${TACE_REPO}:${PYTHONPATH:-}" \
  TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=0 \
  "${TACE_PYTHON}" "${BENCH}" \
    --engine tece-eager \
    --checkpoint "${TECE_CHECKPOINT}" \
    --sides "${SIDES}" \
    --output "${OUT_ROOT}/tece_eager.json" \
    > "${OUT_ROOT}/tece_eager.log" 2>&1
echo "END tece-eager $(date -Is)" | tee -a "${status}"

echo "START tece-cue $(date -Is)" | tee -a "${status}"
env PYTHONPATH="${TACE_REPO}:${PYTHONPATH:-}" \
  TACE_USE_OEQ=0 TACE_USE_CUE=1 TACE_USE_EQT=0 TACE_USE_COMPILE=0 \
  "${TACE_PYTHON}" "${BENCH}" \
    --engine tece-cue \
    --checkpoint "${TECE_CHECKPOINT}" \
    --sides "${SIDES}" \
    --output "${OUT_ROOT}/tece_cue.json" \
    > "${OUT_ROOT}/tece_cue.log" 2>&1
echo "END tece-cue $(date -Is)" | tee -a "${status}"

echo "ALL_OK $(date -Is)" | tee -a "${status}"
