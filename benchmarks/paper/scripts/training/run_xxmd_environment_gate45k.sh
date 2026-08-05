#!/usr/bin/env bash
set -euo pipefail

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
RUN_ONE="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_xxmd_adaptive_one.sh"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/xxmd_environment_gate_steps45000_20260722}"
MAX_STEPS="${MAX_STEPS:-45000}"
COHERENCE_INIT="${COHERENCE_INIT:-0.5}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100}"
MODE="ictc_phase_full_l_adaptive_env_makefx"

mkdir -p "${OUT_ROOT}"

for molecule in azo dia mal; do
  if [[ -f "${OUT_ROOT}/${molecule}/.complete" ]]; then
    echo "SKIP_COMPLETE molecule=${molecule} $(date)" | tee -a "${OUT_ROOT}/status.log"
    continue
  fi
  echo "START molecule=${molecule} init=${COHERENCE_INIT} steps=${MAX_STEPS} $(date)" \
    | tee -a "${OUT_ROOT}/status.log"
  OUT_ROOT="${OUT_ROOT}" \
  MOLECULE="${molecule}" \
  MODE="${MODE}" \
  COHERENCE_INIT="${COHERENCE_INIT}" \
  MAX_STEPS="${MAX_STEPS}" \
  FORCE_WEIGHT="${FORCE_WEIGHT}" \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  bash "${RUN_ONE}"
  echo "END molecule=${molecule} $(date)" | tee -a "${OUT_ROOT}/status.log"
done

echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
