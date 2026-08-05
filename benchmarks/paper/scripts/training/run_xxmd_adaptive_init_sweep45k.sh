#!/usr/bin/env bash
set -euo pipefail

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
RUN_ONE="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_xxmd_adaptive_one.sh"
OUT_BASE="${OUT_BASE:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/xxmd_adaptive_init_sweep_steps45000_20260722}"
WAIT_SCREENS="${WAIT_SCREENS:-chorus_four_system_fill}"
MAX_STEPS="${MAX_STEPS:-45000}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100}"

mkdir -p "${OUT_BASE}"

screen_is_alive() {
  screen -ls 2>/dev/null | grep -Eq "[.]$1[[:space:]]"
}

IFS=',' read -r -a wait_array <<<"${WAIT_SCREENS}"
while true; do
  active=""
  for raw in "${wait_array[@]}"; do
    name="$(echo "${raw}" | xargs)"
    if [[ -n "${name}" ]] && screen_is_alive "${name}"; then
      active="${name}"
      break
    fi
  done
  [[ -z "${active}" ]] && break
  echo "WAIT active_screen=${active} $(date)" | tee -a "${OUT_BASE}/status.log"
  sleep 30
done

run_one() {
  local molecule="$1"
  local init="$2"
  local tag="${init/./p}"
  local root="${OUT_BASE}/init_${tag}"
  if [[ -f "${root}/${molecule}/.complete" ]]; then
    echo "SKIP_COMPLETE molecule=${molecule} init=${init} $(date)" | tee -a "${OUT_BASE}/status.log"
    return
  fi
  echo "START molecule=${molecule} init=${init} steps=${MAX_STEPS} $(date)" | tee -a "${OUT_BASE}/status.log"
  OUT_ROOT="${root}" \
  MOLECULE="${molecule}" \
  COHERENCE_INIT="${init}" \
  MAX_STEPS="${MAX_STEPS}" \
  FORCE_WEIGHT="${FORCE_WEIGHT}" \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  bash "${RUN_ONE}"
  echo "END molecule=${molecule} init=${init} $(date)" | tee -a "${OUT_BASE}/status.log"
}

# Neutral initialization first across the three diagnostic regimes.  The
# diagonal- and full-biased starts follow.  Azo and Dia already have matched
# init=0.1 runs, so only Mal needs that missing endpoint.
run_one azo 0.5
run_one dia 0.5
run_one mal 0.5
run_one azo 0.9
run_one dia 0.9
run_one mal 0.9
run_one mal 0.1

echo "ALL_OK $(date)" | tee -a "${OUT_BASE}/status.log"
