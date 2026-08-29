#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/lrx/CHORUS-MLIP-main-17ccf67}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/home/ylzhang/lrx/compat_e3nn044/mace_torch_0_3_16}"
MODE="${MODE:-chorus}"
case "${MODE}" in
  chorus) SCOPE_LABEL="final" ;;
  persistent) SCOPE_LABEL="persistent" ;;
  *) echo "unsupported MODE=${MODE}; expected chorus or persistent" >&2; exit 2 ;;
esac
ROOT="${ROOT:-/home/ylzhang/lrx/home-organized-20260820/data-runs/chorus_runs/mace_corr3_vs_chorus_corr2_20260826/chorus_${SCOPE_LABEL}_c128_l2_corr2_r16}"
BASE_QUEUE="${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"
STATUS="${ROOT}/queue_status.log"

mkdir -p "${ROOT}"
exec 9>"${ROOT}/run.lock"
if ! flock -n 9; then
  printf 'SKIP_LOCKED %s\n' "$(date -Is)" | tee -a "${STATUS}"
  exit 0
fi

if [[ -f "${ROOT}/ALL_DONE" ]]; then
  printf 'SKIP_ALL_DONE %s\n' "$(date -Is)" | tee -a "${STATUS}"
  exit 0
fi

cat >"${ROOT}/protocol.env" <<EOF
comparison=MACE-ICTC CHORUS-Off correlation=3 (existing) versus CHORUS-${SCOPE_LABEL} correlation=2 (this run)
channels=128
hidden_lmax=2
max_ell=2
num_interactions=2
correlation=2
phase_density_rank=16
phase_scope=${SCOPE_LABEL}
phase_density_pairs=full-nonlinear
seed=20260616
precision=float32
tf32=false
ema=false
energy_force_weight=1:100
checkpoint_selection=minimum validation Force MAE; earliest step breaks ties
test_used_for_selection=false
EOF

run_one() {
  local system="$1"
  rm -f "${ROOT}/ALL_DONE"
  REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
    ROOT="${ROOT}" STATUS="${STATUS}" SYSTEMS="${system}" \
    CHANNELS=128 RANK=16 CORRELATION=2 NUM_INTERACTIONS=2 \
    HIDDEN_LMAX=2 MAX_ELL=2 MODE="${MODE}" PHASE_CONTEXT=content \
    SEED=20260616 \
    bash "${BASE_QUEUE}"
  rm -f "${ROOT}/ALL_DONE"
}

# Put the shorter systems first so the controlled comparison yields evidence early.
for system in mal bucky sti t1x; do
  run_one "${system}"
done

touch "${ROOT}/ALL_DONE"
printf 'ALL_CORR2_%s_CONTROL_TASKS_DONE %s\n' "${SCOPE_LABEL}" "$(date -Is)" | tee -a "${STATUS}"
