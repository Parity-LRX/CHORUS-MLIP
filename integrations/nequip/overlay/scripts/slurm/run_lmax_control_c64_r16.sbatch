#!/usr/bin/env bash
#SBATCH --job-name=nequip-lmax
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/lmax-control-%j.log

set -euo pipefail

LMAX="${LMAX:?set LMAX to 1 or 3}"
if [[ "${LMAX}" != "1" && "${LMAX}" != "3" ]]; then
  echo "LMAX must be 1 or 3, got ${LMAX}" >&2
  exit 2
fi

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/lmax_controls_c64_r16_20260815/L${LMAX}}"
DATASETS="${DATASETS:-transition1x xxmd_mal xxmd_sti md22_buckyball 3bpa}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"

mkdir -p "${RUN_ROOT}"
IFS=',' read -r -a allocated_devices <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#allocated_devices[@]} != 1 )); then
  echo "expected exactly one allocated GPU, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
device="${allocated_devices[0]}"

launch() {
  local chorus="$1"
  local scope="$2"
  local label="$3"
  env \
    REPO="${REPO}" \
    OPERATOR_REPO="${OPERATOR_REPO}" \
    PYTHON="${PYTHON}" \
    RUN_ROOT="${RUN_ROOT}" \
    DATASETS="${DATASETS}" \
    CHANNELS=64 \
    CHORUS_RANK=16 \
    CHORUS_SCOPE="${scope}" \
    NUM_LAYERS=2 \
    LMAX="${LMAX}" \
    SEED=20260616 \
    EVAL_BATCH_SIZE=25 \
    OPENEQUIVARIANCE=0 \
    bash "${REPO}/scripts/run_extended_variant.sh" \
      e3nn "${chorus}" "${device}" \
      >"${RUN_ROOT}/${label}.driver.log" 2>&1 &
  LAST_PID=$!
}

# These three runs share one allocated GPU, matching the existing L=2 control.
launch off final off
off_pid="${LAST_PID}"
launch on final final
final_pid="${LAST_PID}"
launch on persistent persistent
persistent_pid="${LAST_PID}"

status=0
if ! wait "${off_pid}"; then status=1; fi
if ! wait "${final_pid}"; then status=1; fi
if ! wait "${persistent_pid}"; then status=1; fi
if (( status == 0 )); then
  touch "${RUN_ROOT}/ALL_DONE"
fi
exit "${status}"
