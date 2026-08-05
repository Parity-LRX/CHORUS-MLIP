#!/usr/bin/env bash
#SBATCH --job-name=nequip_d2_c84_oeq
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2-c84-oeq-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_c84_r16_matched_oeq_20260803}"
DATASETS="${DATASETS:-transition1x xxmd_mal xxmd_sti md22_buckyball 3bpa}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
OEQ_SITE="${OEQ_SITE:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages}"
SEED="${SEED:-20260616}"

test -f "/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/READY"
export PYTHONPATH="${OEQ_SITE}:${PYTHONPATH:-}"
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
    CHANNELS=84 \
    CHORUS_RANK=16 \
    CHORUS_SCOPE="${scope}" \
    NUM_LAYERS=2 \
    SEED="${SEED}" \
    EVAL_BATCH_SIZE=25 \
    OPENEQUIVARIANCE=1 \
    bash "${REPO}/scripts/run_extended_variant.sh" \
      e3nn "${chorus}" "${device}" \
      >"${RUN_ROOT}/${label}.driver.log" 2>&1 &
  LAST_PID=$!
}

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
