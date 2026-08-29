#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
SENDKEY_FILE="${SENDKEY_FILE:-${HOME}/.config/chorus-notify/serverchan_sendkey}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
RETRY_SECONDS="${RETRY_SECONDS:-600}"
TOOLS="${REPO}/benchmarks/causal/nequip_phase_specific_20260826"
STATE_DIR="${CAMPAIGN_ROOT}/serverchan"
LOG="${STATE_DIR}/hourly.log"

mkdir -p "${STATE_DIR}"
while true; do
  if [[ ! -s "${SENDKEY_FILE}" ]]; then
    printf '%s WAITING_FOR_SENDKEY file=%s\n' "$(date --iso-8601=seconds)" "${SENDKEY_FILE}" >>"${LOG}"
    sleep 300
    continue
  fi
  if "${PYTHON_BIN}" "${TOOLS}/serverchan_hourly_report.py" \
      --campaign-root "${CAMPAIGN_ROOT}" --tools-dir "${TOOLS}" \
      --sendkey-file "${SENDKEY_FILE}" --state-dir "${STATE_DIR}" \
      >>"${LOG}" 2>&1; then
    sleep "${INTERVAL_SECONDS}"
  else
    printf '%s DELIVERY_FAILED retry_seconds=%s\n' "$(date --iso-8601=seconds)" "${RETRY_SECONDS}" >>"${LOG}"
    sleep "${RETRY_SECONDS}"
  fi
done
