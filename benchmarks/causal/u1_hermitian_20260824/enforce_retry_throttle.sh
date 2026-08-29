#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824}"
CONTROLLER_JOB_ID="${CONTROLLER_JOB_ID:-57093}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
history="${CAMPAIGN_ROOT}/controller/job_history.tsv"
status="${CAMPAIGN_ROOT}/controller/status.log"

while squeue -h -j "${CONTROLLER_JOB_ID}" | grep -q .; do
  if [[ -f "${history}" ]]; then
    while read -r job_id; do
      [[ -n "${job_id}" ]] || continue
      scontrol update JobId="${job_id}" ArrayTaskThrottle="${MAX_CONCURRENT_GPUS}" || true
      marker="RETRY_THROTTLE_ENFORCED job=${job_id} max=${MAX_CONCURRENT_GPUS}"
      if ! grep -Fq "${marker}" "${status}" 2>/dev/null; then
        printf '%s %s\n' "${marker}" "$(date --iso-8601=seconds)" >>"${status}"
      fi
    done < <(awk -F '\t' '$2 ~ /^retry[0-9]+$/ {print $3}' "${history}" | sort -u)
  fi
  [[ ! -f "${CAMPAIGN_ROOT}/DONE" ]] || exit 0
  sleep "${POLL_SECONDS}"
done
