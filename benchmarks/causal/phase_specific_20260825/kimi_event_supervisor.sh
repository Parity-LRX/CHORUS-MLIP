#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-phase-causal-20260825}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/chorus_phase_causal_20260825}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
KIMI="${KIMI:-/public/home/sps-xia/rxlin/.kimi-code/bin/kimi}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TOOLS="${REPO}/benchmarks/causal/phase_specific_20260825"
ROOT="${CAMPAIGN_ROOT}/supervisor"

mkdir -p "${ROOT}/events"
exec 9>"${ROOT}/supervisor.lock"
flock -n 9 || { echo "another phase supervisor holds the lock" >&2; exit 9; }
status="${ROOT}/status.log"
state="${ROOT}/state.env"
snapshot_json="${ROOT}/current_snapshot.json"
snapshot_md="${ROOT}/current_snapshot.md"
mark() { printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${status}"; }

invoke_kimi() {
  local reason="$1" stamp event_dir
  stamp=$(date +%Y%m%dT%H%M%S)
  event_dir="${ROOT}/events/${stamp}_${reason}"
  mkdir -p "${event_dir}"
  cp "${snapshot_json}" "${snapshot_md}" "${event_dir}/"
  mark "KIMI_START reason=${reason}"
  set +e
  (
    cd "${REPO}"
    KIMI_MODEL_THINKING_EFFORT=max KIMI_MODEL_THINKING_KEEP=all \
      timeout 45m "${KIMI}" -m kimi-code-api/k3 --add-dir "${CAMPAIGN_ROOT}" \
      --agent-file "${TOOLS}/kimi_progress_agent.md" \
      --prompt "Event: ${reason}. Audit the attached current snapshot and immutable logs."
  ) >"${event_dir}/KIMI_PROGRESS.md" 2>"${event_dir}/kimi.stderr"
  local rc=$?
  set -e
  mark "KIMI_DONE reason=${reason} rc=${rc}"
}

previous_stage=""
previous_probe_datasets=-1
previous_aligned_datasets=-1
previous_failures=-1
if [[ -f "${state}" ]]; then
  # shellcheck disable=SC1090
  source "${state}"
fi

mark "SUPERVISOR_START poll_seconds=${POLL_SECONDS} mode=event_driven"
while true; do
  if ! "${PYTHON_BIN}" "${TOOLS}/build_progress_snapshot.py" \
    --campaign-root "${CAMPAIGN_ROOT}" --protocol "${TOOLS}/protocol.json" \
    --markdown "${snapshot_md}" --json "${snapshot_json}" \
    >>"${ROOT}/snapshot_builder.log" 2>&1; then
    mark "SNAPSHOT_FAILED"
    sleep "${POLL_SECONDS}"
    continue
  fi
  read -r current_stage probe_datasets aligned_datasets failures < <(
    "${PYTHON_BIN}" - "${snapshot_json}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(x["stage"], len(x["probe_complete_datasets"]),
      len(x["aligned_complete_datasets"]), len(x["failures"]))
PY
  )
  reasons=()
  [[ -z "${previous_stage}" ]] && reasons+=(startup)
  [[ -n "${previous_stage}" && "${current_stage}" != "${previous_stage}" ]] && reasons+=(stage_change)
  (( previous_probe_datasets >= 0 && probe_datasets > previous_probe_datasets )) && reasons+=(probe_dataset_complete)
  (( previous_aligned_datasets >= 0 && aligned_datasets > previous_aligned_datasets )) && reasons+=(aligned_dataset_complete)
  (( previous_failures >= 0 && failures > previous_failures )) && reasons+=(failure)

  cat >"${state}" <<EOF
previous_stage=${current_stage}
previous_probe_datasets=${probe_datasets}
previous_aligned_datasets=${aligned_datasets}
previous_failures=${failures}
EOF
  previous_stage="${current_stage}"
  previous_probe_datasets="${probe_datasets}"
  previous_aligned_datasets="${aligned_datasets}"
  previous_failures="${failures}"

  # The controller owns the single final Kimi analysis.  Suppress redundant
  # progress calls once deterministic summarisation begins.
  if ((${#reasons[@]})) && [[ "${current_stage}" != summarizing ]] \
      && [[ "${current_stage}" != kimi_final_audit ]] \
      && [[ "${current_stage}" != done ]]; then
    reason=$(IFS=+; echo "${reasons[*]}")
    invoke_kimi "${reason}"
  fi
  if [[ -f "${CAMPAIGN_ROOT}/DONE" || "${current_stage}" == done ]]; then
    mark "CAMPAIGN_DONE"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
