#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-u1-causal-20260824}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824}"
RUN_ROOT="${RUN_ROOT:-${CAMPAIGN_ROOT}/runs}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
KIMI="${KIMI:-/public/home/sps-xia/rxlin/.kimi-code/bin/kimi}"
POLL_SECONDS="${POLL_SECONDS:-300}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-4}"
TOOLS="${REPO}/benchmarks/causal/u1_hermitian_20260824"
SUPERVISOR_ROOT="${CAMPAIGN_ROOT}/supervisor"

mkdir -p "${SUPERVISOR_ROOT}/events"
exec 9>"${SUPERVISOR_ROOT}/supervisor.lock"
if ! flock -n 9; then
  echo "another Kimi event supervisor already holds the lock" >&2
  exit 9
fi

status="${SUPERVISOR_ROOT}/status.log"
state="${SUPERVISOR_ROOT}/state.env"
snapshot_md="${SUPERVISOR_ROOT}/current_snapshot.md"
snapshot_json="${SUPERVISOR_ROOT}/current_snapshot.json"

mark() {
  printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${status}"
}

build_snapshot() {
  "${PYTHON_BIN}" "${TOOLS}/build_progress_snapshot.py" \
    --run-root "${RUN_ROOT}" \
    --protocol "${TOOLS}/protocol.json" \
    --markdown "${snapshot_md}" \
    --json "${snapshot_json}"
}

read_counts() {
  "${PYTHON_BIN}" - "${snapshot_json}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(x["completed_runs"], x["complete_blocks"], len(x["complete_datasets"]), len(x["failed_runs"]))
PY
}

enforce_array_throttle() {
  local history="${CAMPAIGN_ROOT}/controller/job_history.tsv"
  [[ -f "${history}" ]] || return 0
  while IFS=$'\t' read -r _ _ job_id _; do
    [[ "${job_id}" =~ ^[0-9]+$ ]] || continue
    if squeue -h -j "${job_id}" | grep -q .; then
      scontrol update JobId="${job_id}" ArrayTaskThrottle="${ARRAY_THROTTLE}" \
        >/dev/null 2>&1 || true
    fi
  done <"${history}"
}

invoke_kimi() {
  local reason="$1"
  local stamp event_dir prompt report
  stamp=$(date +%Y%m%dT%H%M%S)
  event_dir="${SUPERVISOR_ROOT}/events/${stamp}_${reason}"
  mkdir -p "${event_dir}"
  cp "${snapshot_md}" "${event_dir}/snapshot.md"
  cp "${snapshot_json}" "${event_dir}/snapshot.json"
  prompt=$(printf 'Event reason: %s\n\n%s' "${reason}" "$(cat "${snapshot_md}")")
  report="${event_dir}/KIMI_PROGRESS.md"
  mark "KIMI_START reason=${reason}"
  set +e
  (
    cd "${REPO}"
    KIMI_MODEL_THINKING_EFFORT=max KIMI_MODEL_THINKING_KEEP=all \
      timeout 30m "${KIMI}" -m kimi-code-api/k3 \
      --add-dir "${CAMPAIGN_ROOT}" \
      --agent-file "${TOOLS}/kimi_progress_agent.md" --prompt "${prompt}"
  ) >"${report}.tmp" 2>"${event_dir}/kimi.stderr"
  local rc=$?
  set -e
  if (( rc == 0 )) && [[ -s "${report}.tmp" ]]; then
    mv "${report}.tmp" "${event_dir}/kimi.transcript.md"
    "${PYTHON_BIN}" - "${event_dir}/kimi.transcript.md" "${report}" <<'PY'
import pathlib, re, sys
source = pathlib.Path(sys.argv[1]).read_text(errors="replace")
lines = source.splitlines()
start = next(
    (i for i, line in enumerate(lines) if re.match(r"^\s*[•*-]?\s*STATUS:\s*", line)),
    None,
)
if start is None:
    cleaned = source
else:
    lines = lines[start:]
    lines[0] = re.sub(r"^\s*[•*-]?\s*", "", lines[0])
    cleaned = "\n".join(lines).strip() + "\n"
pathlib.Path(sys.argv[2]).write_text(cleaned)
PY
    cp "${report}" "${SUPERVISOR_ROOT}/latest.md"
    mark "KIMI_DONE reason=${reason} report=${report}"
  else
    mv "${report}.tmp" "${event_dir}/kimi.partial.md" 2>/dev/null || true
    mark "KIMI_FAILED reason=${reason} rc=${rc}"
  fi
}

previous_runs=-1
previous_blocks=-1
previous_datasets=-1
previous_failed=-1
if [[ -f "${state}" ]]; then
  # shellcheck disable=SC1090
  source "${state}"
fi

mark "SUPERVISOR_START poll_seconds=${POLL_SECONDS} array_throttle=${ARRAY_THROTTLE}"
while true; do
  enforce_array_throttle
  if ! build_snapshot >>"${SUPERVISOR_ROOT}/snapshot_builder.log" 2>&1; then
    mark "SNAPSHOT_FAILED retry_after=${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi
  read -r completed_runs complete_blocks complete_datasets failed_runs < <(read_counts)

  reasons=()
  if (( previous_runs < 0 )); then
    reasons+=(startup)
  fi
  if (( complete_blocks > previous_blocks && previous_blocks >= 0 )); then
    reasons+=(complete_block)
  fi
  if (( complete_datasets > previous_datasets && previous_datasets >= 0 )); then
    reasons+=(complete_dataset)
  fi
  if (( failed_runs > previous_failed && previous_failed >= 0 )); then
    reasons+=(failure)
  fi

  cat >"${state}" <<EOF
previous_runs=${completed_runs}
previous_blocks=${complete_blocks}
previous_datasets=${complete_datasets}
previous_failed=${failed_runs}
EOF

  # Keep the in-memory baseline in sync with the persisted state.  Without
  # this update, a long-lived supervisor compares every later snapshot with
  # the values loaded only once at startup and repeatedly emits the same
  # completion event.
  previous_runs=${completed_runs}
  previous_blocks=${complete_blocks}
  previous_datasets=${complete_datasets}
  previous_failed=${failed_runs}

  if (( completed_runs < 80 )) && ((${#reasons[@]})); then
    reason=$(IFS=+; echo "${reasons[*]}")
    invoke_kimi "${reason}"
  elif (( completed_runs >= 80 )); then
    mark "ALL_RUNS_COMPLETE final_analysis_deferred_to_controller"
    exit 0
  fi

  if [[ -f "${CAMPAIGN_ROOT}/DONE" ]]; then
    mark "CAMPAIGN_DONE"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
