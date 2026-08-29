#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
NEQUIP_REPO="${NEQUIP_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826}"
MACE_PERSISTENT_ROOT="${MACE_PERSISTENT_ROOT:-/public/home/sps-xia/rxlin/chorus_persistent_causal_20260826}"
NEQUIP_PERSISTENT_ROOT="${NEQUIP_PERSISTENT_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_persistent_phase_specific_20260826}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
KIMI="${KIMI:-/public/home/sps-xia/rxlin/.kimi-code/bin/kimi}"
POLL_SECONDS="${POLL_SECONDS:-300}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-21600}"
TOOLS="${REPO}/benchmarks/causal/nequip_phase_specific_20260826"
ROOT="${CAMPAIGN_ROOT}/kimi_supervisor"

mkdir -p "${ROOT}/events"
exec 9>"${ROOT}/supervisor.lock"
flock -n 9 || { echo "another unified CHORUS Kimi supervisor holds the lock" >&2; exit 9; }
status="${ROOT}/status.log"
state="${ROOT}/state.env"
final_snapshot_json="${ROOT}/nequip_final_snapshot.json"
final_snapshot_md="${ROOT}/nequip_final_snapshot.md"
snapshot_json="${ROOT}/all_campaigns_snapshot.json"
snapshot_md="${ROOT}/all_campaigns_snapshot.md"
mark() { printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${status}"; }

build_aggregate_snapshot() {
  "${PYTHON_BIN}" - \
    "${final_snapshot_json}" "${MACE_PERSISTENT_ROOT}" \
    "${NEQUIP_PERSISTENT_ROOT}" "${snapshot_json}" "${snapshot_md}" <<'PY'
import collections
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime

final_path, mace_root, nequip_root, output_json, output_md = map(pathlib.Path, sys.argv[1:])
final = json.loads(final_path.read_text())


def mode_keys(root: pathlib.Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not root.is_dir():
        return keys
    for path in root.glob("seed_*/[fa][ul][li][lg]*/*/**/metrics.json"):
        parts = path.relative_to(root).parts
        if len(parts) >= 4 and parts[1] in {"full", "aligned"}:
            keys.add((parts[0], parts[1], parts[2]))
    return keys


def unresolved_failures(root: pathlib.Path, persistent_nequip: bool = False) -> list[str]:
    failures: list[str] = []
    if not root.is_dir():
        return failures
    for failed in root.rglob("FAILED"):
        if persistent_nequip:
            parts = failed.relative_to(root).parts
            if len(parts) >= 3 and parts[0].startswith("seed_") and parts[1] in {"full", "aligned"}:
                key_root = root / parts[0] / parts[1] / parts[2]
                if any(key_root.glob("**/metrics.json")):
                    continue
        elif (failed.parent / "metrics.json").is_file():
            continue
        failures.append(str(failed))
    return failures


mace_runs = mace_root / "mace" / "runs"
mace_full = set(mace_runs.glob("*/seed*/full/metrics.json")) if mace_runs.is_dir() else set()
mace_aligned = set(mace_runs.glob("*/seed*/aligned/metrics.json")) if mace_runs.is_dir() else set()
nequip_keys = mode_keys(nequip_root)
nequip_full = {key for key in nequip_keys if key[1] == "full"}
nequip_aligned = {key for key in nequip_keys if key[1] == "aligned"}

relevant_names = {"nequip-causal-v2", "mace-persist-pair", "nequip-persist-pair"}
service_names = {"nequip-kimi-operator", "chorus-kimi-operator", "chorus-serverchan"}
result = subprocess.run(
    ["squeue", "-u", os.environ.get("USER", "rxlin"), "-h", "-o", "%F|%K|%T|%j|%M|%R"],
    check=False,
    capture_output=True,
    text=True,
)
jobs = []
services = []
for line in result.stdout.splitlines():
    fields = line.split("|", 5)
    if len(fields) != 6:
        continue
    row = dict(zip(("array_job_id", "array_task_id", "state", "name", "elapsed", "reason"), fields))
    if row["name"] in relevant_names:
        jobs.append(row)
    elif row["name"] in service_names:
        services.append(row)

grouped = collections.Counter((row["name"], row["state"], row["reason"]) for row in jobs)
job_signature_source = "\n".join(
    f"{name}|{state}|{reason}|{count}"
    for (name, state, reason), count in sorted(grouped.items())
)
job_signature = hashlib.sha256(job_signature_source.encode()).hexdigest()
mace_failures = unresolved_failures(mace_runs)
nequip_failures = unresolved_failures(nequip_root, persistent_nequip=True)
final_done = int(final["state_counts"]["DONE"])
final_failed = int(final["state_counts"]["FAILED"])
mace_done = len(mace_full) + len(mace_aligned)
nequip_done = len(nequip_full) + len(nequip_aligned)
unresolved = final_failed + len(mace_failures) + len(nequip_failures)

snapshot = {
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "expected": {"nequip_final": 80, "mace_persistent": 40, "nequip_persistent": 40},
    "completed": {
        "nequip_final": final_done,
        "mace_persistent": mace_done,
        "mace_persistent_full": len(mace_full),
        "mace_persistent_aligned": len(mace_aligned),
        "nequip_persistent": nequip_done,
        "nequip_persistent_full": len(nequip_full),
        "nequip_persistent_aligned": len(nequip_aligned),
    },
    "total_completed": final_done + mace_done + nequip_done,
    "unresolved_failures": unresolved,
    "failure_paths": {
        "mace_persistent": mace_failures[-20:],
        "nequip_persistent": nequip_failures[-20:],
    },
    "nequip_final_oom_logs": final.get("oom_logs", []),
    "nequip_final_collapsed_pairs": final.get("collapsed_pairs", []),
    "jobs": jobs,
    "services": services,
    "job_signature": job_signature,
    "all_complete": (
        final_done == 80 and mace_done == 40 and nequip_done == 40 and unresolved == 0
    ),
    "nequip_final_snapshot": final,
}
output_json.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

job_lines = [
    f"- {row['array_job_id']}_{row['array_task_id']} {row['name']} "
    f"{row['state']} {row['elapsed']} ({row['reason']})"
    for row in jobs
] or ["- no active or pending campaign GPU jobs"]
service_lines = [
    f"- {row['array_job_id']} {row['name']} {row['state']} {row['elapsed']}"
    for row in services
] or ["- no monitoring services visible"]
output_md.write_text(
    "# Unified CHORUS causal campaign snapshot\n\n"
    f"Generated: {snapshot['generated_at']}\n\n"
    "## Completion\n\n"
    f"- NequIP-Final: {final_done}/80\n"
    f"- MACE-Persistent: {mace_done}/40 "
    f"(Full {len(mace_full)}/20, Aligned {len(mace_aligned)}/20)\n"
    f"- NequIP-Persistent: {nequip_done}/40 "
    f"(Full {len(nequip_full)}/20, Aligned {len(nequip_aligned)}/20)\n"
    f"- Unresolved failures: {unresolved}\n"
    f"- NequIP-Final OOM logs: {len(final.get('oom_logs', []))}\n"
    f"- NequIP-Final identical Full/Aligned alarms: {len(final.get('collapsed_pairs', []))}\n\n"
    "## Campaign jobs\n\n" + "\n".join(job_lines) + "\n\n"
    "## Monitoring services\n\n" + "\n".join(service_lines) + "\n"
)
PY
}

invoke_kimi() {
  local reason="$1" stamp event_dir rc
  stamp=$(date +%Y%m%dT%H%M%S)
  event_dir="${ROOT}/events/${stamp}_${reason}"
  mkdir -p "${event_dir}"
  cp "${snapshot_json}" "${snapshot_md}" \
    "${final_snapshot_json}" "${final_snapshot_md}" "${event_dir}/"
  mark "KIMI_START reason=${reason}"
  set +e
  (
    cd "${REPO}"
    KIMI_MODEL_THINKING_EFFORT=max KIMI_MODEL_THINKING_KEEP=all \
      timeout 90m "${KIMI}" -m kimi-code-api/k3 \
      --add-dir "${CAMPAIGN_ROOT}" --add-dir "${MACE_PERSISTENT_ROOT}" \
      --add-dir "${NEQUIP_PERSISTENT_ROOT}" --add-dir "${NEQUIP_REPO}" \
      --agent-file "${TOOLS}/kimi_cluster_operator.md" \
      --prompt "Event: ${reason}. Read all_campaigns_snapshot.md first, audit every listed campaign and dependency, and autonomously repair or resubmit within the operator policy when evidence requires it."
  ) >"${event_dir}/KIMI_REPORT.md" 2>"${event_dir}/kimi.stderr"
  rc=$?
  set -e
  mark "KIMI_DONE reason=${reason} rc=${rc} report=${event_dir}/KIMI_REPORT.md"
}

previous_total_done=-1
previous_done_bucket=-1
previous_failures=-1
previous_collapses=-1
previous_oom=-1
previous_job_signature=none
previous_kimi_epoch=0
if [[ -f "${state}" ]]; then
  # shellcheck disable=SC1090
  source "${state}"
fi

mark "SUPERVISOR_START scope=all_causal_campaigns poll_seconds=${POLL_SECONDS} heartbeat_seconds=${HEARTBEAT_SECONDS} permissions=repair_and_resubmit"
while true; do
  if ! "${PYTHON_BIN}" "${TOOLS}/build_kimi_snapshot.py" \
    --campaign-root "${CAMPAIGN_ROOT}" --protocol "${TOOLS}/protocol.json" \
    --json "${final_snapshot_json}" --markdown "${final_snapshot_md}" \
    >>"${ROOT}/snapshot_builder.log" 2>&1; then
    mark "SNAPSHOT_FAILED"
    sleep "${POLL_SECONDS}"
    continue
  fi
  if ! build_aggregate_snapshot >>"${ROOT}/snapshot_builder.log" 2>&1; then
    mark "AGGREGATE_SNAPSHOT_FAILED"
    sleep "${POLL_SECONDS}"
    continue
  fi
  read -r total_done done_bucket failures collapses oom_count all_complete job_signature < <(
    "${PYTHON_BIN}" - "${snapshot_json}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
done = x["total_completed"]
final = x["nequip_final_snapshot"]
print(done, done // 8, x["unresolved_failures"],
      len(final["collapsed_pairs"]), len(final["oom_logs"]),
      int(x["all_complete"]), x["job_signature"])
PY
  )
  now=$(date +%s)
  reasons=()
  (( previous_total_done < 0 )) && reasons+=(startup)
  (( previous_done_bucket >= 0 && done_bucket > previous_done_bucket )) && reasons+=(progress_milestone)
  (( previous_failures >= 0 && failures > previous_failures )) && reasons+=(failure)
  (( previous_collapses >= 0 && collapses > previous_collapses )) && reasons+=(full_aligned_collapse)
  (( previous_oom >= 0 && oom_count > previous_oom )) && reasons+=(oom)
  if [[ "${previous_job_signature}" != none && "${job_signature}" != "${previous_job_signature}" ]]; then
    reasons+=(job_state_transition)
  fi
  if (( previous_kimi_epoch > 0 && now - previous_kimi_epoch >= HEARTBEAT_SECONDS )); then
    reasons+=(six_hour_heartbeat)
  fi

  if ((${#reasons[@]})); then
    reason=$(IFS=+; echo "${reasons[*]}")
    invoke_kimi "${reason}"
    previous_kimi_epoch=$(date +%s)
  fi

  cat >"${state}" <<EOF
previous_total_done=${total_done}
previous_done_bucket=${done_bucket}
previous_failures=${failures}
previous_collapses=${collapses}
previous_oom=${oom_count}
previous_job_signature=${job_signature}
previous_kimi_epoch=${previous_kimi_epoch}
EOF
  previous_total_done="${total_done}"
  previous_done_bucket="${done_bucket}"
  previous_failures="${failures}"
  previous_collapses="${collapses}"
  previous_oom="${oom_count}"
  previous_job_signature="${job_signature}"

  if (( all_complete == 1 && failures == 0 && collapses == 0 )); then
    invoke_kimi final_completion
    mark "ALL_CAMPAIGNS_COMPLETE"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
