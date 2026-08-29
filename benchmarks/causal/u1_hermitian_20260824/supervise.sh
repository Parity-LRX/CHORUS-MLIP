#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:?usage: supervise.sh JOB_ID REPO RUN_ROOT CAMPAIGN_ROOT}"
REPO="${2:?usage: supervise.sh JOB_ID REPO RUN_ROOT CAMPAIGN_ROOT}"
RUN_ROOT="${3:?usage: supervise.sh JOB_ID REPO RUN_ROOT CAMPAIGN_ROOT}"
CAMPAIGN_ROOT="${4:?usage: supervise.sh JOB_ID REPO RUN_ROOT CAMPAIGN_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
KIMI="${KIMI:-/public/home/sps-xia/rxlin/.kimi-code/bin/kimi}"

mkdir -p "${CAMPAIGN_ROOT}/supervisor" "${CAMPAIGN_ROOT}/summary"
log="${CAMPAIGN_ROOT}/supervisor/status.log"

printf 'WATCH job=%s %s\n' "${JOB_ID}" "$(date --iso-8601=seconds)" >>"${log}"
while squeue -h -j "${JOB_ID}" | grep -q .; do
  printf 'RUNNING job=%s %s\n' "${JOB_ID}" "$(date --iso-8601=seconds)" >>"${log}"
  sleep 300
done

set +e
"${PYTHON_BIN}" "${REPO}/benchmarks/causal/u1_hermitian_20260824/summarize.py" \
  --run-root "${RUN_ROOT}" --output-dir "${CAMPAIGN_ROOT}/summary" \
  >"${CAMPAIGN_ROOT}/supervisor/summarize.log" 2>&1
summary_rc=$?
set -e

if (( summary_rc == 0 )); then
  touch "${CAMPAIGN_ROOT}/DONE"
  printf 'DONE job=%s %s\n' "${JOB_ID}" "$(date --iso-8601=seconds)" >>"${log}"
  exit 0
fi

{
  echo "# Frozen CHORUS causal campaign failure context"
  echo
  echo "Slurm job: ${JOB_ID}"
  echo "Campaign: ${CAMPAIGN_ROOT}"
  echo
  echo "## Completion audit"
  cat "${CAMPAIGN_ROOT}/summary/completion_audit.json" 2>/dev/null || true
  echo
  echo "## Slurm accounting"
  sacct -j "${JOB_ID}" --format=JobIDRaw,State,ExitCode,Elapsed,NodeList -n -P || true
  echo
  echo "## Recent error tails"
  find "${CAMPAIGN_ROOT}/slurm" -type f -name '*.err' -size +0c -print0 2>/dev/null \
    | xargs -0 -r tail -n 40
} >"${CAMPAIGN_ROOT}/supervisor/triage_context.md"

cd "${CAMPAIGN_ROOT}/supervisor"
"${KIMI}" --agent-file \
  "${REPO}/benchmarks/causal/u1_hermitian_20260824/kimi_triage_agent.md" \
  --prompt "Read triage_context.md and produce the requested evidence-based audit. Do not modify files." \
  >"${CAMPAIGN_ROOT}/supervisor/KIMI_TRIAGE.md" 2>"${CAMPAIGN_ROOT}/supervisor/kimi.stderr" || true
touch "${CAMPAIGN_ROOT}/NEEDS_REVIEW"
printf 'NEEDS_REVIEW job=%s %s\n' "${JOB_ID}" "$(date --iso-8601=seconds)" >>"${log}"
