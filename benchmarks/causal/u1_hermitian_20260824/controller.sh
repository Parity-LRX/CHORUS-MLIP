#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-u1-causal-20260824}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824}"
RUN_ROOT="${RUN_ROOT:-${CAMPAIGN_ROOT}/runs}"
SMOKE_ROOT="${SMOKE_ROOT:-${CAMPAIGN_ROOT}/smoke/runs}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
KIMI="${KIMI:-/public/home/sps-xia/rxlin/.kimi-code/bin/kimi}"
MAX_RETRY_ROUNDS="${MAX_RETRY_ROUNDS:-2}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
TOOLS="${REPO}/benchmarks/causal/u1_hermitian_20260824"

mkdir -p "${CAMPAIGN_ROOT}/controller" "${CAMPAIGN_ROOT}/summary" \
  "${CAMPAIGN_ROOT}/slurm" "${RUN_ROOT}"
exec 9>"${CAMPAIGN_ROOT}/controller/controller.lock"
if ! flock -n 9; then
  echo "another controller holds ${CAMPAIGN_ROOT}/controller/controller.lock" >&2
  exit 9
fi

status="${CAMPAIGN_ROOT}/controller/status.log"
history="${CAMPAIGN_ROOT}/controller/job_history.tsv"
touch "${history}"

mark() {
  printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${status}"
}

preflight() {
  mark "PREFLIGHT_START"
  "${KIMI}" doctor >/dev/null
  for f in SOURCE_STATE protocol.json causal_dag.md causal_dag.dot; do
    [[ -f "${TOOLS}/${f}" ]] || { echo "missing ${TOOLS}/${f}" >&2; return 1; }
  done
  export PYTHONPATH="${REPO}:/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16"
  export PYTHONDONTWRITEBYTECODE=1
  "${PYTHON_BIN}" -m pytest -q "${REPO}/chorus/test/test_training_smoke.py" \
    -k checkpoint_selection_metric >/dev/null
  "${PYTHON_BIN}" -m pytest -q "${REPO}/chorus/test/test_phase_hermitian.py" \
    -k charge2 >/dev/null
  "${PYTHON_BIN}" - "${SMOKE_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "mal" / "seed20260616"
full = json.loads((root / "full" / "metrics.json").read_text())
charge2 = json.loads((root / "charge2" / "metrics.json").read_text())
assert full["parameter_count"] == charge2["parameter_count"]
assert full["max_steps"] == charge2["max_steps"] == 20
assert full["checkpoint_selection"]["rule"].startswith("minimum validation Force MAE")
assert charge2["checkpoint_selection"]["rule"].startswith("minimum validation Force MAE")
PY
  mark "PREFLIGHT_DONE"
}

wait_for_job() {
  local job_id="$1"
  while squeue -h -j "${job_id}" | grep -q .; do
    mark "ARRAY_RUNNING job=${job_id}"
    sleep "${POLL_SECONDS}"
  done
  sacct -j "${job_id}" --format=JobIDRaw,State,ExitCode,Elapsed,NodeList \
    -n -P >"${CAMPAIGN_ROOT}/controller/sacct_${job_id}.tsv" || true
  mark "ARRAY_EXITED job=${job_id}"
}

submit_array() {
  local label="$1"
  local ids="$2"
  local job_id
  job_id=$(sbatch --parsable --array="${ids}%${MAX_CONCURRENT_GPUS}" \
    --export=ALL,REPO="${REPO}",RUN_ROOT="${RUN_ROOT}",MAX_STEPS=45000 \
    "${TOOLS}/run_array_a100.sbatch")
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${label}" \
    "${job_id}" "${ids}" >>"${history}"
  mark "ARRAY_SUBMITTED label=${label} job=${job_id} ids=${ids}" >&2
  printf '%s\n' "${job_id}"
}

try_summary() {
  set +e
  "${PYTHON_BIN}" "${TOOLS}/summarize.py" --run-root "${RUN_ROOT}" \
    --output-dir "${CAMPAIGN_ROOT}/summary" \
    >"${CAMPAIGN_ROOT}/controller/summarize.log" 2>&1
  local rc=$?
  set -e
  return "${rc}"
}

invoke_kimi() {
  local agent_file="$1"
  local prompt="$2"
  local output="$3"
  (
    cd "${REPO}"
    KIMI_MODEL_THINKING_EFFORT=max KIMI_MODEL_THINKING_KEEP=all \
      "${KIMI}" -m kimi-code-api/k3 --add-dir "${CAMPAIGN_ROOT}" \
      --agent-file "${agent_file}" --prompt "${prompt}"
  ) >"${output}" 2>"${output%.md}.stderr" || true
}

preflight
if [[ -f "${CAMPAIGN_ROOT}/DONE" ]]; then
  mark "ALREADY_DONE"
  exit 0
fi

initial_ids=$("${PYTHON_BIN}" "${TOOLS}/missing_task_ids.py" \
  --run-root "${RUN_ROOT}" --format ids)
if [[ -n "${initial_ids}" ]]; then
  job_id=$(submit_array initial "${initial_ids}")
  wait_for_job "${job_id}"
else
  mark "NO_INITIAL_TASKS_MISSING"
fi

retry_round=0
while ! try_summary; do
  missing=$("${PYTHON_BIN}" "${TOOLS}/missing_task_ids.py" \
    --run-root "${RUN_ROOT}" --format ids)
  "${PYTHON_BIN}" "${TOOLS}/missing_task_ids.py" --run-root "${RUN_ROOT}" \
    --format json >"${CAMPAIGN_ROOT}/controller/missing_round${retry_round}.json"
  if [[ -z "${missing}" ]]; then
    mark "SUMMARY_FAILED_WITHOUT_MISSING_RUNS"
    break
  fi
  if (( retry_round >= MAX_RETRY_ROUNDS )); then
    mark "RETRY_LIMIT_REACHED missing=${missing}"
    break
  fi
  retry_round=$((retry_round + 1))
  job_id=$(submit_array "retry${retry_round}" "${missing}")
  wait_for_job "${job_id}"
done

if try_summary; then
  invoke_kimi "${TOOLS}/kimi_causal_analyst.md" \
    "Audit the completed campaign at ${CAMPAIGN_ROOT} using the frozen protocol and DAG. Return the final Markdown causal report only." \
    "${CAMPAIGN_ROOT}/summary/KIMI_CAUSAL_REPORT.md"
  touch "${CAMPAIGN_ROOT}/DONE"
  rm -f "${CAMPAIGN_ROOT}/NEEDS_REVIEW"
  mark "CAMPAIGN_DONE"
  exit 0
fi

{
  echo "# Frozen campaign failure context"
  echo
  echo "Campaign root: ${CAMPAIGN_ROOT}"
  echo
  echo "## Completion audit"
  cat "${CAMPAIGN_ROOT}/summary/completion_audit.json" 2>/dev/null || true
  echo
  echo "## Missing tasks"
  "${PYTHON_BIN}" "${TOOLS}/missing_task_ids.py" --run-root "${RUN_ROOT}" --format json
  echo
  echo "## Job history"
  cat "${history}"
  echo
  echo "## Non-empty Slurm errors"
  find "${CAMPAIGN_ROOT}/slurm" -name '*.err' -type f -size +0c -print0 \
    | xargs -0 -r tail -n 80
} >"${CAMPAIGN_ROOT}/controller/triage_context.md"
invoke_kimi "${TOOLS}/kimi_triage_agent.md" \
  "Read ${CAMPAIGN_ROOT}/controller/triage_context.md and relevant immutable logs. Produce an evidence-based failure audit. Do not edit or resubmit anything." \
  "${CAMPAIGN_ROOT}/controller/KIMI_TRIAGE.md"
touch "${CAMPAIGN_ROOT}/NEEDS_REVIEW"
mark "CAMPAIGN_NEEDS_REVIEW"
exit 1
