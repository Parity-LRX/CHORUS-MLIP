#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-phase-causal-20260825}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/chorus_phase_causal_20260825}"
PRIMARY_ROOT="${PRIMARY_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/runs}"
PROBE_ROOT="${PROBE_ROOT:-${CAMPAIGN_ROOT}/probes}"
ALIGNED_ROOT="${ALIGNED_ROOT:-${CAMPAIGN_ROOT}/aligned_runs}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
KIMI="${KIMI:-/public/home/sps-xia/rxlin/.kimi-code/bin/kimi}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
MAX_RETRY_ROUNDS="${MAX_RETRY_ROUNDS:-2}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TOOLS="${REPO}/benchmarks/causal/phase_specific_20260825"

mkdir -p "${CAMPAIGN_ROOT}/controller" "${CAMPAIGN_ROOT}/summary" \
  "${CAMPAIGN_ROOT}/slurm" "${PROBE_ROOT}" "${ALIGNED_ROOT}"
exec 9>"${CAMPAIGN_ROOT}/controller/controller.lock"
if ! flock -n 9; then
  echo "another controller holds the phase campaign lock" >&2
  exit 9
fi

status="${CAMPAIGN_ROOT}/controller/status.log"
history="${CAMPAIGN_ROOT}/controller/job_history.tsv"
stage_file="${CAMPAIGN_ROOT}/controller/stage"
touch "${history}"

mark() {
  printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${status}"
}

set_stage() {
  printf '%s\n' "$1" >"${stage_file}"
  mark "STAGE $1"
}

preflight() {
  set_stage preflight
  "${KIMI}" doctor >/dev/null
  for file in SOURCE_STATE protocol.json causal_dag.md causal_dag.dot; do
    [[ -f "${TOOLS}/${file}" ]] || { echo "missing ${TOOLS}/${file}" >&2; return 1; }
  done
  export PYTHONPATH="${REPO}:/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16"
  export PYTHONDONTWRITEBYTECODE=1
  "${PYTHON_BIN}" -m py_compile \
    "${TOOLS}/evaluate_phase_interventions.py" \
    "${TOOLS}/summarize.py" "${TOOLS}/build_progress_snapshot.py"
  "${PYTHON_BIN}" -m pytest -q "${REPO}/chorus/test/test_phase_hermitian.py" \
    -k 'aligned or coefficient_controls' >/dev/null
  "${PYTHON_BIN}" - "${TOOLS}/protocol.json" "${PRIMARY_ROOT}" <<'PY'
import json, pathlib, sys
p = json.load(open(sys.argv[1]))
root = pathlib.Path(sys.argv[2])
for dataset in p["datasets"]:
    for seed in p["seeds"]:
        base = root / dataset / f"seed{seed}" / "full"
        assert (base / "DONE").is_file(), base
        assert (base / "metrics.json").is_file(), base
        assert (base / "checkpoints" / "model.pth").is_file(), base
PY
  mark "PREFLIGHT_DONE"
}

wait_for_job() {
  local job_id="$1" label="$2"
  while squeue -h -j "${job_id}" | grep -q .; do
    mark "ARRAY_RUNNING label=${label} job=${job_id}"
    sleep "${POLL_SECONDS}"
  done
  sacct -j "${job_id}" --format=JobIDRaw,State,ExitCode,Elapsed,NodeList \
    -n -P >"${CAMPAIGN_ROOT}/controller/sacct_${job_id}.tsv" || true
  mark "ARRAY_EXITED label=${label} job=${job_id}"
}

submit_array() {
  local kind="$1" ids="$2" label="$3" script job_id
  if [[ "${kind}" == probe ]]; then
    script="${TOOLS}/run_probe_array_a100.sbatch"
    job_id=$(sbatch --parsable --array="${ids}%${MAX_CONCURRENT_GPUS}" \
      --export=ALL,REPO="${REPO}",PROBE_ROOT="${PROBE_ROOT}",PRIMARY_ROOT="${PRIMARY_ROOT}" \
      "${script}")
  else
    script="${TOOLS}/run_aligned_array_a100.sbatch"
    job_id=$(sbatch --parsable --array="${ids}%${MAX_CONCURRENT_GPUS}" \
      --export=ALL,REPO="${REPO}",ALIGNED_ROOT="${ALIGNED_ROOT}",PRIMARY_ROOT="${PRIMARY_ROOT}" \
      "${script}")
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" \
    "${kind}" "${label}" "${job_id}" "${ids}" >>"${history}"
  mark "ARRAY_SUBMITTED kind=${kind} label=${label} job=${job_id} ids=${ids}" >&2
  printf '%s\n' "${job_id}"
}

run_stage() {
  local kind="$1" root="$2" round=0 ids job_id
  set_stage "${kind}"
  while true; do
    ids=$("${PYTHON_BIN}" "${TOOLS}/missing_task_ids.py" --root "${root}" --kind "${kind}")
    if [[ -z "${ids}" ]]; then
      mark "STAGE_COMPLETE kind=${kind}"
      return 0
    fi
    if (( round > MAX_RETRY_ROUNDS )); then
      mark "RETRY_LIMIT_REACHED kind=${kind} ids=${ids}"
      return 1
    fi
    job_id=$(submit_array "${kind}" "${ids}" "round${round}")
    wait_for_job "${job_id}" "${kind}"
    round=$((round + 1))
  done
}

invoke_kimi() {
  local agent_file="$1" prompt="$2" output="$3"
  set +e
  (
    cd "${REPO}"
    KIMI_MODEL_THINKING_EFFORT=max KIMI_MODEL_THINKING_KEEP=all \
      timeout 90m "${KIMI}" -m kimi-code-api/k3 \
      --add-dir "${CAMPAIGN_ROOT}" --add-dir "${PRIMARY_ROOT%/runs}" \
      --agent-file "${agent_file}" --prompt "${prompt}"
  ) >"${output}" 2>"${output%.md}.stderr"
  local rc=$?
  set -e
  return "${rc}"
}

triage_and_stop() {
  set_stage needs_review
  {
    echo "# Phase campaign failure context"
    echo
    echo "## Job history"
    cat "${history}"
    echo
    echo "## Failure markers"
    find "${CAMPAIGN_ROOT}" -name FAILED -type f -print -exec cat {} \;
    echo
    echo "## Non-empty Slurm errors"
    find "${CAMPAIGN_ROOT}/slurm" -name '*.err' -type f -size +0c -print0 \
      | xargs -0 -r tail -n 120
  } >"${CAMPAIGN_ROOT}/controller/triage_context.md"
  invoke_kimi "${TOOLS}/kimi_triage_agent.md" \
    "Audit the immutable failure context and logs. Diagnose only; do not edit, cancel, or resubmit jobs." \
    "${CAMPAIGN_ROOT}/controller/KIMI_TRIAGE.md" || true
  touch "${CAMPAIGN_ROOT}/NEEDS_REVIEW"
  mark "CAMPAIGN_NEEDS_REVIEW"
  exit 1
}

preflight || triage_and_stop
if [[ -f "${CAMPAIGN_ROOT}/DONE" ]]; then
  set_stage done
  mark "ALREADY_DONE"
  exit 0
fi

run_stage probe "${PROBE_ROOT}" || triage_and_stop
run_stage aligned "${ALIGNED_ROOT}" || triage_and_stop

set_stage summarizing
if ! "${PYTHON_BIN}" "${TOOLS}/summarize.py" \
  --primary-root "${PRIMARY_ROOT}" --aligned-root "${ALIGNED_ROOT}" \
  --probe-root "${PROBE_ROOT}" --output-dir "${CAMPAIGN_ROOT}/summary" \
  >"${CAMPAIGN_ROOT}/controller/summarize.log" 2>&1; then
  triage_and_stop
fi

set_stage kimi_final_audit
invoke_kimi "${TOOLS}/kimi_causal_analyst.md" \
  "Audit the completed phase-specific campaign against protocol.json and causal_dag.md. Use summary and paired outcome files. Return a rigorous Markdown report, including negative results and identification limits." \
  "${CAMPAIGN_ROOT}/summary/KIMI_PHASE_CAUSAL_REPORT.md" || true
touch "${CAMPAIGN_ROOT}/DONE"
rm -f "${CAMPAIGN_ROOT}/NEEDS_REVIEW"
set_stage done
mark "CAMPAIGN_DONE"
