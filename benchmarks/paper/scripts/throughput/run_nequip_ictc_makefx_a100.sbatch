#!/usr/bin/env bash
#SBATCH --job-name=nequip_ictc_mfx
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/nequip-ictc-makefx-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
OEQ_SITE="${OEQ_SITE:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages}"
CONFIG_ROOT="${CONFIG_ROOT:?CONFIG_ROOT is required}"
BENCH="${REPO}/scripts/bench_nequip_ictc_makefx.py"
OUT="${OUT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/nequip_ictc_makefx_${SLURM_JOB_ID}}"
SCOPES="${SCOPES:-off final persistent}"
REPEATS="${REPEATS:-3}"

mkdir -p "${OUT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: >"${OUT}/results.jsonl"

run_one() {
  local config="$1"
  local mode="$2"
  PYTHONPATH="${REPO}:${OPERATOR_REPO}:${OEQ_SITE}" "${PYTHON}" \
    "${BENCH}" "${config}" --mode "${mode}" \
    --batch-size 16 --warmup 5 --iterations 20
}

for repeat in $(seq 1 "${REPEATS}"); do
  for scope in ${SCOPES}; do
    for mode in inference training; do
      result="$(run_one "${CONFIG_ROOT}/${scope}.yaml" "${mode}")"
      printf '{"scope":"%s","repeat":%d,"result":%s}\n' \
        "${scope}" "${repeat}" "${result}" >>"${OUT}/results.jsonl"
      printf '%s\n' "${result}"
    done
  done
done

"${PYTHON}" - "${OUT}/results.jsonl" >"${OUT}/summary.json" <<'PY'
import json
import statistics
import sys

records = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
groups = {}
for record in records:
    result = record["result"]
    groups.setdefault((record["scope"], result["mode"]), []).append(result)
summary = {}
for (scope, mode), results in sorted(groups.items()):
    median_ms = statistics.median(
        result["milliseconds_per_iteration"] for result in results
    )
    first = results[0]
    summary[f"{scope}/{mode}"] = {
        "median_ms": median_ms,
        "median_iterations_per_second": 1000.0 / median_ms,
        "median_atoms_per_second": first["atoms"] * 1000.0 / median_ms,
        "parameters": first["parameters"],
        "peak_memory_mib": max(result["peak_memory_mib"] for result in results),
        "median_compile_seconds": statistics.median(
            result["compile_seconds"] for result in results
        ),
    }
print(json.dumps(summary, indent=2, sort_keys=True))
PY

touch "${OUT}/DONE"
cat "${OUT}/summary.json"
