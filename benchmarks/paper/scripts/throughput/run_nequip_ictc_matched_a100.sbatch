#!/usr/bin/env bash
#SBATCH --job-name=nequip_ictc_match
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/nequip-ictc-matched-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
BENCH_REPO="${BENCH_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS-OEQ-OPT}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
OEQ_SITE="${OEQ_SITE:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages}"
SOURCE_ROOT="/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_c84_r16_matched_oeq_20260803/xxmd_mal"
OUT="${OUT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/nequip_ictc_matched_${SLURM_JOB_ID}}"
BENCH="${BENCH_REPO}/scripts/benchmark_full_model_step.py"

mkdir -p "${OUT}/configs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: >"${OUT}/results.jsonl"

make_config() {
  local source="$1"
  local target="$2"
  cp "${source}" "${target}"
  sed -i \
    -e 's/^interaction_backend:.*/interaction_backend: ictc/' \
    -e 's/^openequivariance_enabled:.*/openequivariance_enabled: false/' \
    "${target}"
}

make_config "${SOURCE_ROOT}/e3nn_chorus_off_oeq/train.yaml" \
  "${OUT}/configs/off.yaml"
make_config "${SOURCE_ROOT}/e3nn_chorus_on_oeq/train.yaml" \
  "${OUT}/configs/final.yaml"
make_config "${SOURCE_ROOT}/e3nn_chorus_on_persistent_oeq/train.yaml" \
  "${OUT}/configs/persistent.yaml"

run_one() {
  local config="$1"
  local mode="$2"
  PYTHONPATH="${REPO}:${OPERATOR_REPO}:${OEQ_SITE}" "${PYTHON}" \
    "${BENCH}" "${config}" --mode "${mode}" \
    --batch-size 16 --warmup 5 --iterations 20
}

for repeat in 1 2 3; do
  for scope in off final persistent; do
    for mode in inference training; do
      result="$(run_one "${OUT}/configs/${scope}.yaml" "${mode}")"
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
    key = (record["scope"], result["mode"])
    groups.setdefault(key, []).append(result)

summary = {}
for (scope, mode), results in sorted(groups.items()):
    median_ms = statistics.median(
        result["milliseconds_per_iteration"] for result in results
    )
    exemplar = results[0]
    summary[f"{scope}/{mode}"] = {
        "median_ms": median_ms,
        "median_iterations_per_second": 1000.0 / median_ms,
        "median_atoms_per_second": exemplar["atoms"] * 1000.0 / median_ms,
        "parameters": exemplar["parameters"],
        "peak_memory_mib": max(result["peak_memory_mib"] for result in results),
    }

print(json.dumps(summary, indent=2, sort_keys=True))
PY

touch "${OUT}/DONE"
cat "${OUT}/summary.json"
