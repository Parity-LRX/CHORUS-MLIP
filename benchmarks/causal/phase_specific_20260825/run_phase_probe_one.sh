#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:?set REPO}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
SOLID_DATA_ROOT="${SOLID_DATA_ROOT:-/public/home/sps-xia/rxlin/chorus_solid_data/ms25_20260818/prepared}"
PRIMARY_ROOT="${PRIMARY_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/runs}"
PROBE_ROOT="${PROBE_ROOT:?set PROBE_ROOT}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
TOOLS="${REPO}/benchmarks/causal/phase_specific_20260825"
PROTOCOL_PATH="${PROTOCOL_PATH:-${TOOLS}/protocol.json}"
SOURCE_STATE_PATH="${SOURCE_STATE_PATH:-${TOOLS}/SOURCE_STATE}"

case "${DATASET}" in
  mal)
    data_dir="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal"
    elements="H,C,O"; batch_size=16; tests=(test); splits=(test)
    ;;
  bucky)
    data_dir="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed"
    elements="H,C"; batch_size=4; tests=(test); splits=(test)
    ;;
  3bpa)
    data_dir="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5"
    elements="H,C,N,O"; batch_size=16
    tests=(300K 600K 1200K); splits=(test_300K test_600K test_1200K)
    ;;
  zro)
    data_dir="${SOLID_DATA_ROOT}/zr-o/set1"
    elements="Zr,O"; batch_size=4; tests=(test); splits=(test)
    ;;
  *) echo "unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac

checkpoint="${PRIMARY_ROOT}/${DATASET}/seed${SEED}/full/checkpoints/model.pth"
[[ -f "${checkpoint}" ]] || { echo "missing ${checkpoint}" >&2; exit 3; }
out="${PROBE_ROOT}/${DATASET}/seed${SEED}"
mkdir -p "${out}"
rm -f "${out}/FAILED"; touch "${out}/RUNNING"
mark() { printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${out}/status.log"; }
on_exit() {
  rc=$?; trap - EXIT; rm -f "${out}/RUNNING"
  if (( rc != 0 )); then printf 'exit_code=%s\n' "${rc}" >"${out}/FAILED"; mark "FAILED exit=${rc}"; fi
  exit "${rc}"
}
trap on_exit EXIT

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

for index in "${!tests[@]}"; do
  test_name="${tests[$index]}"; split="${splits[$index]}"; test_out="${out}/${test_name}"
  if [[ -f "${test_out}/DONE" ]]; then continue; fi
  mark "PROBE_START_${test_name}"
  "${PYTHON_BIN}" "${TOOLS}/evaluate_phase_interventions.py" \
    --checkpoint "${checkpoint}" --data-dir "${data_dir}" --split "${split}" \
    --elements "${elements}" --dataset "${DATASET}" --seed "${SEED}" \
    --test-label "${test_name}" --batch-size "${batch_size}" \
    --permutation-repeats 4 --device cuda --output-dir "${test_out}" \
    >"${test_out}.log" 2>&1
  mark "PROBE_DONE_${test_name}"
done

test_csv=$(IFS=,; echo "${tests[*]}")
"${PYTHON_BIN}" - "${PROTOCOL_PATH}" "${out}/run_manifest.json" \
  "${checkpoint}" "${test_csv}" <<PY
import hashlib, json, pathlib, platform, sys
p, out, ckpt = map(pathlib.Path, sys.argv[1:4])
payload = {
  "dataset": "${DATASET}", "seed": int("${SEED}"), "checkpoint": str(ckpt),
  "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
  "protocol": str(p), "protocol_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
  "source_state": pathlib.Path("${SOURCE_STATE_PATH}").read_text().strip(),
  "host": platform.node(), "tests": sys.argv[4].split(","),
}
out.write_text(json.dumps(payload, indent=2) + "\n")
PY

touch "${out}/DONE"; rm -f "${out}/RUNNING" "${out}/FAILED"; trap - EXIT; mark "DONE"
