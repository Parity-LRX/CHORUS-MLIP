#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:?set REPO}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
SOLID_DATA_ROOT="${SOLID_DATA_ROOT:-/public/home/sps-xia/rxlin/chorus_solid_data/ms25_20260818/prepared}"
PRIMARY_ROOT="${PRIMARY_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/runs}"
OBS_ROOT="${OBS_ROOT:?set OBS_ROOT}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
TOOLS="${REPO}/benchmarks/causal/phase_specific_20260825"

case "${DATASET}" in
  mal)
    data_dir="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal"
    elements="H,C,O"; batch_size=16; split=test
    ;;
  bucky)
    data_dir="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed"
    elements="H,C"; batch_size=4; split=test
    ;;
  3bpa)
    data_dir="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5"
    elements="H,C,N,O"; batch_size=16; split=test_300K
    ;;
  zro)
    data_dir="${SOLID_DATA_ROOT}/zr-o/set1"
    elements="Zr,O"; batch_size=4; split=test
    ;;
  *) echo "unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac

checkpoint="${PRIMARY_ROOT}/${DATASET}/seed${SEED}/full/checkpoints/model.pth"
[[ -f "${checkpoint}" ]] || { echo "missing ${checkpoint}" >&2; exit 3; }
out="${OBS_ROOT}/${DATASET}/seed${SEED}"
if [[ -f "${out}/DONE" ]]; then
  printf 'SKIP %s seed=%s\n' "${DATASET}" "${SEED}"
  exit 0
fi
mkdir -p "${out}"
rm -f "${out}/FAILED"
touch "${out}/RUNNING"
on_exit() {
  rc=$?
  trap - EXIT
  rm -f "${out}/RUNNING"
  if (( rc != 0 )); then printf 'exit_code=%s\n' "${rc}" >"${out}/FAILED"; fi
  exit "${rc}"
}
trap on_exit EXIT

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

"${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/training/diagnose_phase_interference.py" \
  --checkpoint "${checkpoint}" \
  --data-dir "${data_dir}" \
  --elements "${elements}" \
  --split "${split}" \
  --dataset-label "${DATASET}" \
  --batch-size "${batch_size}" \
  --max-batches 100000000 \
  --intervention-batches 0 \
  --bins 72 \
  --device cuda \
  --output-dir "${out}" \
  >"${out}/run.log" 2>&1

"${PYTHON_BIN}" - "${TOOLS}/observational_phase_protocol.json" "${out}/run_manifest.json" "${checkpoint}" <<PY
import hashlib, json, pathlib, platform, sys
protocol, output, checkpoint = map(pathlib.Path, sys.argv[1:4])
payload = {
    "dataset": "${DATASET}",
    "seed": int("${SEED}"),
    "split": "${split}",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "protocol": str(protocol),
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "host": platform.node(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

touch "${out}/DONE"
rm -f "${out}/RUNNING" "${out}/FAILED"
trap - EXIT
printf 'DONE %s seed=%s\n' "${DATASET}" "${SEED}"
