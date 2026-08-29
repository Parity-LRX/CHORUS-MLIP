#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-ms25-20260818}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/chorus_data/ms25_20260818/prepared}"
RUN_ROOT="${RUN_ROOT:-/home/ylzhang/chorus_runs/ms25_external_20260818/dpa4}"

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

while screen_exists ms25_dpa4; do
  sleep 30
done
while [[ ! -f /home/ylzhang/chorus_data/ms25_20260818/MGO4X4_TRANSFER_DONE ]]; do
  sleep 10
done

for channels in 32 48; do
  run="${RUN_ROOT}/mgo-2x2/c${channels}_mix3_fp32_compiled"
  "${DPA_ENV}/bin/python" \
    "${REPO}/benchmarks/paper/scripts/training/evaluate_dpa4_xxmd_force_mae.py" \
    --run "${run}" --data "${DATA_ROOT}/mgo-2x2/set1/deepmd" \
    --out "${run}/full_val_force_mae_eval" --steps 45000 \
    --checkpoint-every 1 --dp "${DPA_ENV}/bin/dp" \
    --extra-test-system \
    "mgo_4x4_size_transfer=${DATA_ROOT}/mgo-4x4-transfer/set1/deepmd/test" \
    >"${run}/full_val_force_mae_eval/transfer_update.log" 2>&1
done

touch "${RUN_ROOT}/MGO4X4_TRANSFER_EVAL_DONE"
echo "MS25_DPA4_MGO4X4_TRANSFER_EVAL_DONE $(date -Is)"
