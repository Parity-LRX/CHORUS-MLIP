#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
DPA_DATA_ROOT="${DPA_DATA_ROOT:-/home/ylzhang/lrx/xxmd/deepmd_temporal_r5}"
DPA_RUN_ROOT="${DPA_RUN_ROOT:-/home/ylzhang/chorus_runs/large_scale_main_20260724/external/dpa4_xxmd_large}"
TECE_RUN_ROOT="${TECE_RUN_ROOT:-/home/ylzhang/tace_chorus_runs/xxmd_tece_large_mae_seed20260616}"
STATUS="${STATUS:-/home/ylzhang/chorus_runs/large_scale_main_20260724/external/queue_status.log}"

mkdir -p "$(dirname "${STATUS}")"

while screen -ls | grep -Eq '\.(xxmd_dia_eval_queue|prepare_dpa_xxmd)'; do
  sleep 30
done

for system in mal sti dia; do
  echo "START DPA4 ${system} $(date -Is)" | tee -a "${STATUS}"
  DATA="${DPA_DATA_ROOT}/${system}" \
  OUT="${DPA_RUN_ROOT}/${system}_c32_mix3" \
  STEPS=45000 BATCH_SIZE=16 CHANNELS=32 MIXING_LAYERS=3 RCUT=5.0 \
  SEED=20260616 \
    bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_fixed_xxmd.sh"
  echo "END DPA4 ${system} $(date -Is)" | tee -a "${STATUS}"
done

echo "START TECE all $(date -Is)" | tee -a "${STATUS}"
RUN_ROOT="${TECE_RUN_ROOT}" CHANNELS=36 \
  bash "${REPO}/benchmarks/paper/external/tece/run_tece_xxmd_queue.sh"
echo "END TECE all $(date -Is)" | tee -a "${STATUS}"
touch "$(dirname "${STATUS}")/ALL_EXTERNAL_DONE"
