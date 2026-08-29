#!/usr/bin/env bash
set -euo pipefail

pair_task_id="${1:?usage: run_nequip_pair_one.sh TASK_ID}"
if (( pair_task_id < 0 || pair_task_id >= 20 )); then
  echo "invalid persistent pair task id: ${pair_task_id}" >&2
  exit 2
fi
seed_index=$((pair_task_id / 4))
dataset_index=$((pair_task_id % 4))
base_task_id=$((seed_index * 16 + dataset_index * 4))
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_persistent_phase_specific_20260826}"
export CHORUS_SCOPE_OVERRIDE=persistent

# Train Full first so the paired learned-phase checkpoint always exists before
# the Aligned control is finalized.
bash "${REPO}/benchmarks/causal/nequip_phase_specific_20260826/run_one.sh" \
  "$((base_task_id + 3))"
bash "${REPO}/benchmarks/causal/nequip_phase_specific_20260826/run_one.sh" \
  "$((base_task_id + 2))"
