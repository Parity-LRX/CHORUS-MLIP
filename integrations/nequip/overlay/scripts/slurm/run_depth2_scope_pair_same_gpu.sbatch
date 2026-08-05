#!/usr/bin/env bash
#SBATCH --job-name=nequip_d2_scope
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2-scope-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_scope_c64_r16_20260801}"
DATASETS="${DATASETS:-transition1x xxmd_mal xxmd_sti md22_buckyball 3bpa}"
CHANNELS="${CHANNELS:-64}"
CHORUS_RANK="${CHORUS_RANK:-16}"
SEED="${SEED:-20260616}"

mkdir -p "${RUN_ROOT}"
IFS=',' read -r -a allocated_devices <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#allocated_devices[@]} != 1 )); then
  echo "expected exactly one allocated GPU, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
device="${allocated_devices[0]}"

launch() {
  local scope="$1"
  env \
    REPO="${REPO}" \
    OPERATOR_REPO="${OPERATOR_REPO}" \
    RUN_ROOT="${RUN_ROOT}" \
    DATASETS="${DATASETS}" \
    CHANNELS="${CHANNELS}" \
    CHORUS_RANK="${CHORUS_RANK}" \
    CHORUS_SCOPE="${scope}" \
    NUM_LAYERS=2 \
    SEED="${SEED}" \
    EVAL_BATCH_SIZE=25 \
    bash "${REPO}/scripts/run_extended_variant.sh" \
      e3nn on "${device}" \
      >"${RUN_ROOT}/e3nn_chorus_${scope}.driver.log" 2>&1 &
  LAST_PID=$!
}

launch final
final_pid="${LAST_PID}"
launch persistent
persistent_pid="${LAST_PID}"
status=0
if ! wait "${final_pid}"; then status=1; fi
if ! wait "${persistent_pid}"; then status=1; fi
if (( status == 0 )); then
  touch "${RUN_ROOT}/ALL_DONE"
fi
exit "${status}"
