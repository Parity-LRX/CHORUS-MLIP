#!/usr/bin/env bash
#SBATCH --job-name=nequip_persistent
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/persistent-c64-r16-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/persistent_c64_r16_20260801}"
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
  local backend="$1"
  env \
    REPO="${REPO}" \
    OPERATOR_REPO="${OPERATOR_REPO}" \
    RUN_ROOT="${RUN_ROOT}" \
    DATASETS="${DATASETS}" \
    CHANNELS="${CHANNELS}" \
    CHORUS_RANK="${CHORUS_RANK}" \
    CHORUS_SCOPE=persistent \
    SEED="${SEED}" \
    EVAL_BATCH_SIZE=25 \
    bash "${REPO}/scripts/run_extended_variant.sh" \
      "${backend}" on "${device}" \
      >"${RUN_ROOT}/${backend}_chorus_persistent.driver.log" 2>&1 &
  LAST_PID=$!
}

launch e3nn
e3nn_pid="${LAST_PID}"
launch ictc
ictc_pid="${LAST_PID}"
status=0
if ! wait "${e3nn_pid}"; then status=1; fi
if ! wait "${ictc_pid}"; then status=1; fi
if (( status == 0 )); then
  touch "${RUN_ROOT}/ALL_DONE"
fi
exit "${status}"
