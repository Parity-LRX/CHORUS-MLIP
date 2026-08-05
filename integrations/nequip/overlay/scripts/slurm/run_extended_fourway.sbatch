#!/usr/bin/env bash
#SBATCH --job-name=nequip_chorus_extended
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/extended-fourway-%j.log

set -euo pipefail

REPO=/public/home/sps-xia/rxlin/NequIP-CHORUS
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/extended_fourway_20260728}"
DATASETS="${DATASETS:-transition1x xxmd_mal xxmd_sti xxmd_dia md22_buckyball 3bpa}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
mkdir -p "${RUN_ROOT}"

IFS=',' read -r -a allocated_devices <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#allocated_devices[@]} != 2 )); then
  echo "expected exactly two allocated GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

launch() {
  local backend="$1" chorus="$2" device="$3"
  local name="${backend}_chorus_${chorus}"
  env REPO="${REPO}" RUN_ROOT="${RUN_ROOT}" DATASETS="${DATASETS}" \
    EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE}" \
    bash "${REPO}/scripts/run_extended_variant.sh" \
      "${backend}" "${chorus}" "${device}" \
      >"${RUN_ROOT}/${name}.driver.log" 2>&1 &
}

launch e3nn off "${allocated_devices[0]}"; pid_a=$!
launch e3nn on "${allocated_devices[0]}"; pid_b=$!
launch ictc off "${allocated_devices[1]}"; pid_c=$!
launch ictc on "${allocated_devices[1]}"; pid_d=$!

status=0
for pid in "${pid_a}" "${pid_b}" "${pid_c}" "${pid_d}"; do
  if ! wait "${pid}"; then status=1; fi
done
if (( status == 0 )); then touch "${RUN_ROOT}/ALL_DONE"; fi
exit "${status}"
