#!/usr/bin/env bash
#SBATCH --job-name=nequip_chorus_rmd17
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/rmd17-fourway-%j.log

set -euo pipefail

REPO=/public/home/sps-xia/rxlin/NequIP-CHORUS
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/rmd17_fourway_20260728}"
EPOCHS="${EPOCHS:-180}"
MOLECULES="${MOLECULES:-revised_aspirin revised_ethanol revised_benzene}"
mkdir -p "${RUN_ROOT}"

# Preserve Slurm's physical GPU allocation before assigning devices.  Formal
# runs request two GPUs and place one matched pair on each.  Capacity smokes may
# override the request to one GPU; in that case run the four variants
# sequentially so a free card can be used without multiplying peak memory.
IFS=',' read -r -a allocated_devices <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#allocated_devices[@]} < 1 || ${#allocated_devices[@]} > 2 )); then
  echo "expected one or two allocated GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

launch() {
  local backend="$1"
  local chorus="$2"
  local device="$3"
  local name="${backend}_chorus_${chorus}"
  env REPO="${REPO}" RUN_ROOT="${RUN_ROOT}" EPOCHS="${EPOCHS}" \
    MOLECULES="${MOLECULES}" \
    bash "${REPO}/scripts/run_rmd17_variant.sh" \
      "${backend}" "${chorus}" "${device}" \
      >"${RUN_ROOT}/${name}.driver.log" 2>&1 &
}

status=0
if (( ${#allocated_devices[@]} == 1 )); then
  for backend in e3nn ictc; do
    for chorus in off on; do
      launch "${backend}" "${chorus}" "${allocated_devices[0]}"
      if ! wait "$!"; then status=1; fi
    done
  done
else
  # One A100 is dedicated to the two spherical NequIP variants; the other A100
  # is dedicated to the two ICTC variants.
  launch e3nn off "${allocated_devices[0]}"; pid_e3nn=$!
  launch e3nn on "${allocated_devices[0]}"; pid_e3nn_chorus=$!
  launch ictc off "${allocated_devices[1]}"; pid_ictc=$!
  launch ictc on "${allocated_devices[1]}"; pid_ictc_chorus=$!
  for pid in "${pid_e3nn}" "${pid_e3nn_chorus}" "${pid_ictc}" "${pid_ictc_chorus}"; do
    if ! wait "${pid}"; then status=1; fi
  done
fi
if (( status == 0 )); then
  touch "${RUN_ROOT}/ALL_DONE"
fi
exit "${status}"
