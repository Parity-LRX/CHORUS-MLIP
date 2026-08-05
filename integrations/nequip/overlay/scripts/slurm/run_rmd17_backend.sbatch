#!/usr/bin/env bash
#SBATCH --job-name=nequip_chorus_rmd17_one
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/rmd17-backend-%j.log

set -euo pipefail

REPO=/public/home/sps-xia/rxlin/NequIP-CHORUS
BACKEND="${BACKEND:?set BACKEND=e3nn or ictc}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/rmd17_c128_r16_20260729}"
EPOCHS="${EPOCHS:-180}"
MOLECULES="${MOLECULES:-revised_aspirin revised_ethanol revised_benzene}"
mkdir -p "${RUN_ROOT}"

case "${BACKEND}" in
  e3nn|ictc) ;;
  *) echo "unsupported BACKEND=${BACKEND}" >&2; exit 2 ;;
esac

device="${CUDA_VISIBLE_DEVICES}"
launch() {
  local chorus="$1"
  env REPO="${REPO}" RUN_ROOT="${RUN_ROOT}" EPOCHS="${EPOCHS}" \
    MOLECULES="${MOLECULES}" \
    bash "${REPO}/scripts/run_rmd17_variant.sh" \
      "${BACKEND}" "${chorus}" "${device}" \
      >"${RUN_ROOT}/${BACKEND}_chorus_${chorus}.driver.log" 2>&1 &
}

# Matched off/on variants share one allocation, as in the original two-GPU
# launcher.  Accuracy is comparable; contended wall time is not a speed result.
launch off; pid_off=$!
launch on; pid_on=$!
status=0
if ! wait "${pid_off}"; then status=1; fi
if ! wait "${pid_on}"; then status=1; fi
if (( status == 0 )); then touch "${RUN_ROOT}/${BACKEND}_ALL_DONE"; fi
exit "${status}"
