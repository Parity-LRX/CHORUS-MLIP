#!/usr/bin/env bash
# Run two backend controls concurrently inside one A100 allocation.
# The shared-GPU wall time is for scheduling only, not throughput reporting.
#SBATCH --job-name=nequip_c72_same_gpu
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/c72-same-gpu-%j.log

set -euo pipefail

REPO=/public/home/sps-xia/rxlin/NequIP-CHORUS
BACKENDS="${BACKENDS:-e3nn ictc}"
DATASETS="${DATASETS:-transition1x xxmd_mal xxmd_sti xxmd_dia md22_buckyball 3bpa}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/extended_c72_r16_samegpu_20260729}"
CHANNELS="${CHANNELS:-72}"
CHORUS_RANK="${CHORUS_RANK:-16}"
device="${CUDA_VISIBLE_DEVICES}"

mkdir -p "${RUN_ROOT}"
status=0
for backend in ${BACKENDS}; do
  env REPO="${REPO}" RUN_ROOT="${RUN_ROOT}" DATASETS="${DATASETS}" \
    CHANNELS="${CHANNELS}" CHORUS_RANK="${CHORUS_RANK}" \
    bash "${REPO}/scripts/run_extended_variant.sh" \
      "${backend}" off "${device}" \
      >"${RUN_ROOT}/${backend}_off.driver.log" 2>&1 &
done
for pid in $(jobs -p); do
  if ! wait "${pid}"; then status=1; fi
done
if (( status == 0 )); then touch "${RUN_ROOT}/ALL_DONE"; fi
exit "${status}"
