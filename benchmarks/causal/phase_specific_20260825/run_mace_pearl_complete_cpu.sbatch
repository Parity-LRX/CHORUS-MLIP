#!/usr/bin/env bash
#SBATCH --job-name=mace-pearl-complete
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_causal_results/mace_pearl_complete_20260828/slurm-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_causal_results/mace_pearl_complete_20260828/slurm-%j.err

set -euo pipefail

REPO="/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730"
PYTHON="/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python"
SOURCE="/public/home/sps-xia/rxlin/chorus_causal_results/mace_final_pearl_source_20260828"
OUTPUT="/public/home/sps-xia/rxlin/chorus_causal_results/mace_pearl_complete_20260828"
mkdir -p "${OUTPUT}"
rm -f "${OUTPUT}/DONE" "${OUTPUT}/FAILED"

on_exit() {
  status=$?
  if (( status != 0 )); then
    printf 'exit_code=%s\ntime=%s\n' "${status}" "$(date -Is)" >"${OUTPUT}/FAILED"
  fi
}
trap on_exit EXIT

"${PYTHON}" \
  "${REPO}/benchmarks/causal/phase_specific_20260825/analyse_mace_pearl_complete.py" \
  --final-result "${SOURCE}" \
  --final-probe-root /public/home/sps-xia/rxlin/chorus_phase_causal_20260825/probes \
  --persistent-root /public/home/sps-xia/rxlin/chorus_persistent_causal_20260826 \
  --output "${OUTPUT}" \
  --bootstrap 100000 \
  --association-permutations 100000

touch "${OUTPUT}/DONE"
rm -f "${OUTPUT}/FAILED"
trap - EXIT
