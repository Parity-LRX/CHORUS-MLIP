#!/usr/bin/env bash
#SBATCH --job-name=chorus-causal-synth
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_causal_results/scope_backbone_20260828/slurm-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_causal_results/scope_backbone_20260828/slurm-%j.err

set -euo pipefail

REPO="/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730"
PYTHON="/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python"
OUTPUT="/public/home/sps-xia/rxlin/chorus_causal_results/scope_backbone_20260828"
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
  "${REPO}/benchmarks/causal/phase_specific_20260825/analyse_scope_backbone_interactions.py" \
  --mace-final-primary \
  /public/home/sps-xia/rxlin/chorus_causal_results/mace_final_inputs/primary_modes.csv \
  --mace-final-aligned \
  /public/home/sps-xia/rxlin/chorus_causal_results/mace_final_inputs/aligned_runs.csv \
  --mace-persistent-root \
  /public/home/sps-xia/rxlin/chorus_persistent_causal_20260826 \
  --nequip-final-root \
  /public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826 \
  --nequip-persistent-root \
  /public/home/sps-xia/rxlin/nequip_chorus_runs/causal_persistent_phase_specific_20260826 \
  --output "${OUTPUT}" --bootstrap 100000

touch "${OUTPUT}/DONE"
rm -f "${OUTPUT}/FAILED"
trap - EXIT
