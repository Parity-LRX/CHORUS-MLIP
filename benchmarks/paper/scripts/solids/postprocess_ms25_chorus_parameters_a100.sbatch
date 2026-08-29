#!/usr/bin/env bash
#SBATCH --job-name=ms25-param-audit
#SBATCH --partition=CPU-AMD
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm/parameter_audit_%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm/parameter_audit_%j.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-ms25-20260818}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/mace_ictc_family}"

for dataset in hea zr-o mgo-2x2; do
  for mode in phaseoff final persistent; do
    run="${RUN_ROOT}/${dataset}/${mode}"
    eval_dir="${run}/validation_force_selected_eval"
    [[ -f "${eval_dir}/DONE" ]]
    "${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/parse_chorus_test.py" \
      --selection "${eval_dir}/selection.json" \
      --train-log "${run}/logs/train.log" \
      --test-log "${eval_dir}/test.log" \
      --output "${eval_dir}/metrics.json"

    if [[ "${dataset}" == "mgo-2x2" ]]; then
      transfer_dir="${run}/mgo_4x4_size_transfer_eval"
      [[ -f "${transfer_dir}/DONE" ]]
      "${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/parse_chorus_test.py" \
        --selection "${eval_dir}/selection.json" \
        --train-log "${run}/logs/train.log" \
        --test-log "${transfer_dir}/test.log" \
        --output "${transfer_dir}/metrics.json"
    fi
  done
done

touch "${RUN_ROOT}/PARAMETER_AUDIT_DONE"
echo "MS25_CHORUS_PARAMETER_AUDIT_DONE $(date -Is)"
