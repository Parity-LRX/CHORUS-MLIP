#!/usr/bin/env bash
#SBATCH --job-name=nequip_chorus_smoke
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/smoke-%j.log

set -euo pipefail

REPO=/public/home/sps-xia/rxlin/NequIP-CHORUS
OPERATOR_REPO=/public/home/sps-xia/rxlin/CHORUS-MLIP-operator
PYTHON=/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python

export PYTHONPATH="${REPO}:${OPERATOR_REPO}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4
cd "${REPO}"

"${PYTHON}" -m pytest tests/unit/nn/test_chorus.py -q

run_variant() {
  local device="$1"
  local backend="$2"
  local chorus="$3"
  CUDA_VISIBLE_DEVICES="${device}" "${PYTHON}" scripts/smoke_variant.py \
    --backend "${backend}" --chorus "${chorus}" \
    >"/public/home/sps-xia/rxlin/nequip_chorus_runs/smoke-${SLURM_JOB_ID}-${backend}-${chorus}.log" 2>&1
}

# Match the intended formal layout: two processes share each A100.
run_variant 0 e3nn off &
pid_e3nn=$!
run_variant 0 e3nn on &
pid_e3nn_chorus=$!
run_variant 1 ictc off &
pid_ictc=$!
run_variant 1 ictc on &
pid_ictc_chorus=$!

status=0
for pid in "${pid_e3nn}" "${pid_e3nn_chorus}" "${pid_ictc}" "${pid_ictc_chorus}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
