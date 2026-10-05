#!/usr/bin/env bash
#SBATCH --job-name=transport-smoke
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/smoke_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/smoke_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
SMOKE_ROOT="${ROOT}/smoke"
task_id="${SLURM_ARRAY_TASK_ID:-0}"

case "${task_id}" in
  0)
    env SLURM_ARRAY_TASK_ID=2 RUN_ROOT="${SMOKE_ROOT}/mace" \
      MAX_STEPS=2 RUN_EVAL=0 \
      bash "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_transport_mace_family_a100.sbatch"
    ;;
  1)
    env SLURM_ARRAY_TASK_ID=0 RUN_ROOT="${SMOKE_ROOT}/native_mace" \
      EPOCHS=1 RUN_EVAL=0 \
      bash "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_transport_native_mace_a100.sbatch"
    ;;
  2)
    env SLURM_ARRAY_TASK_ID=2 RUN_ROOT="${SMOKE_ROOT}/nequip" \
      EPOCHS=1 RUN_EVAL=0 \
      bash "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_transport_nequip_family_a100.sbatch"
    ;;
  3)
    env SLURM_ARRAY_TASK_ID=0 RUN_ROOT="${SMOKE_ROOT}/dpa4" \
      STEPS=2 RUN_EVAL=0 \
      bash "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_transport_dpa4_a100.sbatch"
    ;;
  4)
    env SLURM_ARRAY_TASK_ID=0 RUN_ROOT="${SMOKE_ROOT}/tece" \
      STEPS=2 EPOCHS=1 RUN_EVAL=0 \
      bash "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_transport_tece_a100.sbatch"
    ;;
  *)
    echo "unknown smoke task ${task_id}" >&2
    exit 2
    ;;
esac

echo "TRANSPORT_SMOKE_DONE engine=${task_id} $(date -Is)"
