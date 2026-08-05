#!/usr/bin/env bash
#SBATCH --job-name=chorus_t1x_d3_pair
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_depth3_scope_20260801/slurm-%x-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
SEED="${SEED:-20260617}"
ROOT_BASE="${ROOT_BASE:-/public/home/sps-xia/rxlin/chorus_depth3_scope_20260801}"
SYSTEM_GROUP="${SYSTEM_GROUP:-t1x}"
case "${SYSTEM_GROUP}" in
  t1x) SYSTEMS="t1x" ;;
  remaining) SYSTEMS="mal,sti,bucky,3bpa" ;;
  *) echo "SYSTEM_GROUP must be t1x or remaining, got ${SYSTEM_GROUP}" >&2; exit 2 ;;
esac
RUNNER="${REPO}/benchmarks/paper/scripts/training/run_t1x_depth3_scope_a100.sbatch"
LOG_TAG="$(printf '%s' "${SYSTEMS}" | tr ',' '_')"

mkdir -p "${ROOT_BASE}/pair_logs"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

echo "START $(date --iso-8601=seconds)"
echo "JOB_ID=${SLURM_JOB_ID:-}"
echo "PAIR=final,persistent"
echo "SEED=${SEED}"
echo "SYSTEMS=${SYSTEMS}"
echo "SHARED_GPU=true"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

MODE=chorus SEED="${SEED}" SYSTEMS="${SYSTEMS}" REPO="${REPO}" ROOT_BASE="${ROOT_BASE}" \
  bash "${RUNNER}" >"${ROOT_BASE}/pair_logs/final_${LOG_TAG}.log" 2>&1 &
final_pid=$!

MODE=persistent SEED="${SEED}" SYSTEMS="${SYSTEMS}" REPO="${REPO}" ROOT_BASE="${ROOT_BASE}" \
  bash "${RUNNER}" >"${ROOT_BASE}/pair_logs/persistent_${LOG_TAG}.log" 2>&1 &
persistent_pid=$!

set +e
wait "${final_pid}"
final_rc=$?
wait "${persistent_pid}"
persistent_rc=$?
set -e

echo "FINAL_RC=${final_rc}"
echo "PERSISTENT_RC=${persistent_rc}"
if (( final_rc != 0 || persistent_rc != 0 )); then
  exit 1
fi

touch "${ROOT_BASE}/PAIR_${LOG_TAG}_DONE"
echo "ALL_DONE $(date --iso-8601=seconds)"
