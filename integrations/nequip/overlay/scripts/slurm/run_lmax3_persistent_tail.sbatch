#!/usr/bin/env bash
#SBATCH --job-name=nequip-l3-p-tail
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/lmax3-persistent-tail-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/lmax_controls_c64_r16_20260815/L3}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"

IFS=',' read -r -a allocated_devices <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#allocated_devices[@]} != 1 )); then
  echo "expected exactly one allocated GPU, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
device="${allocated_devices[0]}"

failed_dir="${RUN_ROOT}/md22_buckyball/e3nn_chorus_on_persistent"
archive_dir="${RUN_ROOT}/md22_buckyball/e3nn_chorus_on_persistent_failed_parallel_oom_20260816"
if [[ -d "${failed_dir}" && ! -f "${failed_dir}/DONE" ]]; then
  if [[ -e "${archive_dir}" ]]; then
    echo "refusing to overwrite existing OOM archive: ${archive_dir}" >&2
    exit 2
  fi
  mv "${failed_dir}" "${archive_dir}"
fi

env \
  REPO="${REPO}" \
  OPERATOR_REPO="${OPERATOR_REPO}" \
  PYTHON="${PYTHON}" \
  RUN_ROOT="${RUN_ROOT}" \
  DATASETS="md22_buckyball 3bpa" \
  CHANNELS=64 \
  CHORUS_RANK=16 \
  CHORUS_SCOPE=persistent \
  NUM_LAYERS=2 \
  LMAX=3 \
  SEED=20260616 \
  EVAL_BATCH_SIZE=25 \
  OPENEQUIVARIANCE=0 \
  bash "${REPO}/scripts/run_extended_variant.sh" e3nn on "${device}"

touch "${RUN_ROOT}/PERSISTENT_TAIL_DONE"
