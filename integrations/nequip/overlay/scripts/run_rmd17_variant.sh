#!/usr/bin/env bash
set -euo pipefail

BACKEND="$1"
CHORUS="$2"
DEVICE="$3"

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-operator}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/rmd17_seed20260616}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/rmd17_fourway_20260728}"
EPOCHS="${EPOCHS:-180}"
MOLECULES="${MOLECULES:-revised_aspirin revised_ethanol revised_benzene}"
CHANNELS="${CHANNELS:-64}"

variant="${BACKEND}_chorus_${CHORUS}"
export CUDA_VISIBLE_DEVICES="${DEVICE}"
export PYTHONPATH="${REPO}:${OPERATOR_REPO}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4
# The Slurm launcher prebuilds shared graph caches. Avoid creating a fallback
# multiprocessing pool if a cache is unexpectedly missing.
export NEQUIP_NUM_TASKS=1
cd "${REPO}"

for molecule in ${MOLECULES}; do
  work="${RUN_ROOT}/${molecule}/${variant}"
  train_dir="${work}/run"
  mkdir -p "${work}"
  if [[ -f "${work}/DONE" ]]; then
    printf 'SKIP %s %s\n' "${molecule}" "${variant}"
    continue
  fi
  "${PYTHON}" scripts/write_rmd17_config.py \
    --molecule "${molecule}" --backend "${BACKEND}" --chorus "${CHORUS}" \
    --data-root "${DATA_ROOT}" --run-root "${work}" \
    --output "${work}/train.yaml" --test-output "${work}/test.yaml" \
    --epochs "${EPOCHS}" --channels "${CHANNELS}"
  printf 'START %s %s %s\n' "${molecule}" "${variant}" "$(date -Is)"
  "${PYTHON}" -m nequip.scripts.train "${work}/train.yaml" \
    >"${work}/train.log" 2>&1
  "${PYTHON}" -m nequip.scripts.evaluate \
    --train-dir "${train_dir}" \
    --dataset-config "${work}/test.yaml" \
    --metrics-config "${work}/test.yaml" \
    --batch-size 50 --device cuda \
    --log "${work}/test.log" \
    >"${work}/test.stdout" 2>&1
  touch "${work}/DONE"
  printf 'DONE %s %s %s\n' "${molecule}" "${variant}" "$(date -Is)"
done
