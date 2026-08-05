#!/usr/bin/env bash
set -euo pipefail

BACKEND="$1"
CHORUS="$2"
DEVICE="$3"

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-operator}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/extended_fourway_20260728}"
DATASETS="${DATASETS:-transition1x xxmd_mal xxmd_sti xxmd_dia md22_buckyball 3bpa}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
CHANNELS="${CHANNELS:-64}"
CHORUS_RANK="${CHORUS_RANK:-16}"
CHORUS_SCOPE="${CHORUS_SCOPE:-final}"
NUM_LAYERS="${NUM_LAYERS:-3}"
SEED="${SEED:-20260616}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
REUSE_COMPLETED_TRAIN="${REUSE_COMPLETED_TRAIN:-0}"
OPENEQUIVARIANCE="${OPENEQUIVARIANCE:-0}"

if [[ "${CHORUS}" == "on" && "${CHORUS_SCOPE}" != "final" ]]; then
  variant="${BACKEND}_chorus_${CHORUS}_${CHORUS_SCOPE}"
else
  variant="${BACKEND}_chorus_${CHORUS}"
fi
if [[ "${OPENEQUIVARIANCE}" == "1" ]]; then
  variant="${variant}_oeq"
fi
export CUDA_VISIBLE_DEVICES="${DEVICE}"
export PYTHONPATH="${REPO}:${OPERATOR_REPO}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4
export NEQUIP_NUM_TASKS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${REPO}"

for dataset in ${DATASETS}; do
  work="${RUN_ROOT}/${dataset}/${variant}"
  train_dir="${work}/run"
  mkdir -p "${work}"
  if [[ -f "${work}/DONE" ]]; then
    printf 'SKIP %s %s\n' "${dataset}" "${variant}"
    continue
  fi
  config_args=(
    --dataset "${dataset}" --backend "${BACKEND}" --chorus "${CHORUS}" \
    --data-root "${DATA_ROOT}" --run-root "${work}" --output-dir "${work}" \
    --channels "${CHANNELS}" --chorus-rank "${CHORUS_RANK}"
    --chorus-scope "${CHORUS_SCOPE}"
    --num-layers "${NUM_LAYERS}"
    --seed "${SEED}"
  )
  if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
    config_args+=(--epochs-override "${EPOCHS_OVERRIDE}")
  fi
  if [[ "${OPENEQUIVARIANCE}" == "1" ]]; then
    config_args+=(--openequivariance)
  fi
  "${PYTHON}" scripts/write_extended_config.py "${config_args[@]}"
  printf 'START %s %s %s\n' "${dataset}" "${variant}" "$(date -Is)"
  if [[ "${REUSE_COMPLETED_TRAIN}" == "1" && -f "${train_dir}/last_model.pth" ]]; then
    printf 'REUSE_TRAIN %s %s %s\n' "${dataset}" "${variant}" "$(date -Is)"
  else
    "${PYTHON}" -m nequip.scripts.train "${work}/train.yaml" \
      >"${work}/train.log" 2>&1
  fi
  for test_config in "${work}"/test_*.yaml; do
    label="$(basename "${test_config}" .yaml)"
    "${PYTHON}" -m nequip.scripts.evaluate \
      --train-dir "${train_dir}" \
      --dataset-config "${test_config}" \
      --metrics-config "${test_config}" \
      --batch-size "${EVAL_BATCH_SIZE}" --device cuda \
      --log "${work}/${label}.log" \
      >"${work}/${label}.stdout" 2>&1
  done
  if [[ "${dataset}" == "transition1x" ]]; then
    "${PYTHON}" scripts/calibrate_t1x_energy.py \
      --train-dir "${train_dir}" \
      --train-config "${work}/train.yaml" \
      --test-config "${work}/test_test.yaml" \
      --output "${work}/train_only_energy_calibration.json" \
      >"${work}/calibration.stdout" 2>&1
  fi
  touch "${work}/DONE"
  printf 'DONE %s %s %s\n' "${dataset}" "${variant}" "$(date -Is)"
done
