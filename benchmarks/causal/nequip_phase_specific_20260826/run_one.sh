#!/usr/bin/env bash
set -euo pipefail

task_id="${1:?usage: run_one.sh TASK_ID}"

NEQUIP_REPO="${NEQUIP_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
CHORUS_REPO="${CHORUS_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
MOLECULAR_DATA="${MOLECULAR_DATA:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
SOLID_DATA="${SOLID_DATA:-/public/home/sps-xia/rxlin/chorus_solid_data/ms25_20260818/prepared}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_20260826}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
CHORUS_SCOPE_OVERRIDE="${CHORUS_SCOPE_OVERRIDE:-final}"

datasets=(xxmd_mal md22_buckyball 3bpa zr-o)
modes=(off diagonal aligned full)
seeds=(20260616 20260617 20260618 20260619 20260620)

if (( task_id < 0 || task_id >= 80 )); then
  echo "invalid task id: ${task_id}" >&2
  exit 2
fi

seed_index=$((task_id / 16))
within_seed=$((task_id % 16))
dataset_index=$((within_seed / 4))
mode_index=$((within_seed % 4))
seed="${seeds[$seed_index]}"
dataset="${datasets[$dataset_index]}"
mode="${modes[$mode_index]}"

chorus=on
density_pairs="${mode}"
if [[ "${mode}" == "off" ]]; then
  chorus=off
  density_pairs=full
fi

task_root="${CAMPAIGN_ROOT}/seed_${seed}/${mode}/${dataset}"
status_root="${CAMPAIGN_ROOT}/status/task_${task_id}"
mkdir -p "${task_root}" "${status_root}" "${CAMPAIGN_ROOT}/slurm"
if [[ -f "${status_root}/DONE" ]]; then
  echo "SKIP task=${task_id} seed=${seed} dataset=${dataset} mode=${mode}"
  exit 0
fi

on_error() {
  status=$?
  printf '%s\n' "status=${status}" "time=$(date -Is)" >"${status_root}/FAILED"
  exit "${status}"
}
trap on_error ERR

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4
export NEQUIP_NUM_TASKS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${NEQUIP_REPO}:${CHORUS_REPO}:${PYTHONPATH:-}"

cat >"${status_root}/identity.txt" <<EOF
task_id=${task_id}
seed=${seed}
dataset=${dataset}
mode=${mode}
chorus=${chorus}
density_pairs=${density_pairs}
phase_scope=${CHORUS_SCOPE_OVERRIDE}
started=$(date -Is)
EOF

if [[ "${dataset}" != "zr-o" ]]; then
  export REPO="${NEQUIP_REPO}"
  export OPERATOR_REPO="${CHORUS_REPO}"
  export PYTHON="${PYTHON_BIN}"
  export DATA_ROOT="${MOLECULAR_DATA}"
  export RUN_ROOT="${task_root}"
  export DATASETS="${dataset}"
  export CHANNELS=64
  export CHORUS_RANK=16
  export CHORUS_DENSITY_PAIRS="${density_pairs}"
  export CHORUS_SCOPE="${CHORUS_SCOPE_OVERRIDE}"
  export NUM_LAYERS=2
  export LMAX=2
  export SEED="${seed}"
  export EVAL_BATCH_SIZE=25
  export OPENEQUIVARIANCE=0
  export EPOCHS_OVERRIDE
  bash "${NEQUIP_REPO}/scripts/run_extended_variant.sh" e3nn "${chorus}" 0
else
  scope="${CHORUS_SCOPE_OVERRIDE}"
  if [[ "${chorus}" == "off" ]]; then
    scope=off
  fi
  work="${task_root}/e3nn_chorus_${mode}"
  base="${SOLID_DATA}/zr-o/set1"
  epochs=180
  if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
    epochs="${EPOCHS_OVERRIDE}"
  fi
  mkdir -p "${work}"
  "${PYTHON_BIN}" "${CHORUS_REPO}/benchmarks/paper/scripts/solids/write_ms25_nequip_config.py" \
    --metadata "${base}/metadata.json" \
    --data-dir "${base}/chorus" \
    --output-dir "${work}" \
    --scope "${scope}" --channels 64 --rank 16 \
    --chorus-density-pairs "${density_pairs}" --disable-openequivariance \
    --epochs "${epochs}" --batch-size 4 --seed "${seed}" \
    >"${work}/config.stdout"

  if [[ ! -f "${work}/TRAIN_DONE" ]]; then
    cd "${NEQUIP_REPO}"
    "${PYTHON_BIN}" -m nequip.scripts.train "${work}/train.yaml" \
      >"${work}/train.log" 2>&1
    touch "${work}/TRAIN_DONE"
  fi
  if [[ ! -f "${work}/TEST_DONE" ]]; then
    cd "${NEQUIP_REPO}"
    "${PYTHON_BIN}" -m nequip.scripts.evaluate \
      --train-dir "${work}/run" \
      --dataset-config "${work}/test_test.yaml" \
      --metrics-config "${work}/test_test.yaml" \
      --batch-size 4 --device cuda \
      --log "${work}/test_test.log" \
      >"${work}/test_test.stdout" 2>&1
    touch "${work}/TEST_DONE"
  fi
  "${PYTHON_BIN}" "${CHORUS_REPO}/benchmarks/paper/scripts/solids/summarize_ms25_nequip.py" \
    --work-dir "${work}" --test-log "${work}/test_test.log" \
    --train-log "${work}/train.log" --steps-per-epoch 250 \
    --output "${work}/metrics.json"

  calibration="${work}/train_only_energy_calibration"
  if [[ ! -f "${calibration}/DONE" ]]; then
    mkdir -p "${calibration}"
    for split in train val test; do
      cd "${NEQUIP_REPO}"
      "${PYTHON_BIN}" -m nequip.scripts.evaluate \
        --model "${work}/run/best_model.pth" \
        --dataset-config "${work}/calibration_${split}.yaml" \
        --metrics-config None --batch-size 4 --device cuda \
        --output "${calibration}/${split}_predicted.xyz" \
        --output-fields-from-original-dataset total_energy \
        --log "${calibration}/${split}_predict.log" \
        >"${calibration}/${split}_predict.stdout" 2>&1
    done
    "${PYTHON_BIN}" "${CHORUS_REPO}/benchmarks/paper/scripts/solids/calibrate_nequip_energy_predictions.py" \
      --train "${calibration}/train_predicted.xyz" \
      --val "${calibration}/val_predicted.xyz" \
      --test "${calibration}/test_predicted.xyz" \
      --elements O,Zr --selected-checkpoint "${work}/run/best_model.pth" \
      --output "${calibration}/metrics.json"
    touch "${calibration}/DONE"
  fi
fi

rm -f "${status_root}/FAILED"
printf '%s\n' "completed=$(date -Is)" >"${status_root}/DONE"
echo "DONE task=${task_id} seed=${seed} dataset=${dataset} mode=${mode}"
