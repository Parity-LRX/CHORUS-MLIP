#!/usr/bin/env bash
#SBATCH --job-name=ms25-nequip
#SBATCH --partition=GPU-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-8%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm/nequip_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm/nequip_%A_%a.err

set -euo pipefail

CHORUS_REPO="${CHORUS_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-ms25-20260818}"
NEQUIP_REPO="${NEQUIP_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS-OEQ-OPT}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
OEQ_SITE="${OEQ_SITE:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages}"
TORCH211_SITE="${TORCH211_SITE:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/dpa4_torch211_overlay}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/chorus_solid_data/ms25_20260818/prepared}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/nequip_family}"

datasets=(hea zr-o mgo-2x2)
scopes=(off final persistent)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[$((task_id / 3))]}"
scope="${scopes[$((task_id % 3))]}"
base="${DATA_ROOT}/${dataset}/set1"
work="${RUN_ROOT}/${dataset}/${scope}"
train_dir="${work}/run"

mkdir -p "${RUN_ROOT}/slurm" "${work}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4
export NEQUIP_NUM_TASKS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -f "${TORCH211_SITE}/torch/__init__.py" ]]
torch_lib="${TORCH211_SITE}/torch/lib"
cuda_libs=""
while IFS= read -r libdir; do
  cuda_libs="${cuda_libs:+${cuda_libs}:}${libdir}"
done < <(find "${TORCH211_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="${torch_lib}:${cuda_libs}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TORCH211_SITE}:${NEQUIP_REPO}:${CHORUS_REPO}:${OEQ_SITE}:${PYTHONPATH:-}"

config_extra=()
if [[ "${dataset}" == "mgo-2x2" ]]; then
  config_extra=(--size-transfer-data-dir "${DATA_ROOT}/mgo-4x4-transfer/set1/chorus")
fi
"${PYTHON_BIN}" "${CHORUS_REPO}/benchmarks/paper/scripts/solids/write_ms25_nequip_config.py" \
  --metadata "${base}/metadata.json" \
  --data-dir "${base}/chorus" \
  --output-dir "${work}" \
  --scope "${scope}" --channels 84 --rank 16 \
  --epochs 180 --batch-size 4 --seed 20260616 \
  "${config_extra[@]}"

if [[ ! -f "${work}/TRAIN_DONE" ]]; then
  cd "${NEQUIP_REPO}"
  "${PYTHON_BIN}" -m nequip.scripts.train "${work}/train.yaml" \
    >"${work}/train.log" 2>&1
  touch "${work}/TRAIN_DONE"
fi

if [[ ! -f "${work}/DONE" ]]; then
  cd "${NEQUIP_REPO}"
  "${PYTHON_BIN}" -m nequip.scripts.evaluate \
    --train-dir "${train_dir}" \
    --dataset-config "${work}/test_test.yaml" \
    --metrics-config "${work}/test_test.yaml" \
    --batch-size 4 --device cuda \
    --log "${work}/test_test.log" \
    >"${work}/test_test.stdout" 2>&1
  touch "${work}/DONE"
fi

summary_extra=()
if [[ "${dataset}" == "mgo-2x2" ]]; then
  if [[ ! -f "${work}/MGO4X4_DONE" ]]; then
    cd "${NEQUIP_REPO}"
    "${PYTHON_BIN}" -m nequip.scripts.evaluate \
      --train-dir "${train_dir}" \
      --dataset-config "${work}/test_mgo4x4.yaml" \
      --metrics-config "${work}/test_mgo4x4.yaml" \
      --batch-size 4 --device cuda \
      --log "${work}/test_mgo4x4.log" \
      >"${work}/test_mgo4x4.stdout" 2>&1
    touch "${work}/MGO4X4_DONE"
  fi
  summary_extra=(--size-transfer-log "${work}/test_mgo4x4.log")
fi
"${PYTHON_BIN}" "${CHORUS_REPO}/benchmarks/paper/scripts/solids/summarize_ms25_nequip.py" \
  --work-dir "${work}" --test-log "${work}/test_test.log" \
  --train-log "${work}/train.log" --steps-per-epoch 250 \
  "${summary_extra[@]}" --output "${work}/metrics.json"

if [[ "${dataset}" == "zr-o" ]]; then
  calibration="${work}/train_only_energy_calibration"
  if [[ ! -f "${calibration}/DONE" ]]; then
    mkdir -p "${calibration}"
    for split in train val test; do
      cd "${NEQUIP_REPO}"
      "${PYTHON_BIN}" -m nequip.scripts.evaluate \
        --model "${train_dir}/best_model.pth" \
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
      --elements O,Zr --selected-checkpoint "${train_dir}/best_model.pth" \
      --output "${calibration}/metrics.json"
    touch "${calibration}/DONE"
  fi
fi

echo "MS25_NEQUIP_DONE dataset=${dataset} scope=${scope} $(date -Is)"
