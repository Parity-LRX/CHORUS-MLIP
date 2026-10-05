#!/usr/bin/env bash
#SBATCH --job-name=transport-nequip
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/nequip_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/nequip_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
CHORUS_REPO="${CHORUS_REPO:-${ROOT}/source/CHORUS-MLIP}"
NEQUIP_REPO="${NEQUIP_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS-OEQ-OPT}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
OEQ_SITE="${OEQ_SITE:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages}"
TORCH211_SITE="${TORCH211_SITE:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/dpa4_torch211_overlay}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/prepared}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/nequip_family}"
EPOCHS="${EPOCHS:-180}"
RUN_EVAL="${RUN_EVAL:-1}"

datasets=(water lips)
scopes=(off final persistent)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[$((task_id / 3))]}"
scope="${scopes[$((task_id % 3))]}"
base="${DATA_ROOT}/${dataset}"
work="${RUN_ROOT}/${dataset}/${scope}"
train_dir="${work}/run"

[[ -f "${ROOT}/DATA_READY" ]]
mkdir -p "${work}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4
export NEQUIP_NUM_TASKS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -f "${TORCH211_SITE}/torch/__init__.py" ]]
torch_lib="${TORCH211_SITE}/torch/lib"
cuda_libs=""
while IFS= read -r libdir; do
  cuda_libs="${cuda_libs:+${cuda_libs}:}${libdir}"
done < <(find "${TORCH211_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="${torch_lib}:${cuda_libs}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TORCH211_SITE}:${NEQUIP_REPO}:${CHORUS_REPO}:${OEQ_SITE}:${PYTHONPATH:-}"

"${PYTHON_BIN}" \
  "${CHORUS_REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/write_transport_nequip_config.py" \
  --metadata "${base}/metadata.json" --data-dir "${base}" \
  --output-dir "${work}" --scope "${scope}" \
  --channels 84 --rank 16 --epochs "${EPOCHS}" --batch-size 4 --seed 20260616

if [[ ! -f "${work}/TRAIN_DONE" ]]; then
  cd "${NEQUIP_REPO}"
  "${PYTHON_BIN}" -m nequip.scripts.train "${work}/train.yaml" \
    >"${work}/train.log" 2>&1
  touch "${work}/TRAIN_DONE"
fi

if [[ "${RUN_EVAL}" == "1" && ! -f "${work}/DONE" ]]; then
  cd "${NEQUIP_REPO}"
  "${PYTHON_BIN}" -m nequip.scripts.evaluate \
    --train-dir "${train_dir}" --dataset-config "${work}/test.yaml" \
    --metrics-config "${work}/test.yaml" --batch-size 4 --device cuda \
    --log "${work}/test.log" >"${work}/test.stdout" 2>&1
  "${PYTHON_BIN}" "${CHORUS_REPO}/benchmarks/paper/scripts/solids/summarize_ms25_nequip.py" \
    --work-dir "${work}" --test-log "${work}/test.log" \
    --train-log "${work}/train.log" --steps-per-epoch 250 \
    --output "${work}/metrics.json"
  touch "${work}/DONE"
fi

echo "TRANSPORT_NEQUIP_DONE dataset=${dataset} scope=${scope} $(date -Is)"
