#!/usr/bin/env bash
#SBATCH --job-name=neq_phase_probe
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-39%2
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_frozen_probes_20260829/slurm-%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_frozen_probes_20260829/slurm-%A_%a.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
FINAL_ROOT="${FINAL_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_v2_20260826}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_persistent_phase_specific_20260826}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_frozen_probes_20260829}"

scopes=(final persistent)
datasets=(xxmd_mal md22_buckyball 3bpa zr-o)
labels=(mal bucky 3bpa zro)
seeds=(20260616 20260617 20260618 20260619 20260620)

task="${SLURM_ARRAY_TASK_ID:?}"
scope_index=$((task / 20))
within=$((task % 20))
dataset_index=$((within / 5))
seed_index=$((within % 5))
scope="${scopes[$scope_index]}"
dataset="${datasets[$dataset_index]}"
label="${labels[$dataset_index]}"
seed="${seeds[$seed_index]}"

if [[ "${scope}" == "final" ]]; then
  source_root="${FINAL_ROOT}"
else
  source_root="${PERSISTENT_ROOT}"
fi
base="${source_root}/seed_${seed}/full/${dataset}"
metrics_csv="$(find "${base}" -type f -path '*/run/metrics_epoch.csv' -print -quit)"
if [[ -z "${metrics_csv}" ]]; then
  echo "missing completed training directory under ${base}" >&2
  exit 2
fi
run_dir="$(dirname "${metrics_csv}")"
work="$(dirname "${run_dir}")"

if [[ "${dataset}" == "3bpa" ]]; then
  test_config="${work}/test_300K.yaml"
  batch_size=64
elif [[ "${dataset}" == "md22_buckyball" ]]; then
  test_config="${work}/test_test.yaml"
  batch_size=64
elif [[ "${dataset}" == "zr-o" ]]; then
  test_config="${work}/test_test.yaml"
  batch_size=64
else
  test_config="${work}/test_test.yaml"
  batch_size=128
fi

output="${OUTPUT_ROOT}/${scope}/${label}/seed${seed}"
mkdir -p "${output}"
if [[ -f "${output}/DONE" ]]; then
  echo "SKIP ${scope} ${label} ${seed}"
  exit 0
fi

export PYTHONPATH="${REPO}:${OPERATOR_REPO}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4
export NEQUIP_NUM_TASKS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" "${REPO}/scripts/evaluate_phase_interventions.py" \
  --work-dir "${work}" \
  --test-config "${test_config}" \
  --scope "${scope}" \
  --dataset "${label}" \
  --seed "${seed}" \
  --batch-size "${batch_size}" \
  --device cuda \
  --output-dir "${output}"

echo "DONE ${scope} ${label} ${seed}"
