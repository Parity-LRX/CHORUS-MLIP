#!/usr/bin/env bash
#SBATCH --job-name=carnet-md-mace
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-29%4
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/md_mace_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/md_mace_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
STEPS="${STEPS:-300000}"
MAX_CONCURRENT_NOTE="four A100 array tasks; precision results may be compared, wall time is not a throughput result"

datasets=(water lips)
modes=(phaseoff final persistent)
seeds=(20260616 20260617 20260618 20260619 20260620)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
seed_index="$((task_id % 5))"
mode_index="$(((task_id / 5) % 3))"
dataset_index="$((task_id / 15))"
dataset="${datasets[${dataset_index}]}"
mode="${modes[${mode_index}]}"
seed="${seeds[${seed_index}]}"

case "${dataset}" in
  water)
    temperature=300
    repeat="2,2,2"
    ;;
  lips)
    temperature=520
    repeat="2,2,2"
    ;;
  *)
    echo "unknown dataset ${dataset}" >&2
    exit 2
    ;;
esac

metrics="${ROOT}/runs/mace_family/${dataset}/${mode}/validation_force_selected_eval/metrics.json"
checkpoint="$(${PYTHON_BIN} - "${metrics}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_checkpoint"])
PY
)"
initial="${ROOT}/data/prepared/${dataset}/extxyz/test.extxyz"
out="${ROOT}/md/carnet_protocol/mace_family/${dataset}/${mode}/seed${seed}"

[[ -f "${metrics}" ]]
[[ -f "${checkpoint}" ]]
[[ -f "${initial}" ]]
mkdir -p "${out}"
export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

"${PYTHON_BIN}" \
  "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_carnet_md.py" \
  --checkpoint "${checkpoint}" --initial-structure "${initial}" \
  --output-dir "${out}" --temperature-k "${temperature}" \
  --seed "${seed}" --steps "${STEPS}" --repeat "${repeat}"

echo "CARNET_MACE_MD_DONE dataset=${dataset} mode=${mode} seed=${seed} ${MAX_CONCURRENT_NOTE} $(date -Is)"
