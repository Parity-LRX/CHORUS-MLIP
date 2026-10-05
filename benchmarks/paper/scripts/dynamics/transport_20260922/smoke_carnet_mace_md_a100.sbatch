#!/usr/bin/env bash
#SBATCH --job-name=carnet-md-smoke
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/md_smoke_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/md_smoke_%A_%a.err

set -euo pipefail

ROOT="/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922"
REPO="${ROOT}/source/CHORUS-MLIP"
PYTHON_BIN="/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python"
MACE_TORCH_PATH="/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16"
datasets=(water lips)
dataset="${datasets[${SLURM_ARRAY_TASK_ID:-0}]}"
if [[ "${dataset}" == water ]]; then
  temperature=300
else
  temperature=520
fi
metrics="${ROOT}/runs/mace_family/${dataset}/persistent/validation_force_selected_eval/metrics.json"
checkpoint="$(${PYTHON_BIN} - "${metrics}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_checkpoint"])
PY
)"
out="${ROOT}/md/smoke/mace_family/${dataset}/persistent"
rm -rf "${out}"
mkdir -p "${out}"
export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

"${PYTHON_BIN}" \
  "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_carnet_md.py" \
  --checkpoint "${checkpoint}" \
  --initial-structure "${ROOT}/data/prepared/${dataset}/extxyz/test.extxyz" \
  --output-dir "${out}" --temperature-k "${temperature}" \
  --seed 20260616 --steps 20 --repeat 2,2,2 \
  --log-interval 1 --trajectory-interval 5 --restart-interval 10

echo "CARNET_MD_SMOKE_DONE dataset=${dataset} $(date -Is)"
