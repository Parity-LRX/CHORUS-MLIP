#!/usr/bin/env bash
#SBATCH --job-name=carnet-md-dpa-smoke
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%1
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/md_dpa_smoke_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/md_dpa_smoke_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
DPA_ENV="${DPA_ENV:-/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling}"
TORCH211_SITE="${TORCH211_SITE:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/dpa4_torch211_overlay}"
MPI_COMPAT="${MPI_COMPAT:-${ROOT}/assets/openmpi4_compat}"

datasets=(water lips)
widths=(c32 c48)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[$((task_id / 2))]}"
width="${widths[$((task_id % 2))]}"

case "${dataset}" in
  water) temperature=300 ;;
  lips) temperature=520 ;;
  *) exit 2 ;;
esac

metrics="${ROOT}/runs/dpa4/${dataset}/${width}/full_val_force_mae_eval/metrics.json"
checkpoint="$(${DPA_ENV}/bin/python - "${metrics}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])
PY
)"
initial="${ROOT}/data/prepared/${dataset}/extxyz/test.extxyz"
out="${ROOT}/smoke/carnet_protocol/dpa4/${dataset}/${width}"

[[ -f "${checkpoint}" ]]
[[ -f "${initial}" ]]
mkdir -p "${out}"
torch_lib="${TORCH211_SITE}/torch/lib"
cuda_libs=""
while IFS= read -r libdir; do
  cuda_libs="${cuda_libs:+${cuda_libs}:}${libdir}"
done < <(find "${TORCH211_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="${MPI_COMPAT}:${torch_lib}:${cuda_libs}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TORCH211_SITE}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

rm -f "${out}/DONE" "${out}/restart.json" "${out}/progress.json" \
  "${out}/trajectory.traj" "${out}/thermo.log" "${out}/manifest.json"
"${DPA_ENV}/bin/python" \
  "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_carnet_dpa_md.py" \
  --checkpoint "${checkpoint}" --initial-structure "${initial}" \
  --output-dir "${out}" --temperature-k "${temperature}" \
  --seed 20260616 --steps 20 --repeat 2,2,2

echo "CARNET_DPA_SMOKE_DONE dataset=${dataset} width=${width} $(date -Is)"
