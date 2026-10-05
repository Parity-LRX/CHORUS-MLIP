#!/usr/bin/env bash
#SBATCH --job-name=transport-dpa4
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/dpa4_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/dpa4_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
DPA_ENV="${DPA_ENV:-/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling}"
TORCH211_SITE="${TORCH211_SITE:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/dpa4_torch211_overlay}"
MPI_COMPAT="${MPI_COMPAT:-${ROOT}/assets/openmpi4_compat}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/prepared}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/dpa4}"
STEPS="${STEPS:-45000}"
RUN_EVAL="${RUN_EVAL:-1}"

datasets=(water lips)
channels_list=(32 48)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[$((task_id / 2))]}"
channels="${channels_list[$((task_id % 2))]}"
data="${DATA_ROOT}/${dataset}/deepmd"
out="${RUN_ROOT}/${dataset}/c${channels}"

[[ -f "${ROOT}/DATA_READY" ]]
mkdir -p "${out}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -f "${TORCH211_SITE}/torch/__init__.py" ]]
[[ -f "${MPI_COMPAT}/libmpi_cxx.so.40" ]]
torch_lib="${TORCH211_SITE}/torch/lib"
cuda_libs=""
while IFS= read -r libdir; do
  cuda_libs="${cuda_libs:+${cuda_libs}:}${libdir}"
done < <(find "${TORCH211_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="${MPI_COMPAT}:${torch_lib}:${cuda_libs}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${TORCH211_SITE}:${PYTHONPATH:-}"

DATA="${data}" OUT="${out}" STEPS="${STEPS}" BATCH_SIZE=4 \
CHANNELS="${channels}" MIXING_LAYERS=3 RCUT=5.0 SEED=20260616 \
USE_COMPILE=1 USE_AMP=0 DPA_ENV="${DPA_ENV}" \
BASE_CONFIG="${REPO}/benchmarks/paper/external/dpa4/buckyball_master_286f12d.json" \
REPO="${REPO}" \
  bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_fixed_xxmd.sh"

eval_dir="${out}/full_val_force_mae_eval"
if [[ "${RUN_EVAL}" == "1" && ! -f "${eval_dir}/DONE" ]]; then
  "${DPA_ENV}/bin/python" \
    "${REPO}/benchmarks/paper/scripts/training/evaluate_dpa4_xxmd_force_mae.py" \
    --run "${out}" --data "${data}" --out "${eval_dir}" \
    --steps "${STEPS}" --dp "${DPA_ENV}/bin/dp"
fi

echo "TRANSPORT_DPA4_DONE dataset=${dataset} channels=${channels} $(date -Is)"
