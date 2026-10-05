#!/usr/bin/env bash
#SBATCH --job-name=carnet-md-mace-lmp
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --array=0-29%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/md_mace_lmp_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/md_mace_lmp_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
BUILD_ROOT="${BUILD_ROOT:-${ROOT}/deploy/mfftorch_lammps_20260924}"
PAIR_STYLE="${PAIR_STYLE:-mff/torch}"
if [[ "${PAIR_STYLE}" == "mff/torch/kk" ]]; then
  LMP="${BUILD_ROOT}/build-mfftorch-kk/lmp"
  BUILD_MARKER="${BUILD_ROOT}/BUILD_KK_DONE"
  backend_tag=kk
else
  LMP="${BUILD_ROOT}/build-mfftorch/lmp"
  BUILD_MARKER="${BUILD_ROOT}/BUILD_DONE"
  backend_tag=plain
fi
STEPS="${STEPS:-300000}"
PREPARE="${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/prepare_mfftorch_lammps_production.py"

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
    elements=(H O)
    temperature=300
    ;;
  lips)
    elements=(Li P S)
    temperature=520
    ;;
  *)
    echo "unknown dataset ${dataset}" >&2
    exit 2
    ;;
esac

MODEL="${BUILD_ROOT}/models/${dataset}_${mode}_dynamic.pt2"
SMOKE_DONE="${BUILD_ROOT}/smoke_${dataset}_${mode}_${backend_tag}/DONE"
INITIAL="${ROOT}/data/prepared/${dataset}/extxyz/test.extxyz"
# Keep the single-engine LAMMPS production ensemble separate from the earlier
# ASE pilot.  This avoids mixing integrators or partially converted states in
# the reported transport statistics.
OUT="${ROOT}/md/carnet_protocol_lammps/mace_family/${dataset}/${mode}/seed${seed}"
ASE_STATE="${OUT}/restart.json"

[[ -x "${LMP}" ]]
[[ -f "${BUILD_MARKER}" ]]
[[ -f "${MODEL}" ]]
[[ -f "${MODEL}.meta" ]]
[[ -f "${SMOKE_DONE}" ]]
[[ -f "${INITIAL}" ]]
mkdir -p "${OUT}"

if [[ -f "${OUT}/DONE" ]]; then
  echo "already complete: ${OUT}/DONE"
  exit 0
fi

TORCH_LIB="$(${PYTHON_BIN} -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
NVIDIA_LIB_DIRS="$(${PYTHON_BIN} - <<'PY'
import pathlib
import site

root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

"${PYTHON_BIN}" "${PREPARE}" \
  --model "${MODEL}" --initial-structure "${INITIAL}" \
  --ase-state "${ASE_STATE}" --output-dir "${OUT}" \
  --elements "${elements[@]}" --temperature-k "${temperature}" \
  --seed "${seed}" --steps "${STEPS}" --repeat 2,2,2 \
  --pair-style "${PAIR_STYLE}"

if [[ -f "${OUT}/DONE" ]]; then
  exit 0
fi

if [[ "${PAIR_STYLE}" == "mff/torch/kk" ]]; then
  MFF_DEBUG_BUNDLE=1 "${LMP}" \
    -k on g 1 -sf kk -pk kokkos newton off neigh full \
    -log "${OUT}/thermo_lammps.log" \
    -in "${OUT}/in.mfftorch_production" \
    2>&1 | tee -a "${OUT}/lammps_stdout.log"
else
  MFF_DEBUG_BUNDLE=1 "${LMP}" \
    -log "${OUT}/thermo_lammps.log" \
    -in "${OUT}/in.mfftorch_production" \
    2>&1 | tee -a "${OUT}/lammps_stdout.log"
fi

grep -q "MFF_PRODUCTION_DONE" "${OUT}/lammps_stdout.log"
touch "${OUT}/DONE_LAMMPS" "${OUT}/DONE"
echo "CARNET_MACE_LAMMPS_DONE dataset=${dataset} mode=${mode} seed=${seed} $(date -Is)"
