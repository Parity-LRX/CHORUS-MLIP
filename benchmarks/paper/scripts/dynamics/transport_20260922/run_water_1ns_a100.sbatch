#!/usr/bin/env bash
#SBATCH --job-name=water-1ns
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/water_1ns_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/water_1ns_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
MFF_PYTHON="${MFF_PYTHON:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
BUILD_ROOT="${BUILD_ROOT:-${ROOT}/deploy/mfftorch_lammps_20260924}"
DPA_ENV="${DPA_ENV:-/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling}"
TORCH211_SITE="${TORCH211_SITE:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/dpa4_torch211_overlay}"
MPI_COMPAT="${MPI_COMPAT:-${ROOT}/assets/openmpi4_compat}"
STEPS="${STEPS:-1000000}"
SEED="${SEED:-20260616}"
INITIAL="${ROOT}/data/prepared/water/extxyz/test.extxyz"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

labels=(mace_phaseoff mace_final mace_persistent dpa4_c32 dpa4_c48)
label="${labels[${TASK_ID}]}"
out="${ROOT}/md/water_1ns/${label}/seed${SEED}"
mkdir -p "${out}"

if [[ -f "${out}/DONE" ]]; then
  echo "already complete: ${out}/DONE"
  exit 0
fi

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

if (( TASK_ID < 3 )); then
  modes=(phaseoff final persistent)
  mode="${modes[${TASK_ID}]}"
  model="${BUILD_ROOT}/models/water_${mode}_dynamic.pt2"
  lmp="${BUILD_ROOT}/build-mfftorch/lmp"
  prepare="${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/prepare_mfftorch_lammps_production.py"

  [[ -x "${lmp}" ]]
  [[ -f "${BUILD_ROOT}/BUILD_DONE" ]]
  [[ -f "${model}" ]]
  [[ -f "${model}.meta" ]]
  [[ -f "${BUILD_ROOT}/smoke_water_${mode}_plain/DONE" ]]
  [[ -f "${INITIAL}" ]]

  torch_lib="$(${MFF_PYTHON} -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  nvidia_lib_dirs="$(${MFF_PYTHON} - <<'PY'
import pathlib
import site

root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)"
  export LD_LIBRARY_PATH="${torch_lib}:${nvidia_lib_dirs}:${LD_LIBRARY_PATH:-}"

  "${MFF_PYTHON}" "${prepare}" \
    --model "${model}" --initial-structure "${INITIAL}" \
    --ase-state "${out}/restart.json" --output-dir "${out}" \
    --elements H O --temperature-k 300 --seed "${SEED}" \
    --steps "${STEPS}" --repeat 2,2,2 --pair-style mff/torch

  if [[ -f "${out}/DONE" ]]; then
    exit 0
  fi

  MFF_DEBUG_BUNDLE=1 "${lmp}" \
    -log "${out}/thermo_lammps.log" \
    -in "${out}/in.mfftorch_production" \
    2>&1 | tee -a "${out}/lammps_stdout.log"
  grep -q "MFF_PRODUCTION_DONE" "${out}/lammps_stdout.log"
  touch "${out}/DONE_LAMMPS" "${out}/DONE"
else
  widths=(c32 c48)
  width="${widths[$((TASK_ID - 3))]}"
  metrics="${ROOT}/runs/dpa4/water/${width}/full_val_force_mae_eval/metrics.json"
  checkpoint="$(${DPA_ENV}/bin/python - "${metrics}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])
PY
)"
  [[ -f "${checkpoint}" ]]
  [[ -f "${INITIAL}" ]]

  torch_lib="${TORCH211_SITE}/torch/lib"
  cuda_libs=""
  while IFS= read -r libdir; do
    cuda_libs="${cuda_libs:+${cuda_libs}:}${libdir}"
  done < <(find "${TORCH211_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
  export LD_LIBRARY_PATH="${MPI_COMPAT}:${torch_lib}:${cuda_libs}:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${TORCH211_SITE}:${PYTHONPATH:-}"

  "${DPA_ENV}/bin/python" \
    "${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/run_carnet_dpa_md.py" \
    --checkpoint "${checkpoint}" --initial-structure "${INITIAL}" \
    --output-dir "${out}" --temperature-k 300 --seed "${SEED}" \
    --steps "${STEPS}" --repeat 2,2,2
fi

echo "WATER_1NS_DONE label=${label} seed=${SEED} steps=${STEPS} $(date -Is)"
