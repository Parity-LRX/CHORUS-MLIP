#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ylzhang/lrx/chorus_water_1ns_20260926}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
LMP="${LMP:-/home/ylzhang/lrx/lammps-stable_22Jul2025/build-mfftorch/lmp}"
PREPARE="${ROOT}/source/prepare_mfftorch_lammps_production.py"
INITIAL="${ROOT}/assets/data/test.extxyz"
SEED="${SEED:-20260616}"
STEPS="${STEPS:-1000000}"

mkdir -p "${ROOT}/logs" "${ROOT}/smoke" "${ROOT}/md/water_1ns"
exec 9>"${ROOT}/water_1ns.lock"
if ! flock -n 9; then
  echo "another water 1 ns queue already owns ${ROOT}/water_1ns.lock" >&2
  exit 75
fi

[[ -x "${LMP}" ]]
[[ -f "${PREPARE}" ]]
[[ -f "${INITIAL}" ]]

torch_lib="$(${PYTHON_BIN} -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
nvidia_lib_dirs="$(${PYTHON_BIN} - <<'PY'
import pathlib
import site

root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)"
export LD_LIBRARY_PATH="${torch_lib}:${nvidia_lib_dirs}:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

run_lammps_case() {
  local mode="$1"
  local target_steps="$2"
  local out="$3"
  local model="${ROOT}/assets/models/water_${mode}_dynamic.pt2"

  [[ -f "${model}" ]]
  [[ -f "${model}.meta" ]]
  mkdir -p "${out}"
  if [[ -f "${out}/DONE" ]]; then
    echo "already complete: ${out}/DONE"
    return 0
  fi

  "${PYTHON_BIN}" "${PREPARE}" \
    --model "${model}" --initial-structure "${INITIAL}" \
    --ase-state "${out}/restart.json" --output-dir "${out}" \
    --elements H O --temperature-k 300 --seed "${SEED}" \
    --steps "${target_steps}" --repeat 2,2,2 --pair-style mff/torch

  if [[ -f "${out}/DONE" ]]; then
    return 0
  fi

  MFF_DEBUG_BUNDLE=1 "${LMP}" \
    -log "${out}/thermo_lammps.log" \
    -in "${out}/in.mfftorch_production" \
    2>&1 | tee -a "${out}/lammps_stdout.log"
  grep -q "MFF_PRODUCTION_DONE" "${out}/lammps_stdout.log"
  touch "${out}/DONE_LAMMPS" "${out}/DONE"
}

# Validate that the AOTI packages transferred from A100 execute correctly on
# the RTX 4090 before starting any production trajectory.
for mode in phaseoff final persistent; do
  smoke="${ROOT}/smoke/${mode}"
  run_lammps_case "${mode}" 10 "${smoke}"
done
touch "${ROOT}/SMOKE_ALL_DONE"

# Prioritise the main CHORUS result, then its final-layer variant and control.
for mode in persistent final phaseoff; do
  printf '%s\n' "${mode}" > "${ROOT}/CURRENT_MODE"
  run_lammps_case \
    "${mode}" "${STEPS}" \
    "${ROOT}/md/water_1ns/mace_${mode}/seed${SEED}"
  echo "WATER_1NS_4090_MODE_DONE mode=${mode} seed=${SEED} $(date -Is)"
done

rm -f "${ROOT}/CURRENT_MODE"
touch "${ROOT}/DONE"
echo "WATER_1NS_4090_ALL_DONE seed=${SEED} steps=${STEPS} $(date -Is)"
