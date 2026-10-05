#!/usr/bin/env bash
#SBATCH --job-name=smoke-mfftorch-lmp
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/smoke_mfftorch_lmp_%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/smoke_mfftorch_lmp_%j.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
BUILD_ROOT="${BUILD_ROOT:-${ROOT}/deploy/mfftorch_lammps_20260924}"
WORK="${WORK:-${BUILD_ROOT}/water_phaseoff_dynamic_smoke}"
AVG_NUM_NEIGHBORS="${AVG_NUM_NEIGHBORS:-47.087885416666666}"
MD_STEPS="${MD_STEPS:-1000}"
METRICS="${ROOT}/runs/mace_family/water/phaseoff/validation_force_selected_eval/metrics.json"
INITIAL="${ROOT}/data/prepared/water/extxyz/test.extxyz"
MODEL="${WORK}/water_phaseoff_dynamic.pt2"
LMP="${BUILD_ROOT}/build-mfftorch-kk/lmp"
SCRIPT="${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/prepare_mfftorch_lammps_smoke.py"

mkdir -p "${WORK}"
rm -f "${WORK}/DONE" "${WORK}/audit.json"
[[ -x "${LMP}" ]]
[[ -f "${BUILD_ROOT}/BUILD_KK_DONE" ]]
[[ -f "${METRICS}" ]]
[[ -f "${INITIAL}" ]]

CHECKPOINT="$(${PYTHON_BIN} - "${METRICS}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_checkpoint"])
PY
)"
[[ -f "${CHECKPOINT}" ]]

TORCH_LIB="$(${PYTHON_BIN} -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
NVIDIA_LIB_DIRS="$(${PYTHON_BIN} - <<'PY'
import pathlib
import site

root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

if [[ ! -f "${MODEL}" || ! -f "${MODEL}.meta" ]]; then
  "${PYTHON_BIN}" -m chorus.cli.export_aoti_core \
    --checkpoint "${CHECKPOINT}" \
    --elements H,O \
    --avg-num-neighbors "${AVG_NUM_NEIGHBORS}" \
    --atoms 1536 --degree 48 --vary-atoms 192 --vary-degree 56 \
    --dtype float32 --device cuda --dynamic --embed-e0 \
    --assume-cutoff-edges \
    --iters 5 --warmup 2 \
    --out "${MODEL}" \
    2>&1 | tee "${WORK}/export.log"
fi

"${PYTHON_BIN}" "${SCRIPT}" prepare \
  --checkpoint "${CHECKPOINT}" --model "${MODEL}" \
  --initial-structure "${INITIAL}" --output-dir "${WORK}" \
  --elements H O --repeat 2,2,2 \
  --avg-num-neighbors "${AVG_NUM_NEIGHBORS}" \
  --md-steps "${MD_STEPS}" --device cuda --pair-style mff/torch/kk \
  2>&1 | tee "${WORK}/prepare.log"

MFF_DEBUG_BUNDLE=1 "${LMP}" \
  -k on g 1 -sf kk -pk kokkos newton off neigh full \
  -in "${WORK}/in.mfftorch_smoke" \
  2>&1 | tee "${WORK}/lammps.log"

"${PYTHON_BIN}" "${SCRIPT}" compare \
  --output-dir "${WORK}" --log "${WORK}/lammps.log" --md-steps "${MD_STEPS}" \
  2>&1 | tee "${WORK}/compare.log"

echo "MFFTORCH_LAMMPS_SMOKE_DONE ${WORK} $(date -Is)"
