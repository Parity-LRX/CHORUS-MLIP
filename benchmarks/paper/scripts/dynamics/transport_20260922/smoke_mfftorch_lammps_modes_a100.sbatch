#!/usr/bin/env bash
#SBATCH --job-name=smoke-mfftorch-modes
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/smoke_mfftorch_modes_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/smoke_mfftorch_modes_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
BUILD_ROOT="${BUILD_ROOT:-${ROOT}/deploy/mfftorch_lammps_20260924}"
MD_STEPS="${MD_STEPS:-1000}"
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
SCRIPT="${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/prepare_mfftorch_lammps_smoke.py"

datasets=(water water water lips lips lips)
modes=(phaseoff final persistent phaseoff final persistent)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[${task_id}]}"
mode="${modes[${task_id}]}"

case "${dataset}" in
  water)
    elements=(H O)
    avg_num_neighbors=47.087885416666666
    natoms=1536
    vary_atoms=192
    degree=48
    vary_degree=56
    temperature=300
    ;;
  lips)
    elements=(Li P S)
    avg_num_neighbors=25.186819277108434
    natoms=664
    vary_atoms=83
    degree=32
    vary_degree=40
    temperature=520
    ;;
  *)
    echo "unknown dataset ${dataset}" >&2
    exit 2
    ;;
esac
element_csv="$(IFS=,; echo "${elements[*]}")"

METRICS="${ROOT}/runs/mace_family/${dataset}/${mode}/validation_force_selected_eval/metrics.json"
INITIAL="${ROOT}/data/prepared/${dataset}/extxyz/test.extxyz"
WORK="${BUILD_ROOT}/smoke_${dataset}_${mode}_${backend_tag}"
MODEL_DIR="${BUILD_ROOT}/models"
MODEL="${MODEL_DIR}/${dataset}_${mode}_dynamic.pt2"

mkdir -p "${WORK}" "${MODEL_DIR}"
rm -f "${WORK}/DONE" "${WORK}/audit.json"
[[ -x "${LMP}" ]]
[[ -f "${BUILD_MARKER}" ]]
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
    --elements "${element_csv}" \
    --avg-num-neighbors "${avg_num_neighbors}" \
    --atoms "${natoms}" --degree "${degree}" \
    --vary-atoms "${vary_atoms}" --vary-degree "${vary_degree}" \
    --dtype float32 --device cuda --dynamic --embed-e0 \
    --assume-cutoff-edges \
    --iters 5 --warmup 2 \
    --out "${MODEL}" \
    2>&1 | tee "${WORK}/export.log"
fi

"${PYTHON_BIN}" "${SCRIPT}" prepare \
  --checkpoint "${CHECKPOINT}" --model "${MODEL}" \
  --initial-structure "${INITIAL}" --output-dir "${WORK}" \
  --elements "${elements[@]}" --repeat 2,2,2 \
  --avg-num-neighbors "${avg_num_neighbors}" \
  --temperature-k "${temperature}" \
  --md-steps "${MD_STEPS}" --device cuda --pair-style "${PAIR_STYLE}" \
  2>&1 | tee "${WORK}/prepare.log"

if [[ "${PAIR_STYLE}" == "mff/torch/kk" ]]; then
  MFF_DEBUG_BUNDLE=1 "${LMP}" \
    -k on g 1 -sf kk -pk kokkos newton off neigh full \
    -in "${WORK}/in.mfftorch_smoke" \
    2>&1 | tee "${WORK}/lammps.log"
else
  MFF_DEBUG_BUNDLE=1 "${LMP}" \
    -in "${WORK}/in.mfftorch_smoke" \
    2>&1 | tee "${WORK}/lammps.log"
fi

"${PYTHON_BIN}" "${SCRIPT}" compare \
  --output-dir "${WORK}" --log "${WORK}/lammps.log" --md-steps "${MD_STEPS}" \
  2>&1 | tee "${WORK}/compare.log"

echo "MFFTORCH_MODE_SMOKE_DONE dataset=${dataset} mode=${mode} ${WORK} $(date -Is)"
