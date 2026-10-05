#!/usr/bin/env bash
#SBATCH --job-name=lips-precrash-parity
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/lips_precrash_parity_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/lips_precrash_parity_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
BUILD_ROOT="${ROOT}/deploy/mfftorch_lammps_20260924"
LMP="${BUILD_ROOT}/build-mfftorch/lmp"
EXTRACT="${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/extract_lammps_tail.py"
SMOKE="${REPO}/benchmarks/paper/scripts/dynamics/transport_20260922/prepare_mfftorch_lammps_smoke.py"

modes=(final persistent persistent persistent persistent)
seeds=(20260617 20260616 20260618 20260619 20260620)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
mode="${modes[${task_id}]}"
seed="${seeds[${task_id}]}"
source_dir="${ROOT}/md/carnet_protocol_lammps/mace_family/lips/${mode}/seed${seed}"
work="${ROOT}/diagnostics/lips_instability_20260926/parity/${mode}_seed${seed}"
metrics="${ROOT}/runs/mace_family/lips/${mode}/validation_force_selected_eval/metrics.json"
model="${BUILD_ROOT}/models/lips_${mode}_dynamic.pt2"

mkdir -p "${work}"
rm -f "${work}/DONE" "${work}/audit.json"
checkpoint="$(${PYTHON_BIN} - "${metrics}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_checkpoint"])
PY
)"

"${PYTHON_BIN}" "${EXTRACT}" \
  --trajectory "${source_dir}/trajectory_lammps.lammpstrj" \
  --output-dir "${work}" --tail-frames 50 \
  > "${work}/extract.log" 2>&1

torch_lib="$(${PYTHON_BIN} -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
nvidia_lib_dirs="$(${PYTHON_BIN} - <<'PY'
import pathlib, site
root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)"
export LD_LIBRARY_PATH="${torch_lib}:${nvidia_lib_dirs}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

"${PYTHON_BIN}" "${SMOKE}" prepare \
  --checkpoint "${checkpoint}" --model "${model}" \
  --initial-structure "${work}/last_safe_frame.extxyz" \
  --output-dir "${work}" --elements Li P S --repeat 1,1,1 \
  --avg-num-neighbors 25.186819277108434 --temperature-k 520 \
  --md-steps 1 --device cuda --pair-style mff/torch \
  > "${work}/prepare.log" 2>&1

MFF_DEBUG_BUNDLE=1 "${LMP}" -in "${work}/in.mfftorch_smoke" \
  > "${work}/lammps.log" 2>&1

"${PYTHON_BIN}" "${SMOKE}" compare \
  --output-dir "${work}" --log "${work}/lammps.log" --md-steps 1 \
  > "${work}/compare.log" 2>&1

touch "${work}/DONE"
echo "LIPS_PRECRASH_PARITY_DONE mode=${mode} seed=${seed} $(date -Is)"
