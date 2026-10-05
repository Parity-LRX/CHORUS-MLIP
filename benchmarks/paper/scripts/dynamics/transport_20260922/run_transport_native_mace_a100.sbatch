#!/usr/bin/env bash
#SBATCH --job-name=transport-mace-sh
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/mace_sh_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/mace_sh_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/prepared}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/native_mace}"
SEED="${SEED:-20260616}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-180}"
RUN_EVAL="${RUN_EVAL:-1}"

datasets=(water lips)
dataset="${datasets[${SLURM_ARRAY_TASK_ID:-0}]}"
base="${DATA_ROOT}/${dataset}"
extxyz="${base}/extxyz"
metadata="${base}/metadata.json"
out="${RUN_ROOT}/${dataset}"
name="native_mace_transport_${dataset}"

[[ -f "${ROOT}/DATA_READY" ]]
mkdir -p "${out}/logs" "${out}/models" "${out}/checkpoints" \
  "${out}/results" "${out}/validation_force_selected_eval"
export PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

readarray -t values < <("${PYTHON_BIN}" - "${metadata}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
zs = x["atomic_numbers"]
e0 = x["element_reference_energies_ev"]
print(x["cutoff_angstrom"])
print(x["average_train_neighbours"])
print("[" + ", ".join(map(str, zs)) + "]")
print("{" + ", ".join(f"{z}: {v:.17g}" for z, v in zip(zs, e0)) + "}")
print(",".join(map(str, zs)))
PY
)
cutoff="${values[0]}"
avg_neighbors="${values[1]}"
atomic_numbers_arg="${values[2]}"
e0_arg="${values[3]}"
atomic_numbers_csv="${values[4]}"
gamma="$(${PYTHON_BIN} - "${EPOCHS}" <<'PY'
import sys
print((1e-6 / 1e-3) ** (1.0 / int(sys.argv[1])))
PY
)"

if [[ ! -f "${out}/DONE" ]]; then
  set +e
  "${PYTHON_BIN}" -m mace.cli.run_train \
    --name "${name}" --seed "${SEED}" \
    --device cuda --default_dtype float32 \
    --log_dir "${out}/logs" --model_dir "${out}/models" \
    --checkpoints_dir "${out}/checkpoints" --results_dir "${out}/results" \
    --model ScaleShiftMACE --r_max "${cutoff}" \
    --radial_type bessel --num_radial_basis 8 --num_cutoff_basis 6 \
    --max_ell 2 --num_interactions 2 --correlation 3 --use_reduced_cg True \
    --num_channels 128 --max_L 2 \
    --hidden_irreps "128x0e + 128x1o + 128x2e" \
    --MLP_irreps "64x0e" --radial_MLP "[64, 64, 64]" \
    --interaction RealAgnosticResidualInteractionBlock \
    --interaction_first RealAgnosticResidualInteractionBlock \
    --enable_cueq True --only_cueq True \
    --train_file "${extxyz}/train.extxyz" --valid_file "${extxyz}/val.extxyz" \
    --energy_key energy --forces_key forces \
    --atomic_numbers "${atomic_numbers_arg}" --E0s "${e0_arg}" \
    --avg_num_neighbors "${avg_neighbors}" --scaling std_scaling \
    --loss weighted --energy_weight 1 --forces_weight 100 \
    --batch_size "${BATCH_SIZE}" --valid_batch_size "${BATCH_SIZE}" \
    --max_num_epochs "${EPOCHS}" \
    --lr 0.001 --weight_decay 5e-7 --optimizer adamw \
    --scheduler ExponentialLR --lr_scheduler_gamma "${gamma}" --amsgrad \
    --num_workers 0 --compute_forces True --compute_stress False \
    --eval_interval 1 --keep_checkpoints --save_all_checkpoints \
    >"${out}/train.log" 2>&1
  code=$?
  set -e
  checkpoint_count="$(find "${out}/checkpoints" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
  if (( code == 0 )); then
    touch "${out}/DONE"
  elif grep -q 'Training complete' "${out}/train.log" \
    && grep -q 'ScriptFunction cannot be pickled' "${out}/train.log" \
    && (( checkpoint_count >= EPOCHS )); then
    touch "${out}/RECOVERED_CUEQ_EXPORT_FAILURE" "${out}/DONE"
  else
    echo "native MACE failed before audited completion, exit=${code}" >&2
    exit "${code}"
  fi
fi

eval_dir="${out}/validation_force_selected_eval"
if [[ "${RUN_EVAL}" == "1" && ! -f "${eval_dir}/DONE" ]]; then
  env PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/training/evaluate_native_mace_selected.py" \
      --run-dir "${out}" --test-file "${extxyz}/test.extxyz" \
      --name "${name}" --atomic-numbers "${atomic_numbers_csv}" \
      --avg-num-neighbors "${avg_neighbors}" --r-max "${cutoff}" \
      --seed "${SEED}" --device cuda --out "${eval_dir}" \
      >"${eval_dir}/eval.log" 2>&1
fi

echo "TRANSPORT_NATIVE_MACE_DONE dataset=${dataset} $(date -Is)"
