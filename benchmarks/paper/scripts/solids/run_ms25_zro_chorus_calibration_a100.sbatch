#!/usr/bin/env bash
#SBATCH --job-name=ms25-zro-cal
#SBATCH --partition=GPU-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm/zro_cal_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/slurm/zro_cal_%A_%a.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-ms25-20260818}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/chorus_solid_data/ms25_20260818/prepared}"
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/chorus_solid_runs/ms25_20260818/mace_ictc_family}"
BATCH_SIZE="${BATCH_SIZE:-4}"

modes=(phaseoff final persistent)
mode="${modes[${SLURM_ARRAY_TASK_ID:-0}]}"
data_dir="${DATA_ROOT}/zr-o/set1/chorus"
metadata="${DATA_ROOT}/zr-o/set1/metadata.json"
run="${RUN_ROOT}/zr-o/${mode}"
raw_eval="${run}/validation_force_selected_eval"
out="${run}/train_only_energy_calibration"

mkdir -p "${RUN_ROOT}/slurm" "${out}"
export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

if [[ -f "${out}/DONE" ]]; then
  echo "MS25_ZRO_CHORUS_CALIBRATION_SKIP mode=${mode}"
  exit 0
fi
[[ -f "${raw_eval}/DONE" ]]

readarray -t metadata_values < <("${PYTHON_BIN}" - "${metadata}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(x["cutoff_angstrom"])
print(x["average_train_neighbours"])
print(",".join(map(str, x["atomic_numbers"])))
print(",".join(f"{v:.17g}" for v in x["element_reference_energies_ev"]))
if x["composition_matrix_rank"] != len(x["atomic_numbers"]):
    raise RuntimeError("Zr-O train composition matrix is rank deficient")
PY
)
cutoff="${metadata_values[0]}"
avg_neighbors="${metadata_values[1]}"
e0_keys="${metadata_values[2]}"
e0_values="${metadata_values[3]}"
selected="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "${raw_eval}/metrics.json")"

case "${mode}" in
  phaseoff)
    phase_args=(--phase-mode none)
    ;;
  final|persistent)
    phase_args=(
      --phase-mode final-full-l-residual
      --phase-amplitude softplus --phase-coefficient polar
      --phase-context content --phase-density-pairs full-nonlinear
      --phase-normalization avg-neighbors
      --phase-placement pre-product-full-l --phase-scope "${mode}"
      --phase-hidden-channels 32 --phase-scale-init 0.05
      --phase-density-rank 16
    )
    ;;
esac

augmented="${out}/${mode}.zero_correction_source.pth"
calibrated="${out}/${mode}.train_only_calibrated.pth"
audit="${out}/checkpoint_audit.json"
if [[ ! -f "${augmented}" ]]; then
  "${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/augment_chorus_energy_correction.py" \
    --source "${selected}" --output "${augmented}" --audit "${audit}"
fi

common=(
  --data-dir "${data_dir}" --train-prefix train
  --channels 128 --lmax 2 --max-ell 2
  --num-interaction 2 --correlation 3
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg
  --first-layer-self-connection --mace-compatible-random-init
  --readout-hidden-channels 64
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6
  --max-radius "${cutoff}" --avg-num-neighbors "${avg_neighbors}"
  --atomic-energy-keys "${e0_keys}" --atomic-energy-values="${e0_values}"
  --scaling std_scaling --batch-size "${BATCH_SIZE}"
  --dtype float32 --device cuda --num-workers 0
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0
  --optimizer adamw --optimizer-param-groups mace --weight-decay 5e-7
  --amsgrad --max-grad-norm 10 --seed 20260616
  --element-energy-correction
)

"${PYTHON_BIN}" -m chorus.cli.train \
  "${common[@]}" "${phase_args[@]}" \
  --val-prefix val --epochs 0 --max-steps 0 \
  --final-fit-element-energy-correction \
  --resume-checkpoint "${augmented}" --checkpoint "${calibrated}" \
  >"${out}/calibrate_and_val.log" 2>&1

for split in val test; do
  "${PYTHON_BIN}" -m chorus.cli.train \
    "${common[@]}" "${phase_args[@]}" \
    --val-prefix "${split}" --epochs 0 --max-steps 0 --eval-only \
    --resume-checkpoint "${calibrated}" \
    --checkpoint "${out}/${mode}.${split}.unused.pth" \
    >"${out}/eval_calibrated_${split}.log" 2>&1
done

"${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/summarize_zro_chorus_energy_calibration.py" \
  --raw-metrics "${raw_eval}/metrics.json" \
  --calibration-log "${out}/calibrate_and_val.log" \
  --calibrated-val-log "${out}/eval_calibrated_val.log" \
  --calibrated-test-log "${out}/eval_calibrated_test.log" \
  --audit "${audit}" --calibrated-checkpoint "${calibrated}" \
  --output "${out}/metrics.json"
touch "${out}/DONE"
echo "MS25_ZRO_CHORUS_CALIBRATION_DONE mode=${mode} $(date -Is)"
