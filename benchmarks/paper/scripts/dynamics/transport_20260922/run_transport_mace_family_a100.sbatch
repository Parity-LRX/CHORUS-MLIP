#!/usr/bin/env bash
#SBATCH --job-name=transport-mace
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/mace_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/mace_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/prepared}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/mace_family}"
SEED="${SEED:-20260616}"
MAX_STEPS="${MAX_STEPS:-45000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RUN_EVAL="${RUN_EVAL:-1}"

datasets=(water lips)
modes=(phaseoff final persistent)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[$((task_id / 3))]}"
mode="${modes[$((task_id % 3))]}"
data_dir="${DATA_ROOT}/${dataset}/chorus"
metadata="${DATA_ROOT}/${dataset}/metadata.json"
out="${RUN_ROOT}/${dataset}/${mode}"
tag="transport_${dataset}_${mode}_c128_l2_corr3_r16"

[[ -f "${ROOT}/DATA_READY" ]]
mkdir -p "${out}/checkpoints" "${out}/logs" "${out}/validation_force_selected_eval"
export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

readarray -t metadata_values < <("${PYTHON_BIN}" - "${metadata}" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(x["cutoff_angstrom"])
print(x["average_train_neighbours"])
print(",".join(map(str, x["atomic_numbers"])))
print(",".join(f"{v:.17g}" for v in x["element_reference_energies_ev"]))
PY
)
cutoff="${metadata_values[0]}"
avg_neighbors="${metadata_values[1]}"
e0_keys="${metadata_values[2]}"
e0_values="${metadata_values[3]}"

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
  *)
    echo "unknown mode ${mode}" >&2
    exit 2
    ;;
esac

common_args=(
  --data-dir "${data_dir}" --train-prefix train
  --channels 128 --lmax 2 --max-ell 2
  --num-interaction 2 --correlation 3
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg
  --conv-tp-backend pytorch
  --first-layer-self-connection --mace-compatible-random-init
  --readout-hidden-channels 64
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6
  --max-radius "${cutoff}" --avg-num-neighbors "${avg_neighbors}"
  --atomic-energy-keys "${e0_keys}" --atomic-energy-values="${e0_values}"
  --scaling std_scaling --batch-size "${BATCH_SIZE}"
  --dtype float32 --device cuda --num-workers 0
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0
  --seed "${SEED}"
)

checkpoint="${out}/checkpoints/${tag}_final.pth"
if [[ ! -f "${out}/DONE" ]]; then
  "${PYTHON_BIN}" -m chorus.cli.train \
    "${common_args[@]}" --val-prefix val \
    --epochs 256 --max-steps "${MAX_STEPS}" \
    --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
    --optimizer adamw --optimizer-param-groups mace \
    --weight-decay 5e-7 --amsgrad --max-grad-norm 10 \
    --train-makefx-compile --require-train-makefx-compile \
    --makefx-buckets 4 --makefx-max-slots 8 \
    --pad-nodes-to-max --pad-edges-to-max \
    --log-interval 20 --keep-checkpoints 256 \
    "${phase_args[@]}" --checkpoint "${checkpoint}" \
    >"${out}/logs/train.log" 2>&1
  touch "${out}/DONE"
fi

eval_dir="${out}/validation_force_selected_eval"
if [[ "${RUN_EVAL}" == "1" && ! -f "${eval_dir}/DONE" ]]; then
  selected="$("${PYTHON_BIN}" \
    "${REPO}/benchmarks/paper/scripts/solids/select_chorus_checkpoint.py" \
    --run-dir "${out}" --output "${eval_dir}/selection.json")"
  "${PYTHON_BIN}" -m chorus.cli.train \
    "${common_args[@]}" --val-prefix test --epochs 1 \
    "${phase_args[@]}" --resume-checkpoint "${selected}" --eval-only \
    --checkpoint "${eval_dir}/unused.pth" \
    >"${eval_dir}/test.log" 2>&1
  "${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/solids/parse_chorus_test.py" \
    --selection "${eval_dir}/selection.json" \
    --train-log "${out}/logs/train.log" --test-log "${eval_dir}/test.log" \
    --output "${eval_dir}/metrics.json"
  touch "${eval_dir}/DONE"
fi

echo "TRANSPORT_MACE_DONE dataset=${dataset} mode=${mode} $(date -Is)"
