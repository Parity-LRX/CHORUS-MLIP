#!/usr/bin/env bash
set -euo pipefail

# Apple-to-apple training matrix for public rMD17 subsets.
#
# Expected data layout:
#   ${DATA_ROOT}/revised_ethanol/{train,val,test}.extxyz
#   ${DATA_ROOT}/revised_ethanol/processed_{train,val,test}.h5
#
# Use prepare_md17_public.py to create this layout.

PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/lrx/MACE-ICTC}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/tmp/mace_ictc_public_md17}"
OUT_ROOT="${OUT_ROOT:-/tmp/mace_ictc_public_md17_train_$(date +%Y%m%d_%H%M%S)}"

DATASETS="${DATASETS:-revised_ethanol,revised_benzene,revised_aspirin}"
MODES="${MODES:-ictd_bridge_u_eager,ictd_bridge_u_makefx,ictd_cueq_makefx,mace_e3nn,mace_cueq}"

SEED="${SEED:-20260616}"
MAX_STEPS="${MAX_STEPS:-2000}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHANNELS="${CHANNELS:-64}"
MLP_IRREPS="${MLP_IRREPS:-16x0e}"
HIDDEN_LMAX="${HIDDEN_LMAX:-1}"
MAX_ELL="${MAX_ELL:-2}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
CORRELATION="${CORRELATION:-2}"
R_MAX="${R_MAX:-4.5}"
LR="${LR:-0.001}"
MIN_LR="${MIN_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-7}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100.0}"
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
ICTC_AMSGRAD="${ICTC_AMSGRAD:-1}"
ICTC_USE_REDUCED_CG="${ICTC_USE_REDUCED_CG:-0}"
ICTC_CONV_TP_SCALE_INIT="${ICTC_CONV_TP_SCALE_INIT:-none}"
ICTC_FREEZE_CONV_TP_WEIGHT="${ICTC_FREEZE_CONV_TP_WEIGHT:-0}"
ICTC_INTERACTION_INIT="${ICTC_INTERACTION_INIT:-identity}"

RUN="${RUN:-0}"
SMOKE="${SMOKE:-0}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/models" "${OUT_ROOT}/checkpoints" "${OUT_ROOT}/results"

if [[ "${SMOKE}" == "1" ]]; then
  DATASETS="${SMOKE_DATASETS:-revised_ethanol}"
  MODES="${SMOKE_MODES:-ictd_bridge_u_eager,mace_e3nn}"
  MAX_STEPS="${SMOKE_MAX_STEPS:-50}"
  EPOCHS="${SMOKE_EPOCHS:-5}"
fi

hidden_irreps() {
  local c="$1"
  local lmax="$2"
  if [[ "${lmax}" == "0" ]]; then
    printf "%sx0e" "${c}"
  elif [[ "${lmax}" == "1" ]]; then
    printf "%sx0e + %sx1o" "${c}" "${c}"
  elif [[ "${lmax}" == "2" ]]; then
    printf "%sx0e + %sx1o + %sx2e" "${c}" "${c}" "${c}"
  else
    echo "unsupported HIDDEN_LMAX=${lmax}" >&2
    exit 2
  fi
}

mace_epochs_for_dataset() {
  local data_dir="$1"
  "${PYTHON_BIN}" - "${data_dir}" "${MAX_STEPS}" "${BATCH_SIZE}" <<'PY'
import json
import math
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
max_steps = int(sys.argv[2])
batch_size = int(sys.argv[3])
meta = json.loads((data_dir / "metadata.json").read_text())
n_train = int(meta["splits"]["train"])
steps_per_epoch = max(1, math.ceil(n_train / batch_size))
print(max(1, math.ceil(max_steps / steps_per_epoch)))
PY
}

run_cmd() {
  local name="$1"
  shift
  local log="${OUT_ROOT}/logs/${name}.log"
  printf "\n### %s\n" "${name}"
  printf "%q " "$@"
  printf "\n"
  if [[ "${RUN}" == "1" ]]; then
    set +e
    "$@" > "${log}" 2>&1
    local rc=$?
    set -e
    if [[ "${rc}" != "0" ]]; then
      if grep -q "Training complete" "${log}" && grep -q "ScriptFunction cannot be pickled" "${log}"; then
        echo "WARNING ${name}: training completed but mace-torch model save failed with ScriptFunction pickle error; retaining log for loss comparison." | tee -a "${OUT_ROOT}/logs/status.log"
      else
        echo "ERROR ${name}: command failed with exit code ${rc}; see ${log}" | tee -a "${OUT_ROOT}/logs/status.log"
        return "${rc}"
      fi
    else
      echo "OK ${name}" >> "${OUT_ROOT}/logs/status.log"
    fi
  fi
}

IFS=',' read -r -a DATASET_ARR <<< "${DATASETS}"
IFS=',' read -r -a MODE_ARR <<< "${MODES}"
HIDDEN_IRREPS="$(hidden_irreps "${CHANNELS}" "${HIDDEN_LMAX}")"
ICTC_OPT_FLAGS=()
if [[ "${ICTC_AMSGRAD}" == "1" ]]; then
  ICTC_OPT_FLAGS+=(--amsgrad)
fi
if [[ -n "${MAX_GRAD_NORM}" && "${MAX_GRAD_NORM}" != "0" ]]; then
  ICTC_OPT_FLAGS+=(--max-grad-norm "${MAX_GRAD_NORM}")
fi
ICTC_CUEQ_FLAGS=()
if [[ "${ICTC_USE_REDUCED_CG}" == "1" ]]; then
  ICTC_CUEQ_FLAGS+=(--use-reduced-cg)
fi
ICTC_PARAM_FLAGS=(--conv-tp-scale-init "${ICTC_CONV_TP_SCALE_INIT}" --interaction-init "${ICTC_INTERACTION_INIT}")
if [[ "${ICTC_FREEZE_CONV_TP_WEIGHT}" == "1" ]]; then
  ICTC_PARAM_FLAGS+=(--freeze-conv-tp-weight)
fi

for dataset in "${DATASET_ARR[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  data_dir="${DATA_ROOT}/${dataset}"
  train_xyz="${data_dir}/train.extxyz"
  val_xyz="${data_dir}/val.extxyz"
  if [[ ! -f "${train_xyz}" || ! -f "${data_dir}/processed_train.h5" ]]; then
    echo "Missing prepared dataset at ${data_dir}. Run prepare_md17_public.py first." >&2
    exit 3
  fi
  MACE_EPOCHS="${MACE_MAX_NUM_EPOCHS:-$(mace_epochs_for_dataset "${data_dir}")}"

  for mode in "${MODE_ARR[@]}"; do
    mode="$(echo "${mode}" | xargs)"
    job="${dataset}_${mode}_seed${SEED}_steps${MAX_STEPS}"
    case "${mode}" in
      ictd_bridge_u_eager)
        run_cmd "${job}" env PYTHONPATH="${MACE_ICTC_REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m mace_ictc.cli.train \
          --data-dir "${data_dir}" --train-prefix train --val-prefix val \
          --channels "${CHANNELS}" --lmax "${HIDDEN_LMAX}" --max-ell "${MAX_ELL}" \
          --num-interaction "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" \
          --invariant-channels "${CHANNELS}" \
          --product-backend ictd-bridge-u --angular-basis ictd \
          --function-type bessel --max-radius "${R_MAX}" \
          --atomic-energy-keys "1,6,7,8" --atomic-energy-values "0,0,0,0" \
          --scaling std_scaling --seed "${SEED}" --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}" \
          --batch-size "${BATCH_SIZE}" --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay "${WEIGHT_DECAY}" \
          --optimizer adamw --lr-scheduler exp --lr-scheduler-gamma 0.9993 --loss mse \
          --energy-weight "${ENERGY_WEIGHT}" --force-weight "${FORCE_WEIGHT}" --stress-weight 0 \
          "${ICTC_PARAM_FLAGS[@]}" \
          "${ICTC_OPT_FLAGS[@]}" \
          --device "${DEVICE}" --dtype "${DTYPE}" --num-workers "${NUM_WORKERS}" \
          --checkpoint "${OUT_ROOT}/checkpoints/${job}.pth"
        ;;
      ictd_bridge_u_makefx)
        run_cmd "${job}" env PYTHONPATH="${MACE_ICTC_REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m mace_ictc.cli.train \
          --data-dir "${data_dir}" --train-prefix train --val-prefix val \
          --channels "${CHANNELS}" --lmax "${HIDDEN_LMAX}" --max-ell "${MAX_ELL}" \
          --num-interaction "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" \
          --invariant-channels "${CHANNELS}" \
          --product-backend ictd-bridge-u --angular-basis ictd \
          --function-type bessel --max-radius "${R_MAX}" \
          --atomic-energy-keys "1,6,7,8" --atomic-energy-values "0,0,0,0" \
          --scaling std_scaling --seed "${SEED}" --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}" \
          --batch-size "${BATCH_SIZE}" --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay "${WEIGHT_DECAY}" \
          --optimizer adamw --lr-scheduler exp --lr-scheduler-gamma 0.9993 --loss mse \
          --energy-weight "${ENERGY_WEIGHT}" --force-weight "${FORCE_WEIGHT}" --stress-weight 0 \
          "${ICTC_PARAM_FLAGS[@]}" \
          "${ICTC_OPT_FLAGS[@]}" \
          --train-makefx-compile --makefx-buckets 4 --pad-nodes-to-max --pad-edges-to-max \
          --device "${DEVICE}" --dtype "${DTYPE}" --num-workers "${NUM_WORKERS}" \
          --checkpoint "${OUT_ROOT}/checkpoints/${job}.pth"
        ;;
      ictd_cueq_makefx)
        run_cmd "${job}" env PYTHONPATH="${MACE_ICTC_REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m mace_ictc.cli.train \
          --data-dir "${data_dir}" --train-prefix train --val-prefix val \
          --channels "${CHANNELS}" --lmax "${HIDDEN_LMAX}" --max-ell "${MAX_ELL}" \
          --num-interaction "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" \
          --invariant-channels "${CHANNELS}" \
          --product-backend cueq --angular-basis e3nn "${ICTC_CUEQ_FLAGS[@]}" \
          --function-type bessel --max-radius "${R_MAX}" \
          --atomic-energy-keys "1,6,7,8" --atomic-energy-values "0,0,0,0" \
          --scaling std_scaling --seed "${SEED}" --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}" \
          --batch-size "${BATCH_SIZE}" --lr "${LR}" --min-lr "${MIN_LR}" --weight-decay "${WEIGHT_DECAY}" \
          --optimizer adamw --lr-scheduler exp --lr-scheduler-gamma 0.9993 --loss mse \
          --energy-weight "${ENERGY_WEIGHT}" --force-weight "${FORCE_WEIGHT}" --stress-weight 0 \
          "${ICTC_PARAM_FLAGS[@]}" \
          "${ICTC_OPT_FLAGS[@]}" \
          --train-makefx-compile --makefx-buckets 4 --pad-nodes-to-max --pad-edges-to-max \
          --device "${DEVICE}" --dtype "${DTYPE}" --num-workers "${NUM_WORKERS}" \
          --checkpoint "${OUT_ROOT}/checkpoints/${job}.pth"
        ;;
      mace_e3nn)
        run_cmd "${job}" env PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m mace.cli.run_train \
          --name "${job}" --seed "${SEED}" --device "${DEVICE}" --default_dtype "${DTYPE}" \
          --log_dir "${OUT_ROOT}/logs" --model_dir "${OUT_ROOT}/models" \
          --checkpoints_dir "${OUT_ROOT}/checkpoints" --results_dir "${OUT_ROOT}/results" \
          --model ScaleShiftMACE --r_max "${R_MAX}" --radial_type bessel \
          --num_radial_basis 8 --num_cutoff_basis 6 --max_ell "${MAX_ELL}" \
          --num_interactions "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" \
          --num_channels "${CHANNELS}" --max_L "${HIDDEN_LMAX}" --hidden_irreps "${HIDDEN_IRREPS}" \
          --MLP_irreps "${MLP_IRREPS}" --radial_MLP "[${CHANNELS}, ${CHANNELS}, ${CHANNELS}]" \
          --interaction RealAgnosticResidualInteractionBlock --interaction_first RealAgnosticResidualInteractionBlock \
          --train_file "${train_xyz}" --valid_file "${val_xyz}" --energy_key energy --forces_key forces \
          --atomic_numbers "[1, 6, 7, 8]" --E0s "{1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}" \
          --scaling std_scaling --loss weighted --energy_weight "${ENERGY_WEIGHT}" --forces_weight "${FORCE_WEIGHT}" \
          --batch_size "${BATCH_SIZE}" --valid_batch_size "${BATCH_SIZE}" --max_num_epochs "${MACE_EPOCHS}" \
          --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" --optimizer adamw --scheduler ExponentialLR --lr_scheduler_gamma 0.9993 \
          --num_workers "${NUM_WORKERS}" --compute_forces True --compute_stress False --eval_interval 1
        ;;
      mace_cueq)
        run_cmd "${job}" env PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m mace.cli.run_train \
          --name "${job}" --seed "${SEED}" --device "${DEVICE}" --default_dtype "${DTYPE}" \
          --log_dir "${OUT_ROOT}/logs" --model_dir "${OUT_ROOT}/models" \
          --checkpoints_dir "${OUT_ROOT}/checkpoints" --results_dir "${OUT_ROOT}/results" \
          --model ScaleShiftMACE --r_max "${R_MAX}" --radial_type bessel \
          --num_radial_basis 8 --num_cutoff_basis 6 --max_ell "${MAX_ELL}" \
          --num_interactions "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" \
          --num_channels "${CHANNELS}" --max_L "${HIDDEN_LMAX}" --hidden_irreps "${HIDDEN_IRREPS}" \
          --MLP_irreps "${MLP_IRREPS}" --radial_MLP "[${CHANNELS}, ${CHANNELS}, ${CHANNELS}]" \
          --interaction RealAgnosticResidualInteractionBlock --interaction_first RealAgnosticResidualInteractionBlock \
          --train_file "${train_xyz}" --valid_file "${val_xyz}" --energy_key energy --forces_key forces \
          --atomic_numbers "[1, 6, 7, 8]" --E0s "{1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}" \
          --scaling std_scaling --loss weighted --energy_weight "${ENERGY_WEIGHT}" --forces_weight "${FORCE_WEIGHT}" \
          --batch_size "${BATCH_SIZE}" --valid_batch_size "${BATCH_SIZE}" --max_num_epochs "${MACE_EPOCHS}" \
          --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" --optimizer adamw --scheduler ExponentialLR --lr_scheduler_gamma 0.9993 \
          --num_workers "${NUM_WORKERS}" --compute_forces True --compute_stress False --eval_interval 1 \
          --enable_cueq True --only_cueq True
        ;;
      *)
        echo "Unknown mode: ${mode}" >&2
        exit 4
        ;;
    esac
  done
done

cat > "${OUT_ROOT}/matrix_metadata.json" <<EOF
{
  "data_root": "${DATA_ROOT}",
  "datasets": "${DATASETS}",
  "modes": "${MODES}",
  "seed": ${SEED},
  "max_steps": ${MAX_STEPS},
  "epochs": ${EPOCHS},
  "mace_epoch_rule": "ceil(max_steps / ceil(n_train / batch_size)); override with MACE_MAX_NUM_EPOCHS",
  "mace_max_num_epochs_override": "${MACE_MAX_NUM_EPOCHS:-}",
  "batch_size": ${BATCH_SIZE},
  "channels": ${CHANNELS},
  "mlp_irreps": "${MLP_IRREPS}",
  "hidden_lmax": ${HIDDEN_LMAX},
  "max_ell": ${MAX_ELL},
  "num_interactions": ${NUM_INTERACTIONS},
  "correlation": ${CORRELATION},
  "r_max": ${R_MAX},
  "lr": ${LR},
  "max_grad_norm": "${MAX_GRAD_NORM}",
  "ictd_amsgrad": "${ICTC_AMSGRAD}",
  "ictd_use_reduced_cg": "${ICTC_USE_REDUCED_CG}",
  "ictd_conv_tp_scale_init": "${ICTC_CONV_TP_SCALE_INIT}",
  "ictd_freeze_conv_tp_weight": "${ICTC_FREEZE_CONV_TP_WEIGHT}",
  "ictd_interaction_init": "${ICTC_INTERACTION_INIT}",
  "energy_weight": ${ENERGY_WEIGHT},
  "force_weight": ${FORCE_WEIGHT},
  "dtype": "${DTYPE}"
}
EOF

echo "OUT_ROOT=${OUT_ROOT}"
