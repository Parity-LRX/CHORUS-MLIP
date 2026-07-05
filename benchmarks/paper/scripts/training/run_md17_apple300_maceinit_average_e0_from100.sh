#!/usr/bin/env bash
set -euo pipefail

# Continue the average-E0 apple-to-apple rMD17 ethanol benchmark from the
# 100-epoch checkpoints to epoch 300. ICTC resumes model + optimizer state from
# the MACE-ICTC checkpoint. MACE uses mace-torch --restart_latest after copying
# the epoch-100-run checkpoint into the new run tag.

PY="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
REPO="${MACE_ICTC_REPO:-/home/ylzhang/lrx/MACE-ICTC}"
MACE="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA="${DATA_DIR:-/tmp/mace_ictc_public_md17/revised_ethanol}"
OLD_OUT="${OLD_OUT:-/tmp/mace_ictc_train_apple100_maceinit_average_e0_20260617_002357}"
OUT="${OUT_ROOT:-/tmp/mace_ictc_train_apple300_maceinit_average_e0_from100_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT}/logs" "${OUT}/checkpoints" "${OUT}/models" "${OUT}/results" "${OUT}/commands"

ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100.0}"
LOSS="${LOSS:-mse}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEEDS_CSV="${SEEDS:-20260616,20260617,20260618}"
MODES_CSV="${MODES:-ictd_bridge_u_eager,ictd_bridge_u_makefx,ictd_cueq_makefx,mace_e3nn,mace_cueq}"

CHANNELS="${CHANNELS:-64}"
HIDDEN_LMAX="${HIDDEN_LMAX:-1}"
MAX_ELL="${MAX_ELL:-2}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
CORRELATION="${CORRELATION:-2}"
R_MAX="${R_MAX:-4.5}"
LR="${LR:-0.001}"
LR_GAMMA="${LR_GAMMA:-0.9993}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-7}"
DTYPE="${DTYPE:-float32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AVG_NEIGHBORS="${AVG_NEIGHBORS:-8.0}"
READOUT_HIDDEN="${READOUT_HIDDEN:-64}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
source /tmp/mace_ictc_average_e0.env

IFS=',' read -r -a SEEDS_ARR <<< "${SEEDS_CSV}"
IFS=',' read -r -a MODES_ARR <<< "${MODES_CSV}"

cat > "${OUT}/metadata.json" <<EOF
{
  "benchmark": "apple_to_apple_training_300_epoch_mace_compatible_init_average_e0_from100",
  "old_out": "${OLD_OUT}",
  "repo": "${REPO}",
  "mace_torch_path": "${MACE}",
  "data": "${DATA}",
  "target_epochs": ${EPOCHS},
  "seeds_csv": "${SEEDS_CSV}",
  "modes_csv": "${MODES_CSV}",
  "fixed_loss": {"loss": "${LOSS}", "energy_weight": ${ENERGY_WEIGHT}, "force_weight": ${FORCE_WEIGHT}, "stress_weight": 0.0},
  "scheduler": {"type": "ExponentialLR", "gamma": ${LR_GAMMA}},
  "optimizer": {"type": "AdamW", "lr": ${LR}, "weight_decay": ${WEIGHT_DECAY}, "amsgrad": true},
  "architecture": {"channels": ${CHANNELS}, "hidden_lmax": ${HIDDEN_LMAX}, "max_ell": ${MAX_ELL}, "num_interactions": ${NUM_INTERACTIONS}, "correlation": ${CORRELATION}, "readout_hidden_channels": ${READOUT_HIDDEN}, "first_layer_self_connection": true, "use_reduced_cg": true},
  "radial": {"type": "bessel", "num_basis": 8, "polynomial_cutoff_p": 6, "r_max": ${R_MAX}},
  "scaling": {"mode": "std_scaling", "avg_num_neighbors": ${AVG_NEIGHBORS}, "e0_keys": "${E0_KEYS}", "e0_vals": "${E0_VALS}", "mace_e0s": "${MACE_E0S}"},
  "note": "Continuation from 100-epoch checkpoints. ICTC checkpoints contain optimizer/global_step and are resumed with --resume-training-state. MACE checkpoints are resumed with --restart_latest after copying the previous run checkpoint to the new tag."
}
EOF

echo "OUT=${OUT}" | tee "${OUT}/status.log"
echo "OLD_OUT=${OLD_OUT}" | tee -a "${OUT}/status.log"
date | tee -a "${OUT}/status.log"

write_command_file() {
  local path="$1"
  shift
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf '%q ' "$@"
    printf '\n'
  } > "${path}"
  chmod +x "${path}"
}

run_logged() {
  local name="$1"
  shift
  local log="${OUT}/logs/${name}.log"
  local cmdfile="${OUT}/commands/${name}.sh"
  write_command_file "${cmdfile}" "$@"
  echo "START ${name} $(date)" | tee -a "${OUT}/status.log"
  set +e
  /usr/bin/time -f 'WALL_SECONDS %e' "$@" > "${log}" 2>&1
  local rc=$?
  set -e
  if [[ "${rc}" != "0" ]]; then
    if grep -q "Training complete" "${log}" && grep -q "ScriptFunction cannot be pickled" "${log}"; then
      echo "OK_WITH_SAVE_WARNING ${name} $(date)" | tee -a "${OUT}/status.log"
    else
      echo "FAIL ${name} rc=${rc} $(date) log=${log}" | tee -a "${OUT}/status.log"
      return "${rc}"
    fi
  else
    echo "OK ${name} $(date)" | tee -a "${OUT}/status.log"
  fi
}

copy_mace_checkpoint() {
  local old_job="$1"
  local new_job="$2"
  local seed="$3"
  "${PY}" - "${OLD_OUT}" "${OUT}" "${old_job}" "${new_job}" "${seed}" <<'PY'
from pathlib import Path
import re
import shutil
import sys

old_out = Path(sys.argv[1])
out = Path(sys.argv[2])
old_job = sys.argv[3]
new_job = sys.argv[4]
seed = sys.argv[5]
pat = re.compile(r"_epoch-(\d+)\.pt$")
files = []
for path in (old_out / "checkpoints").glob(f"{old_job}_run-{seed}_epoch-*.pt"):
    m = pat.search(path.name)
    if m:
        files.append((int(m.group(1)), path))
if not files:
    raise SystemExit(f"no mace checkpoint found for {old_job} seed={seed}")
epoch, src = max(files, key=lambda item: item[0])
dst = out / "checkpoints" / f"{new_job}_run-{seed}_epoch-{epoch}.pt"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print(f"copied {src} -> {dst}")
PY
}

ictd_common_flags() {
  local backend="$1"
  shift
  printf '%s\n' \
    --data-dir "${DATA}" --train-prefix train --val-prefix val \
    --channels "${CHANNELS}" --lmax "${HIDDEN_LMAX}" --max-ell "${MAX_ELL}" \
    --num-interaction "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" \
    --product-backend "${backend}" --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels "${READOUT_HIDDEN}" \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius "${R_MAX}" --avg-num-neighbors "${AVG_NEIGHBORS}" \
    --atomic-energy-keys 1,6,7,8 "--atomic-energy-values=${E0_VALS}" \
    --scaling std_scaling \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" --device cuda --num-workers "${NUM_WORKERS}" \
    --loss "${LOSS}" --energy-weight "${ENERGY_WEIGHT}" \
    --force-weight "${FORCE_WEIGHT}" --stress-weight 0 \
    --lr "${LR}" --lr-scheduler exp --lr-scheduler-gamma "${LR_GAMMA}" \
    --optimizer adamw --optimizer-param-groups mace --weight-decay "${WEIGHT_DECAY}" \
    --amsgrad --max-grad-norm "${MAX_GRAD_NORM}" \
    "$@"
}

for raw_seed in "${SEEDS_ARR[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  for raw_mode in "${MODES_ARR[@]}"; do
    mode="$(echo "${raw_mode}" | xargs)"
    old_job="revised_ethanol_${mode}_seed${seed}_epochs100"
    job="revised_ethanol_${mode}_seed${seed}_epochs${EPOCHS}_from100"
    case "${mode}" in
      ictd_bridge_u_eager)
        old_ckpt="${OLD_OUT}/checkpoints/${old_job}.pth"
        test -f "${old_ckpt}"
        mapfile -t flags < <(ictd_common_flags ictd-bridge-u --seed "${seed}" --resume-checkpoint "${old_ckpt}" --resume-training-state --checkpoint "${OUT}/checkpoints/${job}.pth" --log-interval 200)
        run_logged "${job}" env PYTHONPATH="${REPO}:${MACE}:${PYTHONPATH:-}" "${PY}" -m mace_ictc.cli.train "${flags[@]}"
        ;;
      ictd_bridge_u_makefx)
        old_ckpt="${OLD_OUT}/checkpoints/${old_job}.pth"
        test -f "${old_ckpt}"
        mapfile -t flags < <(ictd_common_flags ictd-bridge-u --seed "${seed}" --resume-checkpoint "${old_ckpt}" --resume-training-state --train-makefx-compile --makefx-buckets 4 --pad-nodes-to-max --pad-edges-to-max --checkpoint "${OUT}/checkpoints/${job}.pth" --log-interval 200)
        run_logged "${job}" env PYTHONPATH="${REPO}:${MACE}:${PYTHONPATH:-}" "${PY}" -m mace_ictc.cli.train "${flags[@]}"
        ;;
      ictd_cueq_makefx)
        old_ckpt="${OLD_OUT}/checkpoints/${old_job}.pth"
        test -f "${old_ckpt}"
        mapfile -t flags < <(ictd_common_flags cueq --seed "${seed}" --resume-checkpoint "${old_ckpt}" --resume-training-state --train-makefx-compile --makefx-buckets 4 --pad-nodes-to-max --pad-edges-to-max --checkpoint "${OUT}/checkpoints/${job}.pth" --log-interval 200)
        run_logged "${job}" env PYTHONPATH="${REPO}:${MACE}:${PYTHONPATH:-}" "${PY}" -m mace_ictc.cli.train "${flags[@]}"
        ;;
      mace_e3nn)
        copy_mace_checkpoint "${old_job}" "${job}" "${seed}" | tee -a "${OUT}/status.log"
        run_logged "${job}" env PYTHONPATH="${MACE}:${PYTHONPATH:-}" "${PY}" -m mace.cli.run_train \
          --name "${job}" --seed "${seed}" --device cuda --default_dtype "${DTYPE}" \
          --log_dir "${OUT}/logs" --model_dir "${OUT}/models" --checkpoints_dir "${OUT}/checkpoints" --results_dir "${OUT}/results" \
          --model ScaleShiftMACE --r_max "${R_MAX}" --radial_type bessel --num_radial_basis 8 --num_cutoff_basis 6 \
          --max_ell "${MAX_ELL}" --num_interactions "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" --use_reduced_cg True \
          --num_channels "${CHANNELS}" --max_L "${HIDDEN_LMAX}" --hidden_irreps "${CHANNELS}x0e + ${CHANNELS}x1o" --MLP_irreps "${READOUT_HIDDEN}x0e" --radial_MLP "[64, 64, 64]" \
          --interaction RealAgnosticResidualInteractionBlock --interaction_first RealAgnosticResidualInteractionBlock \
          --train_file "${DATA}/train.extxyz" --valid_file "${DATA}/val.extxyz" --energy_key energy --forces_key forces \
          --atomic_numbers "[1, 6, 7, 8]" --E0s "${MACE_E0S}" --avg_num_neighbors "${AVG_NEIGHBORS}" --scaling std_scaling \
          --loss weighted --energy_weight "${ENERGY_WEIGHT}" --forces_weight "${FORCE_WEIGHT}" \
          --batch_size "${BATCH_SIZE}" --valid_batch_size "${BATCH_SIZE}" --max_num_epochs "${EPOCHS}" \
          --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" --optimizer adamw --scheduler ExponentialLR --lr_scheduler_gamma "${LR_GAMMA}" --amsgrad \
          --num_workers "${NUM_WORKERS}" --compute_forces True --compute_stress False --eval_interval 1 --restart_latest
        ;;
      mace_cueq)
        copy_mace_checkpoint "${old_job}" "${job}" "${seed}" | tee -a "${OUT}/status.log"
        run_logged "${job}" env PYTHONPATH="${MACE}:${PYTHONPATH:-}" "${PY}" -m mace.cli.run_train \
          --name "${job}" --seed "${seed}" --device cuda --default_dtype "${DTYPE}" \
          --log_dir "${OUT}/logs" --model_dir "${OUT}/models" --checkpoints_dir "${OUT}/checkpoints" --results_dir "${OUT}/results" \
          --model ScaleShiftMACE --r_max "${R_MAX}" --radial_type bessel --num_radial_basis 8 --num_cutoff_basis 6 \
          --max_ell "${MAX_ELL}" --num_interactions "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" --use_reduced_cg True \
          --num_channels "${CHANNELS}" --max_L "${HIDDEN_LMAX}" --hidden_irreps "${CHANNELS}x0e + ${CHANNELS}x1o" --MLP_irreps "${READOUT_HIDDEN}x0e" --radial_MLP "[64, 64, 64]" \
          --interaction RealAgnosticResidualInteractionBlock --interaction_first RealAgnosticResidualInteractionBlock \
          --train_file "${DATA}/train.extxyz" --valid_file "${DATA}/val.extxyz" --energy_key energy --forces_key forces \
          --atomic_numbers "[1, 6, 7, 8]" --E0s "${MACE_E0S}" --avg_num_neighbors "${AVG_NEIGHBORS}" --scaling std_scaling \
          --loss weighted --energy_weight "${ENERGY_WEIGHT}" --forces_weight "${FORCE_WEIGHT}" \
          --batch_size "${BATCH_SIZE}" --valid_batch_size "${BATCH_SIZE}" --max_num_epochs "${EPOCHS}" \
          --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" --optimizer adamw --scheduler ExponentialLR --lr_scheduler_gamma "${LR_GAMMA}" --amsgrad \
          --num_workers "${NUM_WORKERS}" --compute_forces True --compute_stress False --eval_interval 1 \
          --enable_cueq True --only_cueq True --restart_latest
        ;;
      *)
        echo "Unknown mode: ${mode}" >&2
        exit 4
        ;;
    esac
  done
done

date | tee -a "${OUT}/status.log"
echo "ALL_DONE ${OUT}" | tee -a "${OUT}/status.log"
