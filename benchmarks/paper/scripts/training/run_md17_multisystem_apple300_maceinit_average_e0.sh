#!/usr/bin/env bash
set -euo pipefail

# Apple-to-apple 300-epoch rMD17 training benchmark for additional molecules.
# This script is intended for the RTX 4090 host. It runs MACE-ICTC and
# mace-torch baselines with matched architecture, optimizer, scheduler, E/F
# weights, average-E0 convention, and seeds.

PY="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
REPO="${MACE_ICTC_REPO:-/home/ylzhang/lrx/MACE-ICTC}"
MACE="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/tmp/mace_ictc_public_md17}"
OUT="${OUT_ROOT:-/tmp/mace_ictc_train_multisystem_apple300_maceinit_average_e0_$(date +%Y%m%d_%H%M%S)}"

SYSTEMS_CSV="${SYSTEMS:-revised_benzene,revised_aspirin}"
SEEDS_CSV="${SEEDS:-20260616,20260617,20260618}"
MODES_CSV="${MODES:-ictd_bridge_u_eager,ictd_bridge_u_makefx,ictd_cueq_makefx,mace_e3nn,mace_cueq}"

ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100.0}"
LOSS="${LOSS:-mse}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"

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

mkdir -p "${OUT}/logs" "${OUT}/checkpoints" "${OUT}/models" "${OUT}/results" "${OUT}/commands" "${OUT}/metadata"

IFS=',' read -r -a SYSTEMS_ARR <<< "${SYSTEMS_CSV}"
IFS=',' read -r -a SEEDS_ARR <<< "${SEEDS_CSV}"
IFS=',' read -r -a MODES_ARR <<< "${MODES_CSV}"

cat > "${OUT}/metadata.json" <<EOF
{
  "benchmark": "apple_to_apple_training_300_epoch_mace_compatible_init_average_e0_multisystem",
  "repo": "${REPO}",
  "mace_torch_path": "${MACE}",
  "data_root": "${DATA_ROOT}",
  "systems_csv": "${SYSTEMS_CSV}",
  "target_epochs": ${EPOCHS},
  "seeds_csv": "${SEEDS_CSV}",
  "modes_csv": "${MODES_CSV}",
  "fixed_loss": {"loss": "${LOSS}", "energy_weight": ${ENERGY_WEIGHT}, "force_weight": ${FORCE_WEIGHT}, "stress_weight": 0.0},
  "scheduler": {"type": "ExponentialLR", "gamma": ${LR_GAMMA}},
  "optimizer": {"type": "AdamW", "lr": ${LR}, "weight_decay": ${WEIGHT_DECAY}, "amsgrad": true},
  "architecture": {"channels": ${CHANNELS}, "hidden_lmax": ${HIDDEN_LMAX}, "max_ell": ${MAX_ELL}, "num_interactions": ${NUM_INTERACTIONS}, "correlation": ${CORRELATION}, "readout_hidden_channels": ${READOUT_HIDDEN}, "first_layer_self_connection": true, "use_reduced_cg": true},
  "radial": {"type": "bessel", "num_basis": 8, "polynomial_cutoff_p": 6, "r_max": ${R_MAX}},
  "scaling": {"mode": "std_scaling", "avg_num_neighbors": ${AVG_NEIGHBORS}, "e0_rule": "For each fixed-composition molecule, solve the minimum-norm average-E0 linear system E0_Z = mean(E) n_Z / sum_Z n_Z^2, then use the same E0s in MACE-ICTC and mace-torch."}
}
EOF

echo "OUT=${OUT}" | tee "${OUT}/status.log"
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

write_system_metadata() {
  local system="$1"
  local data="${DATA_ROOT}/${system}"
  "${PY}" - "${data}" "${OUT}/metadata/${system}.env" "${OUT}/metadata/${system}.json" "${CHANNELS}" "${HIDDEN_LMAX}" <<'PY'
from __future__ import annotations

import json
import shlex
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import iread

data = Path(sys.argv[1])
env_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])
channels = int(sys.argv[4])
hidden_lmax = int(sys.argv[5])

energies = []
composition = None
n_train = 0
for atoms in iread(str(data / "train.extxyz"), index=":"):
    n_train += 1
    if composition is None:
        composition = Counter(int(z) for z in atoms.get_atomic_numbers())
    energies.append(float(atoms.get_potential_energy()))

if composition is None:
    raise SystemExit(f"empty training set: {data}")

mean_energy = float(np.mean(energies))
denom = int(sum(v * v for v in composition.values()))
items = sorted((int(z), int(n)) for z, n in composition.items())
e0 = [(z, mean_energy * n / denom) for z, n in items]

keys = ",".join(str(z) for z, _ in e0)
vals = ",".join(f"{v:.16g}" for _, v in e0)
mace_e0s = "{" + ", ".join(f"{z}: {v:.16g}" for z, v in e0) + "}"
atomic_numbers = "[" + ", ".join(str(z) for z, _ in e0) + "]"
hidden_irreps = " + ".join(f"{channels}x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(hidden_lmax + 1))

env_path.write_text(
    "\n".join(
        [
            f"E0_KEYS={shlex.quote(keys)}",
            f"E0_VALS={shlex.quote(vals)}",
            f"MACE_E0S={shlex.quote(mace_e0s)}",
            f"ATOMIC_NUMBERS={shlex.quote(atomic_numbers)}",
            f"HIDDEN_IRREPS={shlex.quote(hidden_irreps)}",
            "",
        ]
    )
)
json_path.write_text(
    json.dumps(
        {
            "data": str(data),
            "n_train": n_train,
            "composition": {str(z): n for z, n in items},
            "mean_train_energy": mean_energy,
            "minimum_norm_e0": {str(z): v for z, v in e0},
            "atomic_numbers": [z for z, _ in e0],
            "hidden_irreps": hidden_irreps,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
}

ictd_common_flags() {
  local data="$1"
  local backend="$2"
  local e0_keys="$3"
  local e0_vals="$4"
  shift 4
  printf '%s\n' \
    --data-dir "${data}" --train-prefix train --val-prefix val \
    --channels "${CHANNELS}" --lmax "${HIDDEN_LMAX}" --max-ell "${MAX_ELL}" \
    --num-interaction "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" \
    --product-backend "${backend}" --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels "${READOUT_HIDDEN}" \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius "${R_MAX}" --avg-num-neighbors "${AVG_NEIGHBORS}" \
    --atomic-energy-keys "${e0_keys}" "--atomic-energy-values=${e0_vals}" \
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

for raw_system in "${SYSTEMS_ARR[@]}"; do
  system="$(echo "${raw_system}" | xargs)"
  data="${DATA_ROOT}/${system}"
  if [[ ! -f "${data}/train.extxyz" || ! -f "${data}/processed_train.h5" ]]; then
    echo "missing prepared dataset for ${system}: ${data}" >&2
    exit 2
  fi
  write_system_metadata "${system}"
  # shellcheck disable=SC1090
  source "${OUT}/metadata/${system}.env"

  for raw_seed in "${SEEDS_ARR[@]}"; do
    seed="$(echo "${raw_seed}" | xargs)"
    for raw_mode in "${MODES_ARR[@]}"; do
      mode="$(echo "${raw_mode}" | xargs)"
      job="${system}_${mode}_seed${seed}_epochs${EPOCHS}"
      case "${mode}" in
        ictd_bridge_u_eager)
          mapfile -t flags < <(ictd_common_flags "${data}" ictd-bridge-u "${E0_KEYS}" "${E0_VALS}" --seed "${seed}" --checkpoint "${OUT}/checkpoints/${job}.pth" --log-interval 200)
          run_logged "${job}" env PYTHONPATH="${REPO}:${MACE}:${PYTHONPATH:-}" "${PY}" -m mace_ictc.cli.train "${flags[@]}"
          ;;
        ictd_bridge_u_makefx)
          mapfile -t flags < <(ictd_common_flags "${data}" ictd-bridge-u "${E0_KEYS}" "${E0_VALS}" --seed "${seed}" --train-makefx-compile --makefx-buckets 4 --pad-nodes-to-max --pad-edges-to-max --checkpoint "${OUT}/checkpoints/${job}.pth" --log-interval 200)
          run_logged "${job}" env PYTHONPATH="${REPO}:${MACE}:${PYTHONPATH:-}" "${PY}" -m mace_ictc.cli.train "${flags[@]}"
          ;;
        ictd_cueq_makefx)
          mapfile -t flags < <(ictd_common_flags "${data}" cueq "${E0_KEYS}" "${E0_VALS}" --seed "${seed}" --train-makefx-compile --makefx-buckets 4 --pad-nodes-to-max --pad-edges-to-max --checkpoint "${OUT}/checkpoints/${job}.pth" --log-interval 200)
          run_logged "${job}" env PYTHONPATH="${REPO}:${MACE}:${PYTHONPATH:-}" "${PY}" -m mace_ictc.cli.train "${flags[@]}"
          ;;
        mace_e3nn)
          run_logged "${job}" env PYTHONPATH="${MACE}:${PYTHONPATH:-}" "${PY}" -m mace.cli.run_train \
            --name "${job}" --seed "${seed}" --device cuda --default_dtype "${DTYPE}" \
            --log_dir "${OUT}/logs" --model_dir "${OUT}/models" --checkpoints_dir "${OUT}/checkpoints" --results_dir "${OUT}/results" \
            --model ScaleShiftMACE --r_max "${R_MAX}" --radial_type bessel --num_radial_basis 8 --num_cutoff_basis 6 \
            --max_ell "${MAX_ELL}" --num_interactions "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" --use_reduced_cg True \
            --num_channels "${CHANNELS}" --max_L "${HIDDEN_LMAX}" --hidden_irreps "${HIDDEN_IRREPS}" --MLP_irreps "${READOUT_HIDDEN}x0e" --radial_MLP "[64, 64, 64]" \
            --interaction RealAgnosticResidualInteractionBlock --interaction_first RealAgnosticResidualInteractionBlock \
            --train_file "${data}/train.extxyz" --valid_file "${data}/val.extxyz" --energy_key energy --forces_key forces \
            --atomic_numbers "${ATOMIC_NUMBERS}" --E0s "${MACE_E0S}" --avg_num_neighbors "${AVG_NEIGHBORS}" --scaling std_scaling \
            --loss weighted --energy_weight "${ENERGY_WEIGHT}" --forces_weight "${FORCE_WEIGHT}" \
            --batch_size "${BATCH_SIZE}" --valid_batch_size "${BATCH_SIZE}" --max_num_epochs "${EPOCHS}" \
            --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" --optimizer adamw --scheduler ExponentialLR --lr_scheduler_gamma "${LR_GAMMA}" --amsgrad \
            --num_workers "${NUM_WORKERS}" --compute_forces True --compute_stress False --eval_interval 1
          ;;
        mace_cueq)
          run_logged "${job}" env PYTHONPATH="${MACE}:${PYTHONPATH:-}" "${PY}" -m mace.cli.run_train \
            --name "${job}" --seed "${seed}" --device cuda --default_dtype "${DTYPE}" \
            --log_dir "${OUT}/logs" --model_dir "${OUT}/models" --checkpoints_dir "${OUT}/checkpoints" --results_dir "${OUT}/results" \
            --model ScaleShiftMACE --r_max "${R_MAX}" --radial_type bessel --num_radial_basis 8 --num_cutoff_basis 6 \
            --max_ell "${MAX_ELL}" --num_interactions "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" --use_reduced_cg True \
            --num_channels "${CHANNELS}" --max_L "${HIDDEN_LMAX}" --hidden_irreps "${HIDDEN_IRREPS}" --MLP_irreps "${READOUT_HIDDEN}x0e" --radial_MLP "[64, 64, 64]" \
            --interaction RealAgnosticResidualInteractionBlock --interaction_first RealAgnosticResidualInteractionBlock \
            --train_file "${data}/train.extxyz" --valid_file "${data}/val.extxyz" --energy_key energy --forces_key forces \
            --atomic_numbers "${ATOMIC_NUMBERS}" --E0s "${MACE_E0S}" --avg_num_neighbors "${AVG_NEIGHBORS}" --scaling std_scaling \
            --loss weighted --energy_weight "${ENERGY_WEIGHT}" --forces_weight "${FORCE_WEIGHT}" \
            --batch_size "${BATCH_SIZE}" --valid_batch_size "${BATCH_SIZE}" --max_num_epochs "${EPOCHS}" \
            --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" --optimizer adamw --scheduler ExponentialLR --lr_scheduler_gamma "${LR_GAMMA}" --amsgrad \
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
done

date | tee -a "${OUT}/status.log"
echo "ALL_DONE ${OUT}" | tee -a "${OUT}/status.log"
