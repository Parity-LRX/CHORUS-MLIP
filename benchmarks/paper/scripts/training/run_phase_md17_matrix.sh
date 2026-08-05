#!/usr/bin/env bash
set -euo pipefail

# Matched rMD17 comparison for the existing ICTC operator and phase variants.
# Defaults reproduce the archived 300-epoch, three-seed protocol.
# Set EPOCHS/SEEDS/MAX_STEPS explicitly only for a labelled smoke run.

PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/tmp/mace_ictd_public_md17}"
DATASETS="${DATASETS:-revised_aspirin,revised_ethanol,revised_benzene}"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/md17_three_system_$(date +%Y%m%d_%H%M%S)}"

SEEDS="${SEEDS:-20260616,20260617,20260618}"
MODES="${MODES:-ictc_bridge_u_eager,ictc_phase_unit_eager,ictc_phase_softplus_eager}"
EPOCHS="${EPOCHS:-300}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHANNELS="${CHANNELS:-64}"
HIDDEN_LMAX="${HIDDEN_LMAX:-1}"
MAX_ELL="${MAX_ELL:-2}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
CORRELATION="${CORRELATION:-2}"
R_MAX="${R_MAX:-4.5}"
LR="${LR:-0.001}"
MIN_LR="${MIN_LR:-1e-6}"
LR_SCHEDULER="${LR_SCHEDULER:-exp}"
LR_GAMMA="${LR_GAMMA:-0.9993}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-7}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-100.0}"
AVG_NEIGHBORS="${AVG_NEIGHBORS:-8.0}"
READOUT_HIDDEN="${READOUT_HIDDEN:-64}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
PHASE_HIDDEN_CHANNELS="${PHASE_HIDDEN_CHANNELS:-32}"
PHASE_SCALE_INIT="${PHASE_SCALE_INIT:-0.05}"
PHASE_PLACEMENT="${PHASE_PLACEMENT:-post-product}"
PHASE_DENSITY_RANK="${PHASE_DENSITY_RANK:-8}"
PHASE_COHERENCE_INIT="${PHASE_COHERENCE_INIT:-0.1}"
ATTN_HEADS="${ATTN_HEADS:-4}"
ATTN_MODE="${ATTN_MODE:-density-preserving}"
ATTN_SCOPE="${ATTN_SCOPE:-final}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_MAKEFX_COMPILE="${TRAIN_MAKEFX_COMPILE:-0}"
MAKEFX_BUCKETS="${MAKEFX_BUCKETS:-4}"
MAKEFX_MAX_SLOTS="${MAKEFX_MAX_SLOTS:-8}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
EMA_DECAY="${EMA_DECAY:-0.0}"
EMA_START_STEP="${EMA_START_STEP:-0}"

if ! [[ "${PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_JOBS must be a positive integer, got ${PARALLEL_JOBS}" >&2
  exit 2
fi
if [[ "${TRAIN_MAKEFX_COMPILE}" != "0" && "${TRAIN_MAKEFX_COMPILE}" != "1" ]]; then
  echo "TRAIN_MAKEFX_COMPILE must be 0 or 1, got ${TRAIN_MAKEFX_COMPILE}" >&2
  exit 2
fi
if ! [[ "${MAKEFX_MAX_SLOTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAKEFX_MAX_SLOTS must be a positive integer, got ${MAKEFX_MAX_SLOTS}" >&2
  exit 2
fi

MAKEFX_JSON=false
if [[ "${TRAIN_MAKEFX_COMPILE}" == "1" ]]; then
  MAKEFX_JSON=true
fi

mkdir -p "${OUT_ROOT}"/{logs,checkpoints,commands,metadata,analysis}
IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
IFS=',' read -r -a MODE_ARRAY <<< "${MODES}"

cat > "${OUT_ROOT}/matrix_metadata.json" <<EOF
{
  "benchmark": "phase_hermitian_rmd17_matched_archived_baseline",
  "datasets": "${DATASETS}",
  "data_root": "${DATA_ROOT}",
  "mace_torch_path": "${MACE_TORCH_PATH}",
  "modes": "${MODES}",
  "seeds": "${SEEDS}",
  "epochs": ${EPOCHS},
  "max_steps": "${MAX_STEPS}",
  "architecture": {"channels": ${CHANNELS}, "hidden_lmax": ${HIDDEN_LMAX}, "max_ell": ${MAX_ELL}, "num_interactions": ${NUM_INTERACTIONS}, "correlation": ${CORRELATION}, "readout_hidden_channels": ${READOUT_HIDDEN}, "first_layer_self_connection": true, "use_reduced_cg": true},
  "radial": {"type": "bessel", "num_basis": 8, "polynomial_cutoff_p": 6, "r_max": ${R_MAX}},
  "optimizer": {"type": "AdamW", "lr": ${LR}, "min_lr": ${MIN_LR}, "weight_decay": ${WEIGHT_DECAY}, "amsgrad": true, "scheduler": "${LR_SCHEDULER}", "gamma": ${LR_GAMMA}},
  "loss": {"type": "mse", "energy_weight": ${ENERGY_WEIGHT}, "force_weight": ${FORCE_WEIGHT}, "stress_weight": 0.0},
  "phase": {"hidden_channels": ${PHASE_HIDDEN_CHANNELS}, "residual_scale_init": ${PHASE_SCALE_INIT}, "default_placement": "${PHASE_PLACEMENT}", "density_rank": ${PHASE_DENSITY_RANK}, "adaptive_coherence_init": ${PHASE_COHERENCE_INIT}, "scope_is_mode_specific": true},
  "attention": {"heads": ${ATTN_HEADS}, "mode": "${ATTN_MODE}", "scope": "${ATTN_SCOPE}"},
  "execution": {"train_makefx_compile": ${MAKEFX_JSON}, "require_makefx": ${MAKEFX_JSON}, "makefx_buckets": "${MAKEFX_BUCKETS}", "makefx_max_slots": ${MAKEFX_MAX_SLOTS}, "pad_nodes_to_max": ${MAKEFX_JSON}, "pad_edges_to_max": ${MAKEFX_JSON}},
  "parallel_jobs": ${PARALLEL_JOBS},
  "average_e0_rule": "minimum-norm E0_Z = mean(E) n_Z / sum_Z n_Z^2",
  "device": "${DEVICE}",
  "dtype": "${DTYPE}"
}
EOF

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
  local log="${OUT_ROOT}/logs/${name}.log"
  write_command_file "${OUT_ROOT}/commands/${name}.sh" "$@"
  echo "START ${name} $(date)" | tee -a "${OUT_ROOT}/status.log"
  set +e
  /usr/bin/time -f 'WALL_SECONDS %e' "$@" > "${log}" 2>&1
  local rc=$?
  set -e
  if [[ "${rc}" != "0" ]]; then
    if grep -q "Training complete" "${log}" && grep -q "ScriptFunction cannot be pickled" "${log}"; then
      echo "OK_WITH_SAVE_WARNING ${name} $(date)" | tee -a "${OUT_ROOT}/status.log"
      return 0
    fi
    echo "FAIL ${name} rc=${rc} $(date) log=${log}" | tee -a "${OUT_ROOT}/status.log"
    return "${rc}"
  fi
  echo "OK ${name} $(date)" | tee -a "${OUT_ROOT}/status.log"
}

ACTIVE_PIDS=()
RUN_FAILED=0

wait_oldest_job() {
  local pid="${ACTIVE_PIDS[0]}"
  if ! wait "${pid}"; then
    RUN_FAILED=1
  fi
  ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
}

queue_logged() {
  run_logged "$@" &
  ACTIVE_PIDS+=("$!")
  if (( ${#ACTIVE_PIDS[@]} >= PARALLEL_JOBS )); then
    wait_oldest_job
  fi
}

write_system_metadata() {
  local dataset="$1"
  local data="${DATA_ROOT}/${dataset}"
  "${PYTHON_BIN}" - "${data}" "${OUT_ROOT}/metadata/${dataset}.env" "${OUT_ROOT}/metadata/${dataset}.json" <<'PY'
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
energies = []
composition = None
for atoms in iread(str(data / "train.extxyz"), index=":"):
    if composition is None:
        composition = Counter(int(z) for z in atoms.get_atomic_numbers())
    energies.append(float(atoms.get_potential_energy()))
if composition is None:
    raise SystemExit(f"empty training set: {data}")
mean_energy = float(np.mean(energies))
denom = sum(n * n for n in composition.values())
items = sorted(composition.items())
e0 = [(z, mean_energy * n / denom) for z, n in items]
keys = ",".join(str(z) for z, _ in e0)
values = ",".join(f"{value:.16g}" for _, value in e0)
env_path.write_text(f"E0_KEYS={shlex.quote(keys)}\nE0_VALS={shlex.quote(values)}\n")
json_path.write_text(json.dumps({
    "data": str(data),
    "n_train": len(energies),
    "composition": {str(z): n for z, n in items},
    "mean_train_energy": mean_energy,
    "minimum_norm_e0": {str(z): value for z, value in e0},
}, indent=2, sort_keys=True) + "\n")
PY
}

ictc_common_flags() {
  local data="$1"
  local e0_keys="$2"
  local e0_values="$3"
  shift 3
  printf '%s\n' \
    --data-dir "${data}" --train-prefix train --val-prefix val \
    --channels "${CHANNELS}" --lmax "${HIDDEN_LMAX}" --max-ell "${MAX_ELL}" \
    --num-interaction "${NUM_INTERACTIONS}" --correlation "${CORRELATION}" \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels "${READOUT_HIDDEN}" \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius "${R_MAX}" --avg-num-neighbors "${AVG_NEIGHBORS}" \
    --atomic-energy-keys "${e0_keys}" "--atomic-energy-values=${e0_values}" \
    --scaling std_scaling --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" --device "${DEVICE}" --num-workers "${NUM_WORKERS}" \
    --loss mse --energy-weight "${ENERGY_WEIGHT}" --force-weight "${FORCE_WEIGHT}" --stress-weight 0 \
    --lr "${LR}" --min-lr "${MIN_LR}" --lr-scheduler "${LR_SCHEDULER}" \
    --lr-scheduler-gamma "${LR_GAMMA}" \
    --optimizer adamw --optimizer-param-groups mace --weight-decay "${WEIGHT_DECAY}" \
    --amsgrad --max-grad-norm "${MAX_GRAD_NORM}" \
    --phase-hidden-channels "${PHASE_HIDDEN_CHANNELS}" --phase-scale-init "${PHASE_SCALE_INIT}" \
    --phase-density-rank "${PHASE_DENSITY_RANK}"
  if [[ "${TRAIN_MAKEFX_COMPILE}" == "1" ]]; then
    printf '%s\n' \
      --train-makefx-compile --require-train-makefx-compile \
      --makefx-buckets "${MAKEFX_BUCKETS}" --makefx-max-slots "${MAKEFX_MAX_SLOTS}" \
      --pad-nodes-to-max --pad-edges-to-max
  fi
  printf '%s\n' "$@"
}

for raw_dataset in "${DATASET_ARRAY[@]}"; do
  dataset="$(echo "${raw_dataset}" | xargs)"
  data="${DATA_ROOT}/${dataset}"
  if [[ ! -f "${data}/train.extxyz" || ! -f "${data}/processed_train.h5" || ! -f "${data}/processed_val.h5" ]]; then
    echo "missing prepared rMD17 data under ${data}" >&2
    exit 3
  fi
  write_system_metadata "${dataset}"
  # shellcheck disable=SC1090
  source "${OUT_ROOT}/metadata/${dataset}.env"

  for raw_seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${raw_seed}" | xargs)"
    for raw_mode in "${MODE_ARRAY[@]}"; do
      mode="$(echo "${raw_mode}" | xargs)"
      phase_args=(--phase-mode none --phase-amplitude unit --phase-coefficient polar --phase-context content --phase-density-pairs full --phase-placement post-product --phase-scope final)
      case "${mode}" in
        ictc_bridge_u_eager|ictc_bridge_u_makefx) ;;
        ictc_phase_unit_eager)
          phase_args=(--phase-mode final-scalar-residual --phase-amplitude unit --phase-placement "${PHASE_PLACEMENT}" --phase-scope final)
          ;;
        ictc_phase_softplus_eager)
          phase_args=(--phase-mode final-scalar-residual --phase-amplitude softplus --phase-placement "${PHASE_PLACEMENT}" --phase-scope final)
          ;;
        ictc_phase_full_l_softplus_eager|ictc_phase_full_l_softplus_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full --phase-normalization avg-neighbors --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_charge2_full_l_eager|ictc_phase_charge2_full_l_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs charge2 --phase-normalization avg-neighbors --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_full_l_local_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full --phase-normalization local-effective --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_full_l_gated_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full-gated --phase-normalization avg-neighbors --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_full_l_gated_local_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full-gated --phase-normalization local-effective --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_full_l_adaptive_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full-adaptive --phase-coherence-init "${PHASE_COHERENCE_INIT}" --phase-normalization avg-neighbors --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_full_l_adaptive_env_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full-adaptive-env --phase-coherence-init "${PHASE_COHERENCE_INIT}" --phase-normalization avg-neighbors --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_full_l_pair_balanced_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full-balanced --phase-normalization avg-neighbors --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_positive_full_l_eager)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient positive --phase-context content --phase-density-pairs full --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_signed_full_l_eager)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient signed --phase-context content --phase-density-pairs full --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_cartesian_full_l_eager)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient cartesian --phase-context content --phase-density-pairs full --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_radial_full_l_eager)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context radial --phase-density-pairs full --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_diagonal_full_l_eager|ictc_phase_diagonal_full_l_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs diagonal --phase-placement pre-product-full-l --phase-scope final)
          ;;
        ictc_phase_diagonal_full_l_all_layers_softplus_eager|ictc_phase_diagonal_full_l_all_layers_softplus_makefx)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs diagonal --phase-normalization avg-neighbors --phase-placement pre-product-full-l --phase-scope persistent)
          ;;
        ictc_attention_eager|ictc_attention_makefx)
          phase_args=(--attn-heads "${ATTN_HEADS}" --attn-mode "${ATTN_MODE}" --attn-scope "${ATTN_SCOPE}" --phase-mode none --phase-amplitude unit --phase-coefficient polar --phase-context content --phase-density-pairs full --phase-placement post-product --phase-scope final)
          ;;
        ictc_attention_legacy_eager|ictc_attention_legacy_makefx)
          phase_args=(--attn-heads "${ATTN_HEADS}" --attn-mode legacy-softmax --attn-scope all --phase-mode none --phase-amplitude unit --phase-coefficient polar --phase-context content --phase-density-pairs full --phase-placement post-product --phase-scope final)
          ;;
        ictc_phase_scalar_persistent_softplus_eager)
          phase_args=(--phase-mode final-scalar-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full --phase-normalization avg-neighbors --phase-placement pre-product-l0 --phase-scope persistent)
          ;;
        ictc_phase_full_l_persistent_softplus_eager)
          phase_args=(--phase-mode final-full-l-residual --phase-amplitude softplus --phase-coefficient polar --phase-context content --phase-density-pairs full --phase-normalization avg-neighbors --phase-placement pre-product-full-l --phase-scope persistent)
          ;;
        *) echo "unknown mode ${mode}" >&2; exit 4 ;;
      esac
      job="${dataset}_${mode}_seed${seed}_epochs${EPOCHS}"
      checkpoint_dir="${OUT_ROOT}/checkpoints"
      if (( PARALLEL_JOBS > 1 )); then
        checkpoint_dir="${checkpoint_dir}/${job}"
        mkdir -p "${checkpoint_dir}"
      fi
      mapfile -t flags < <(ictc_common_flags "${data}" "${E0_KEYS}" "${E0_VALS}" \
        --seed "${seed}" --checkpoint "${checkpoint_dir}/${job}.pth" \
        --keep-checkpoints "${KEEP_CHECKPOINTS}" \
        --ema-decay "${EMA_DECAY}" --ema-start-step "${EMA_START_STEP}" \
        --log-interval 200 "${phase_args[@]}")
      if [[ -n "${MAX_STEPS}" ]]; then
        flags+=(--max-steps "${MAX_STEPS}")
      fi
      queue_logged "${job}" env PYTHONPATH="${MACE_ICTC_REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
        "${PYTHON_BIN}" -m chorus.cli.train "${flags[@]}"
    done
  done
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  wait_oldest_job
done

if (( RUN_FAILED != 0 )); then
  echo "one or more matrix jobs failed; see ${OUT_ROOT}/status.log" >&2
  exit 5
fi

"${PYTHON_BIN}" "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/analyze_md17_convergence.py" \
  "${OUT_ROOT}/logs" --out-dir "${OUT_ROOT}/analysis" --target-epoch "${EPOCHS}" --plots

echo "results: ${OUT_ROOT}"
