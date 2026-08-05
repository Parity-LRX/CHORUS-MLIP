#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
TACE_REPO="${TACE_REPO:-/home/ylzhang/tace_chorus_benchmark}"
TACE_PY="${TACE_PY:-/home/ylzhang/tace_chorus_venv/bin}"
RUN_ROOT="${RUN_ROOT:-/home/ylzhang/tace_chorus_runs/large_missing_seed20260616}"
SEED="${SEED:-20260616}"
CHANNELS="${CHANNELS:-36}"
SYSTEMS="${SYSTEMS:-t1x bucky}"
PRIORITY_REQUEST_FILE="${PRIORITY_REQUEST_FILE:-/home/ylzhang/chorus_runs/tece_c48_priority_20260726/REQUESTED}"
PRIORITY_DONE_FILE="${PRIORITY_DONE_FILE:-/home/ylzhang/chorus_runs/tece_c48_priority_20260726/DONE}"

# When the recommended-capacity C48 xxMD comparison is explicitly promoted,
# defer the lower-priority C36 T1x/Bucky completion until that three-system
# suite finishes.  The request marker is created only by the orchestration
# step, so ordinary standalone runs are unaffected.
if [[ -f "${PRIORITY_REQUEST_FILE}" && ! -f "${PRIORITY_DONE_FILE}" ]]; then
  echo "WAIT_TECE_C48_PRIORITY $(date -Is)"
  while [[ ! -f "${PRIORITY_DONE_FILE}" ]]; do
    sleep 30
  done
  echo "TECE_C48_PRIORITY_READY $(date -Is)"
fi

declare -A config
declare -A train_file
declare -A val_file
declare -A test_file
declare -A energy_key
declare -A forces_key
declare -A batch_size
declare -A steps
declare -A epochs
declare -A atomic_energy

config[t1x]=tace_transition1x_tece_s.yaml
train_file[t1x]=/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616/train.extxyz
val_file[t1x]=/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616/val.extxyz
test_file[t1x]=/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616/test.extxyz
energy_key[t1x]=energy
forces_key[t1x]=forces
batch_size[t1x]=16
steps[t1x]=100000
epochs[t1x]=32
atomic_energy[t1x]='{1: -13.62222753701504, 6: -1029.4130839658328, 7: -1484.8710358098756, 8: -2041.8396277138045}'

config[bucky]=tace_buckyball_tece_s.yaml
train_file[bucky]=/home/ylzhang/tace_chorus_data/buckyball/train.extxyz
val_file[bucky]=/home/ylzhang/tace_chorus_data/buckyball/val.extxyz
test_file[bucky]=/home/ylzhang/tace_chorus_data/buckyball/test.extxyz
energy_key[bucky]=Energy
forces_key[bucky]=force
batch_size[bucky]=4
steps[bucky]=45000
epochs[bucky]=300
atomic_energy[bucky]='{1: -230.09867339, 6: -986.13717166}'

mkdir -p "${RUN_ROOT}"
status="${RUN_ROOT}/status.log"

for system in ${SYSTEMS}; do
  if [[ -z "${config[$system]+set}" ]]; then
    echo "unknown system: ${system}" >&2
    exit 2
  fi
  for path in \
    "${train_file[$system]}" "${val_file[$system]}" "${test_file[$system]}"; do
    if [[ ! -f "${path}" ]]; then
      echo "missing dataset file: ${path}" >&2
      exit 3
    fi
  done

  run_dir="${RUN_ROOT}/${system}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/.complete" ]]; then
    echo "SKIP ${system} $(date -Is)" | tee -a "${status}"
    continue
  fi

  echo \
    "START ${system} steps=${steps[$system]} epochs=${epochs[$system]} $(date -Is)" \
    | tee -a "${status}"
  cd "${run_dir}"
  export NVIDIA_TF32_OVERRIDE=0
  export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
  env TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=0 \
    "${TACE_PY}/tace-train" \
      --config-path "${TACE_REPO}/example/train" -cn "${config[$system]}" \
      "misc.project_name=TECE_Large_${system}_Seed${SEED}" \
      "misc.global_seed=${SEED}" \
      misc.allow_tf32=false \
      "dataset.split_seed=${SEED}" \
      "dataset.train_file=${train_file[$system]}" \
      "dataset.valid_file=${val_file[$system]}" \
      "dataset.keys.energy_key=${energy_key[$system]}" \
      "dataset.keys.forces_key=${forces_key[$system]}" \
      "dataset.train_dataloader.batch_size=${batch_size[$system]}" \
      "dataset.valid_dataloader.batch_size=${batch_size[$system]}" \
      "trainer.max_steps=${steps[$system]}" \
      "trainer.max_epochs=${epochs[$system]}" \
      "scheduler.T_max=${steps[$system]}" \
      model.config.fidelity.0.name=PBE \
      "model.config.fidelity.0.atomic_energy=${atomic_energy[$system]}" \
      "model.config.num_channel=${CHANNELS}" \
      model.config.mmax=2 model.config.Lmax=2 model.config.lmax=2 \
      model.config.num_layers=2 model.config.product_basis.correlation=3 \
      synth_metric.monitor_metric_name=val/forces_mae \
      callbacks.checkpoint_epoch.monitor=val/forces_mae \
      callbacks.checkpoint_epoch.save_top_k=-1 \
      '~callbacks.ema' \
      >train.log 2>&1

  best_ckpt="$("${TACE_PY}/python" - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_root = Path("lightning_logs")
versions = sorted(event_root.glob("version_*"), key=lambda path: int(path.name.split("_")[-1]))
if not versions:
    raise SystemExit("no TensorBoard validation log found")
events = EventAccumulator(str(versions[-1]))
events.Reload()
points = events.Scalars("val/forces_mae")
if not points:
    raise SystemExit("no val/forces_mae values found")
best = min(points, key=lambda point: (point.value, point.step))
checkpoint_step = best.step + 1
candidates = sorted(Path("checkpoints_epoch").glob(f"TECE-*-{checkpoint_step}.ckpt"))
if len(candidates) != 1:
    raise SystemExit(
        f"expected one checkpoint for validation step {checkpoint_step}, got {candidates}"
    )
Path("selected_by_force_mae.txt").write_text(
    f"{candidates[0]}\nval_forces_mae={best.value}\n"
    f"tensorboard_step={best.step}\ncheckpoint_step={checkpoint_step}\n"
    "selection_rule=minimum validation Force MAE; test never selects\n"
)
print(candidates[0])
PY
)"
  if [[ -z "${best_ckpt}" ]]; then
    echo "ERROR no Force-MAE-selected checkpoint for ${system}" | tee -a "${status}"
    exit 4
  fi

  "${TACE_PY}/tace-eval" \
    -i "${val_file[$system]}" \
    -m "${best_ckpt}" -t 1 -e 1 -b "${batch_size[$system]}" \
    --device cuda --dtype float32 --nl_backend matscipy \
    --energy_key "${energy_key[$system]}" \
    --forces_key "${forces_key[$system]}" \
    >val.log 2>&1
  "${TACE_PY}/tace-eval" \
    -i "${test_file[$system]}" \
    -m "${best_ckpt}" -t 1 -e 1 -b "${batch_size[$system]}" \
    --device cuda --dtype float32 --nl_backend matscipy \
    --energy_key "${energy_key[$system]}" \
    --forces_key "${forces_key[$system]}" \
    >test.log 2>&1
  touch .complete
  echo \
    "END ${system} Force-MAE checkpoint=${best_ckpt} $(date -Is)" \
    | tee -a "${status}"
done

echo "ALL_REQUESTED_OK systems=${SYSTEMS} $(date -Is)" | tee -a "${status}"
