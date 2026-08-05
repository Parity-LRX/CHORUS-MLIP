#!/usr/bin/env bash
set -euo pipefail

TACE_REPO="${TACE_REPO:-/home/ylzhang/tace_chorus_benchmark}"
TACE_PY="${TACE_PY:-/home/ylzhang/tace_chorus_venv/bin}"
CONFIG="${CONFIG:-tace_buckyball_tece_s.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5}"
RUN_ROOT="${RUN_ROOT:-/home/ylzhang/tace_chorus_runs/xxmd_tece_large_mae_seed20260616}"
SEED="${SEED:-20260616}"
CHANNELS="${CHANNELS:-36}"
SYSTEMS="${SYSTEMS:-mal sti dia}"

# Official temporal splits and exactly the same optimizer-step budgets as the
# large CHORUS comparison.  C=36 is the parameter-matched configuration;
# callers can select the developer-recommended C=48 capacity independently.
declare -A steps=( [mal]=45000 [sti]=45000 [dia]=45000 )
declare -A epochs=( [mal]=52 [sti]=57 [dia]=59 )
declare -A e0
e0[sti]='{1: -518.6243586880087, 6: -605.0617518026768}'
e0[dia]='{1: -1531.021284830928, 6: -1913.77660603866, 16: -382.755321207732}'
e0[mal]='{1: -1001.330686760234, 6: -750.9980150701755, 8: -500.665343380117}'

mkdir -p "${RUN_ROOT}"
status="${RUN_ROOT}/status.log"

for system in ${SYSTEMS}; do
  if [[ -z "${steps[$system]+set}" ]]; then
    echo "unknown system: ${system}" >&2
    exit 2
  fi
  run_dir="${RUN_ROOT}/${system}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/.complete" ]]; then
    echo "SKIP ${system} $(date)" | tee -a "${status}"
    continue
  fi

  echo "START ${system} steps=${steps[$system]} epochs=${epochs[$system]} $(date)" | tee -a "${status}"
  cd "${run_dir}"
  export NVIDIA_TF32_OVERRIDE=0
  export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
  env TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=0 \
    "${TACE_PY}/tace-train" \
      --config-path "${TACE_REPO}/example/train" -cn "${CONFIG}" \
      "misc.project_name=TECE_Large_xxMD_${system}_Seed${SEED}" \
      "misc.global_seed=${SEED}" \
      misc.allow_tf32=false \
      "dataset.split_seed=${SEED}" \
      "dataset.train_file=${DATA_ROOT}/${system}/train.extxyz" \
      "dataset.valid_file=${DATA_ROOT}/${system}/val.extxyz" \
      dataset.keys.energy_key=energy dataset.keys.forces_key=forces \
      dataset.train_dataloader.batch_size=16 dataset.valid_dataloader.batch_size=16 \
      "trainer.max_steps=${steps[$system]}" "trainer.max_epochs=${epochs[$system]}" \
      "scheduler.T_max=${steps[$system]}" \
      "model.config.fidelity.0.name=M06" \
      "model.config.fidelity.0.atomic_energy=${e0[$system]}" \
      "model.config.num_channel=${CHANNELS}" \
      model.config.mmax=2 model.config.Lmax=2 model.config.lmax=2 \
      model.config.num_layers=2 model.config.product_basis.correlation=3 \
      synth_metric.monitor_metric_name=val/forces_mae \
      callbacks.checkpoint_epoch.monitor=val/forces_mae \
      callbacks.checkpoint_epoch.save_top_k=-1 \
      '~callbacks.ema' \
      > train.log 2>&1

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
)
print(candidates[0])
PY
)"
  if [[ -z "${best_ckpt}" ]]; then
    echo "ERROR no Force-MAE-selected checkpoint for ${system}" | tee -a "${status}"
    exit 4
  fi
  "${TACE_PY}/tace-eval" \
    -i "${DATA_ROOT}/${system}/val.extxyz" \
    -m "${best_ckpt}" -t 1 -e 1 -b 16 \
    --device cuda --dtype float32 --nl_backend matscipy \
    --energy_key energy --forces_key forces \
    > val.log 2>&1
  "${TACE_PY}/tace-eval" \
    -i "${DATA_ROOT}/${system}/test.extxyz" \
    -m "${best_ckpt}" -t 1 -e 1 -b 16 \
    --device cuda --dtype float32 --nl_backend matscipy \
    --energy_key energy --forces_key forces \
    > test.log 2>&1
  touch .complete
  echo "END ${system} Force-MAE checkpoint=${best_ckpt} $(date)" | tee -a "${status}"
done

echo "ALL_OK $(date)" | tee -a "${status}"
