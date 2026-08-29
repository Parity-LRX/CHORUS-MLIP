#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-ms25-20260818}"
TACE_REPO="${TACE_REPO:-/home/ylzhang/tace_chorus_benchmark}"
TACE_BIN="${TACE_BIN:-/home/ylzhang/tace_chorus_venv/bin}"
CONFIG="${CONFIG:-${REPO}/benchmarks/paper/external/tece/tace_buckyball_tece_s.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/chorus_data/ms25_20260818/prepared}"
RUN_ROOT="${RUN_ROOT:-/home/ylzhang/chorus_runs/ms25_external_20260818/tece}"
SEED="${SEED:-20260616}"
MAX_STEPS="${MAX_STEPS:-45000}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "${RUN_ROOT}"
status="${RUN_ROOT}/status.log"
failures=0

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${status}"
}

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

while screen_exists ms25_dpa4 || screen_exists ms25_dpa4_transfer_eval; do
  sleep 30
done

for dataset in hea zr-o mgo-2x2; do
  base="${DATA_ROOT}/${dataset}/set1"
  readarray -t metadata < <("${TACE_BIN}/python" - "${base}/metadata.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(x["cutoff_angstrom"])
print("{" + ", ".join(f"{z}: {v:.17g}" for z,v in zip(x["atomic_numbers"],x["element_reference_energies_ev"])) + "}")
PY
)
  cutoff="${metadata[0]}"
  e0="${metadata[1]}"
  for channels in 36 48; do
    run_dir="${RUN_ROOT}/${dataset}/c${channels}"
    mkdir -p "${run_dir}"
    if [[ -f "${run_dir}/DONE" ]]; then
      mark "SKIP_${dataset}_c${channels}_DONE"
      continue
    fi
    mark "START_${dataset}_c${channels}"
    cd "${run_dir}"
    export NVIDIA_TF32_OVERRIDE=0
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
    if env TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=0 \
      "${TACE_BIN}/tace-train" \
        --config-path "$(dirname "${CONFIG}")" -cn "$(basename "${CONFIG}" .yaml)" \
        "misc.project_name=TECE_MS25_${dataset}_C${channels}_Seed${SEED}" \
        "misc.global_seed=${SEED}" misc.allow_tf32=false \
        "dataset.split_seed=${SEED}" \
        "dataset.train_file=${base}/extxyz/train.extxyz" \
        "dataset.valid_file=${base}/extxyz/val.extxyz" \
        dataset.keys.energy_key=energy dataset.keys.forces_key=forces \
        "dataset.train_dataloader.batch_size=${BATCH_SIZE}" \
        "dataset.valid_dataloader.batch_size=${BATCH_SIZE}" \
        "trainer.max_steps=${MAX_STEPS}" trainer.max_epochs=180 \
        "scheduler.T_max=${MAX_STEPS}" \
        "model.config.cutoff=${cutoff}" \
        "model.config.fidelity.0.name=MS25" \
        "model.config.fidelity.0.atomic_energy=${e0}" \
        "model.config.num_channel=${channels}" \
        model.config.mmax=2 model.config.Lmax=2 model.config.lmax=2 \
        model.config.num_layers=2 model.config.product_basis.correlation=3 \
        synth_metric.monitor_metric_name=val/forces_mae \
        callbacks.checkpoint_epoch.monitor=val/forces_mae \
        callbacks.checkpoint_epoch.save_top_k=-1 \
        '~callbacks.ema' >train.log 2>&1; then
      :
    else
      code=$?
      failures=$((failures + 1))
      mark "FAILED_${dataset}_c${channels}_TRAIN_exit${code}"
      continue
    fi

    best_ckpt="$("${TACE_BIN}/python" - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

versions=sorted(Path("lightning_logs").glob("version_*"),key=lambda p:int(p.name.split("_")[-1]))
if not versions: raise SystemExit("no TensorBoard validation log")
events=EventAccumulator(str(versions[-1])); events.Reload()
points=events.Scalars("val/forces_mae")
if not points: raise SystemExit("no val/forces_mae")
best=min(points,key=lambda p:(p.value,p.step))
checkpoint_step=best.step+1
candidates=sorted(Path("checkpoints_epoch").glob(f"TECE-*-{checkpoint_step}.ckpt"))
if len(candidates)!=1:
    raise SystemExit(f"expected one checkpoint for {checkpoint_step}, got {candidates}")
Path("selection.txt").write_text(
    f"{candidates[0]}\nval_forces_mae={best.value}\n"
    f"tensorboard_step={best.step}\ncheckpoint_step={checkpoint_step}\n"
)
print(candidates[0])
PY
)"
    if [[ -z "${best_ckpt}" ]]; then
      failures=$((failures + 1))
      mark "FAILED_${dataset}_c${channels}_SELECT"
      continue
    fi
    "${TACE_BIN}/tace-eval" \
      -i "${base}/extxyz/val.extxyz" -m "${best_ckpt}" -t 1 -e 1 \
      -b "${BATCH_SIZE}" --device cuda --dtype float32 --nl_backend matscipy \
      --energy_key energy --forces_key forces >val.log 2>&1
    "${TACE_BIN}/tace-eval" \
      -i "${base}/extxyz/test.extxyz" -m "${best_ckpt}" -t 1 -e 1 \
      -b "${BATCH_SIZE}" --device cuda --dtype float32 --nl_backend matscipy \
      --energy_key energy --forces_key forces >test.log 2>&1
    summary_extra=()
    if [[ "${dataset}" == "mgo-2x2" ]]; then
      while [[ ! -f /home/ylzhang/chorus_data/ms25_20260818/MGO4X4_TRANSFER_DONE ]]; do
        sleep 10
      done
      "${TACE_BIN}/tace-eval" \
        -i "${DATA_ROOT}/mgo-4x4-transfer/set1/extxyz/test.extxyz" \
        -m "${best_ckpt}" -t 1 -e 1 \
        -b "${BATCH_SIZE}" --device cuda --dtype float32 --nl_backend matscipy \
        --energy_key energy --forces_key forces >mgo4x4_test.log 2>&1
      summary_extra=(--size-transfer-log "${run_dir}/mgo4x4_test.log")
    fi
    "${TACE_BIN}/python" \
      "${REPO}/benchmarks/paper/scripts/solids/summarize_ms25_tece.py" \
      --run-dir "${run_dir}" "${summary_extra[@]}" \
      --output "${run_dir}/metrics.json"
    touch "${run_dir}/DONE"
    mark "DONE_${dataset}_c${channels}"
  done
done

if (( failures == 0 )); then
  touch "${RUN_ROOT}/DONE"
  mark "ALL_MS25_TECE_DONE"
else
  touch "${RUN_ROOT}/COMPLETED_WITH_FAILURES"
  mark "ALL_MS25_TECE_ATTEMPTED_failures${failures}"
  exit 1
fi
