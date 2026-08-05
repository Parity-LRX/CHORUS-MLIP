#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
FSCETP="${FSCETP:-/home/ylzhang/micromamba/envs/FSCETP}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
TACE_REPO="${TACE_REPO:-/home/ylzhang/tace_chorus_benchmark}"
TACE_BIN="${TACE_BIN:-/home/ylzhang/tace_chorus_venv/bin}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA="${DATA:-/home/ylzhang/lrx/3bpa/standard_450_50_seed20260616_r5}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/3bpa_external_20260727}"
DEEP_MD="${DEEP_MD:-/home/ylzhang/lrx/3bpa/deepmd_standard_450_50_seed20260616_r5}"
WAIT_SCREEN="${WAIT_SCREEN:-3bpa_main}"
SEED="${SEED:-20260616}"
STEPS="${STEPS:-45000}"
EPOCHS="${EPOCHS:-1552}"
CONFIG="${CONFIG:-tace_transition1x_tece_s.yaml}"
MODELS="${MODELS:-dpa4 tece36 tece48 native_mace}"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}/driver_logs"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${ROOT}/status.log"
}

screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

mark "WAIT_SCREEN_${WAIT_SCREEN}"
while screen_exists "${WAIT_SCREEN}"; do
  sleep 30
done
mark "WAIT_COMPLETE_${WAIT_SCREEN}"

stage_failures=0
run_stage() {
  local name="$1"
  shift
  mark "START_${name}"
  if "$@" >"${ROOT}/driver_logs/${name}.log" 2>&1; then
    mark "DONE_${name}"
  else
    local code=$?
    stage_failures=$((stage_failures + 1))
    mark "FAILED_${name}_exit${code}"
  fi
}

should_run() {
  [[ " ${MODELS} " == *" $1 "* ]]
}

prepare_deepmd_temperature() {
  local temperature="$1"
  local staging="${DEEP_MD}/h5_${temperature}"
  local output="${DEEP_MD}/${temperature}"
  if [[ -f "${output}/DONE" ]]; then
    return
  fi
  mkdir -p "${staging}"
  ln -sfn "${DATA}/processed_train.h5" "${staging}/processed_train.h5"
  ln -sfn "${DATA}/processed_val.h5" "${staging}/processed_val.h5"
  ln -sfn "${DATA}/processed_test_${temperature}.h5" "${staging}/processed_test.h5"
  "${FSCETP}/bin/python" \
    "${REPO}/benchmarks/paper/scripts/training/prepare_grouped_deepmd_npy_from_ictc_h5.py" \
    --source-dir "${staging}" --output-dir "${output}" \
    --type-map 1,6,7,8 --splits train,val,test
}

if should_run dpa4; then
  run_stage dpa_prepare_300K prepare_deepmd_temperature 300K
  run_stage dpa_prepare_600K prepare_deepmd_temperature 600K
  run_stage dpa_prepare_1200K prepare_deepmd_temperature 1200K

  DPA_OUT="${ROOT}/dpa4_mini_c32_mix3"
  run_stage dpa4_train \
    env DATA="${DEEP_MD}/300K" OUT="${DPA_OUT}" STEPS="${STEPS}" \
      BATCH_SIZE=16 CHANNELS=32 MIXING_LAYERS=3 RCUT=5.0 SEED="${SEED}" \
      REPO="${REPO}" DPA_ENV="${DPA_ENV}" \
      bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_fixed_xxmd.sh"

  run_stage dpa4_select_and_test_300K \
    "${DPA_ENV}/bin/python" \
      "${REPO}/benchmarks/paper/scripts/training/evaluate_dpa4_xxmd_force_mae.py" \
      --run "${DPA_OUT}" --data "${DEEP_MD}/300K" \
      --out "${DPA_OUT}/full_val_force_mae_eval" --steps "${STEPS}" \
      --checkpoint-every 10

  if [[ -f "${DPA_OUT}/full_val_force_mae_eval/metrics.json" ]]; then
    selected_dpa="$("${DPA_ENV}/bin/python" - "${DPA_OUT}/full_val_force_mae_eval/metrics.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])
PY
)"
    for temperature in 600K 1200K; do
      run_stage "dpa4_test_${temperature}" \
        env NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
        "${DPA_ENV}/bin/dp" --pt test -m "${selected_dpa}" \
          -s "${DEEP_MD}/${temperature}/test" -n 0
    done
  fi
fi

run_tece() {
  local channels="$1"
  local run="${ROOT}/tece_c${channels}"
  if [[ -f "${run}/DONE" ]]; then
    return
  fi
  mkdir -p "${run}"
  cd "${run}" || return 2
  env TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=0 \
    NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
    "${TACE_BIN}/tace-train" \
      --config-path "${TACE_REPO}/example/train" -cn "${CONFIG}" \
      "misc.project_name=TECE_C${channels}_3BPA_Seed${SEED}" \
      "misc.global_seed=${SEED}" misc.allow_tf32=false \
      "dataset.split_seed=${SEED}" \
      "dataset.train_file=${DATA}/train.extxyz" \
      "dataset.valid_file=${DATA}/val.extxyz" \
      dataset.keys.energy_key=energy dataset.keys.forces_key=forces \
      dataset.train_dataloader.batch_size=16 \
      dataset.valid_dataloader.batch_size=16 \
      "trainer.max_steps=${STEPS}" "trainer.max_epochs=${EPOCHS}" \
      "scheduler.T_max=${STEPS}" \
      "model.config.fidelity.0.name=M06" \
      "model.config.fidelity.0.atomic_energy={1: -723.2941476475917, 6: -723.2941476475917, 7: -120.549024607932, 8: -60.27451230396598}" \
      "model.config.num_channel=${channels}" \
      model.config.mmax=2 model.config.Lmax=2 model.config.lmax=2 \
      model.config.num_layers=2 model.config.product_basis.correlation=3 \
      synth_metric.monitor_metric_name=val/forces_mae \
      callbacks.checkpoint_epoch.monitor=val/forces_mae \
      callbacks.checkpoint_epoch.save_top_k=-1 \
      '~callbacks.ema' >train.log 2>&1 || return $?

  local selected
  selected="$("${TACE_BIN}/python" - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

versions = sorted(
    Path("lightning_logs").glob("version_*"),
    key=lambda path: int(path.name.split("_")[-1]),
)
events = EventAccumulator(str(versions[-1]))
events.Reload()
points = events.Scalars("val/forces_mae")
best = min(points, key=lambda point: (point.value, point.step))
step = best.step + 1
candidates = sorted(Path("checkpoints_epoch").glob(f"TECE-*-{step}.ckpt"))
if len(candidates) != 1:
    raise SystemExit(f"checkpoint mismatch for step {step}: {candidates}")
Path("selection.json").write_text(
    __import__("json").dumps(
        {
            "selection_rule": "minimum 300K validation Force MAE",
            "tensorboard_step": best.step,
            "checkpoint_step": step,
            "validation_force_mae": best.value,
            "checkpoint": str(candidates[0]),
            "test_used_for_selection": False,
        },
        indent=2,
    )
    + "\n"
)
print(candidates[0])
PY
)" || return $?
  for temperature in 300K 600K 1200K; do
    "${TACE_BIN}/tace-eval" \
      -i "${DATA}/test_${temperature}.extxyz" \
      -m "${selected}" -t 1 -e 1 -b 16 \
      --device cuda --dtype float32 --nl_backend matscipy \
      --energy_key energy --forces_key forces \
      >"test_${temperature}.log" 2>&1 || return $?
  done
  touch DONE
}

if should_run tece36; then run_stage tece_c36 run_tece 36; fi
if should_run tece48; then run_stage tece_c48 run_tece 48; fi

run_native_mace() {
  local out="${ROOT}/native_mace_c128_l2_corr3"
  local name="native_mace_3bpa"
  local train_log="${out}/train.log"
  local recovery_marker="${out}/RECOVERED_CUEQ_EXPORT_FAILURE"

  recover_completed_cueq_run() {
    local checkpoint_count
    checkpoint_count="$(find "${out}/checkpoints" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -f "${train_log}" ]] \
      && grep -q 'Training complete' "${train_log}" \
      && grep -q 'ScriptFunction cannot be pickled' "${train_log}" \
      && (( checkpoint_count >= EPOCHS )); then
      printf 'RECOVERED_CUEQ_EXPORT_FAILURE %s checkpoints=%s expected=%s\n' \
        "$(date -Is)" "${checkpoint_count}" "${EPOCHS}" >"${recovery_marker}"
      mark "RECOVERED_NATIVE_MACE_3BPA_CUEQ_EXPORT_FAILURE_checkpoints${checkpoint_count}"
      return 0
    fi
    return 1
  }

  if [[ -f "${out}/DONE" ]]; then
    return
  fi
  mkdir -p "${out}"/{logs,models,checkpoints,results}
  cd "${out}" || return 2
  if ! recover_completed_cueq_run; then
    local exit_code=0
    env PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
      NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
      "${FSCETP}/bin/python" -m mace.cli.run_train \
      --name "${name}" --seed "${SEED}" \
      --device cuda --default_dtype float32 \
      --log_dir "${out}/logs" --model_dir "${out}/models" \
      --checkpoints_dir "${out}/checkpoints" --results_dir "${out}/results" \
      --model ScaleShiftMACE --r_max 5.0 \
      --radial_type bessel --num_radial_basis 8 --num_cutoff_basis 6 \
      --max_ell 2 --num_interactions 2 --correlation 3 --use_reduced_cg True \
      --num_channels 128 --max_L 2 \
      --hidden_irreps "128x0e + 128x1o + 128x2e" \
      --MLP_irreps "64x0e" --radial_MLP "[64, 64, 64]" \
      --interaction RealAgnosticResidualInteractionBlock \
      --interaction_first RealAgnosticResidualInteractionBlock \
      --train_file "${DATA}/train.extxyz" \
      --valid_file "${DATA}/val.extxyz" \
      --energy_key energy --forces_key forces \
      --atomic_numbers "[1, 6, 7, 8]" \
      --E0s "{1: -723.2941476475917, 6: -723.2941476475917, 7: -120.549024607932, 8: -60.27451230396598}" \
      --avg_num_neighbors 16.712427983539094 --scaling std_scaling \
      --loss weighted --energy_weight 1 --forces_weight 100 \
      --batch_size 16 --valid_batch_size 16 \
      --max_num_epochs "${EPOCHS}" \
      --lr 0.001 --weight_decay 5e-7 --optimizer adamw \
      --scheduler ExponentialLR --lr_scheduler_gamma 0.9955592568 --amsgrad \
      --num_workers 0 --compute_forces True --compute_stress False \
      --eval_interval 1 --keep_checkpoints --save_all_checkpoints \
      >"${train_log}" 2>&1 || exit_code=$?
    if (( exit_code != 0 )) && ! recover_completed_cueq_run; then
      return "${exit_code}"
    fi
  fi
  env PYTHONPATH="${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
    "${FSCETP}/bin/python" \
      "${REPO}/benchmarks/paper/scripts/training/evaluate_native_mace_3bpa.py" \
      --run-dir "${out}" --data-dir "${DATA}" --name "${name}" \
      --seed "${SEED}" --device cuda --out "${out}/validation_force_selected_eval" \
      >"${out}/eval.log" 2>&1 || return $?
  touch "${out}/DONE"
}

if should_run native_mace; then run_stage native_mace run_native_mace; fi

if (( stage_failures == 0 )); then
  touch "${ROOT}/DONE"
  mark "ALL_3BPA_EXTERNAL_DONE"
else
  touch "${ROOT}/COMPLETED_WITH_FAILURES"
  mark "ALL_3BPA_EXTERNAL_ATTEMPTED_failures${stage_failures}"
  exit 1
fi
