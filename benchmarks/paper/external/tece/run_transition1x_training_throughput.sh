#!/usr/bin/env bash
set -euo pipefail

# Serial, matched-data training-throughput benchmark for the deployed CHORUS
# and TECE configurations. One complete Transition1x epoch gives enough
# post-warmup steps to separate steady-state throughput from startup/compile
# and validation overhead.

CHORUS_REPO="${CHORUS_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
CHORUS_RUNNER="${CHORUS_RUNNER:-${CHORUS_REPO}/benchmarks/paper/scripts/training/run_transition1x_chorus_enhanced.sh}"
TACE_REPO="${TACE_REPO:-/home/ylzhang/tace_chorus_benchmark}"
TACE_BIN="${TACE_BIN:-/home/ylzhang/tace_chorus_venv/bin}"
TACE_CONFIG="${TACE_CONFIG:-tace_transition1x_tece_s.yaml}"
OUT_ROOT="${OUT_ROOT:-/home/ylzhang/chorus_runs/transition1x_training_throughput_20260724}"
SEED="${SEED:-20260616}"
STEPS="${STEPS:-3125}"
RUN_CHORUS="${RUN_CHORUS:-1}"
TECE_MODES="${TECE_MODES:-eager,cue,aoti}"

mkdir -p "${OUT_ROOT}"
status="${OUT_ROOT}/status.log"

run_chorus() {
  local out="${OUT_ROOT}/chorus_makefx"
  echo "START chorus_makefx $(date -Is)" | tee -a "${status}"
  env \
    REPO="${CHORUS_REPO}" \
    OUT_ROOT="${out}" \
    MODE=full-nonlinear \
    RANK=8 \
    CHANNELS=48 \
    LMAX=2 \
    MAX_ELL=2 \
    CORRELATION=2 \
    PHASE_HEADS=1 \
    READOUT_HIDDEN_CHANNELS=64 \
    ELEMENT_ENERGY_CORRECTION=1 \
    SCALING=rms_forces_scaling \
    ATOMIC_INTER_SCALE=0.7642794490746259 \
    NO_ATOMIC_INTER_SHIFT=1 \
    MAX_STEPS="${STEPS}" \
    EPOCHS=1 \
    SEED="${SEED}" \
    bash "${CHORUS_RUNNER}"
  echo "END chorus_makefx $(date -Is)" | tee -a "${status}"
}

run_tece() {
  local mode="$1"
  local use_cue=0
  local use_compile=0
  case "${mode}" in
    eager) ;;
    cue) use_cue=1 ;;
    aoti) use_compile=1 ;;
    *) echo "unknown TECE mode: ${mode}" >&2; return 2 ;;
  esac

  local out="${OUT_ROOT}/tece_${mode}"
  mkdir -p "${out}"
  echo "START tece_${mode} $(date -Is)" | tee -a "${status}"
  (
    cd "${out}"
    /usr/bin/time -f 'WALL_SECONDS %e' \
      env \
        TACE_USE_OEQ=0 \
        TACE_USE_CUE="${use_cue}" \
        TACE_USE_EQT=0 \
        TACE_USE_COMPILE="${use_compile}" \
      "${TACE_BIN}/tace-train" \
        --config-path "${TACE_REPO}/example/train" \
        -cn "${TACE_CONFIG}" \
        "misc.project_name=TECE_S_Transition1x_Throughput_${mode}_${SEED}" \
        "misc.global_seed=${SEED}" \
        "dataset.split_seed=${SEED}" \
        "trainer.max_steps=${STEPS}" \
        trainer.max_epochs=1 \
        "scheduler.T_max=${STEPS}" \
        trainer.enable_model_summary=false \
        trainer.log_every_n_steps=20 \
        > train.log 2>&1
  )
  echo "END tece_${mode} $(date -Is)" | tee -a "${status}"
}

# Keep the benchmark serial: every row gets the full RTX 4090. The controls
# also permit resuming after an accelerator-specific smoke failure.
if [[ "${RUN_CHORUS}" == "1" ]]; then
  run_chorus
fi
IFS=',' read -r -a tece_modes <<<"${TECE_MODES}"
for tece_mode in "${tece_modes[@]}"; do
  run_tece "${tece_mode}"
done

echo "ALL_OK $(date -Is)" | tee -a "${status}"
