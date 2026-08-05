#!/usr/bin/env bash
set -euo pipefail

# CHORUS mechanism comparison on the MD22 buckyball-catcher trajectory.
# The prepared 5-A graph data are reused without rewriting the source dataset.

MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/md22/bcc_proc50}"
OUT_ROOT="${OUT_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/md22_buckyball_chorus_20260720}"
DATASET_TAG="${DATASET_TAG:-buckyball_catcher}"
JOB_PREFIX="${JOB_PREFIX:-md22_${DATASET_TAG}}"

export PYTHONPATH="${MACE_ICTC_REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"

MODES="${MODES:-ictc_bridge_u_makefx,ictc_phase_diagonal_full_l_makefx,ictc_phase_full_l_gated_makefx}"
SEED="${SEED:-20260616}"
EPOCHS="${EPOCHS:-30}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CHANNELS="${CHANNELS:-64}"
HIDDEN_LMAX="${HIDDEN_LMAX:-1}"
MAX_ELL="${MAX_ELL:-2}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
CORRELATION="${CORRELATION:-2}"
AVG_NEIGHBORS="${AVG_NEIGHBORS:-30.3929}"
READOUT_HIDDEN="${READOUT_HIDDEN:-64}"
PHASE_DENSITY_RANK="${PHASE_DENSITY_RANK:-8}"
E0_KEYS="${E0_KEYS:-1,6}"
E0_VALUES="${E0_VALUES:--230.09857805,-986.13676308}"
MAKEFX_BUCKETS="${MAKEFX_BUCKETS:-4}"
MAKEFX_MAX_SLOTS="${MAKEFX_MAX_SLOTS:-8}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/checkpoints"

cat >"${OUT_ROOT}/protocol.json" <<EOF
{
  "dataset": "MD22 ${DATASET_TAG}",
  "data_dir": "${DATA_DIR}",
  "modes": "${MODES}",
  "seed": ${SEED},
  "epochs": ${EPOCHS},
  "max_steps": "${MAX_STEPS}",
  "batch_size": ${BATCH_SIZE},
  "channels": ${CHANNELS},
  "hidden_lmax": ${HIDDEN_LMAX},
  "max_ell": ${MAX_ELL},
  "num_interactions": ${NUM_INTERACTIONS},
  "correlation": ${CORRELATION},
  "readout_hidden_channels": ${READOUT_HIDDEN},
  "atomic_energy_keys": "${E0_KEYS}",
  "atomic_energy_values": "${E0_VALUES}",
  "phase_density_rank": ${PHASE_DENSITY_RANK},
  "keep_validation_checkpoints": ${KEEP_CHECKPOINTS},
  "radius_angstrom": 5.0,
  "scheduler": "optimizer-step cosine, 1e-3 to 1e-6",
  "execution": "required make_fx, strictly serial modes"
}
EOF

IFS=',' read -r -a MODE_ARRAY <<<"${MODES}"

common_args=(
  --data-dir "${DATA_DIR}" --train-prefix train --val-prefix val
  --channels "${CHANNELS}" --lmax "${HIDDEN_LMAX}" --max-ell "${MAX_ELL}"
  --num-interaction "${NUM_INTERACTIONS}" --correlation "${CORRELATION}"
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg
  --first-layer-self-connection --mace-compatible-random-init
  --readout-hidden-channels "${READOUT_HIDDEN}"
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6
  --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}"
  --atomic-energy-keys "${E0_KEYS}"
  --atomic-energy-values="${E0_VALUES}"
  --scaling std_scaling
  --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}"
  --dtype float32 --device cuda --num-workers 0
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0
  --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine
  --optimizer adamw --optimizer-param-groups mace --weight-decay 5e-7
  --amsgrad --max-grad-norm 10
  --phase-hidden-channels 32 --phase-scale-init 0.05
  --phase-density-rank "${PHASE_DENSITY_RANK}"
  --train-makefx-compile --require-train-makefx-compile
  --makefx-buckets "${MAKEFX_BUCKETS}" --makefx-max-slots "${MAKEFX_MAX_SLOTS}"
  --pad-nodes-to-max --pad-edges-to-max
  --seed "${SEED}" --log-interval 20
)

if [[ -n "${MAX_STEPS}" ]]; then
  common_args+=(--max-steps "${MAX_STEPS}")
fi
if (( KEEP_CHECKPOINTS > 0 )); then
  common_args+=(--keep-checkpoints "${KEEP_CHECKPOINTS}")
fi

for raw_mode in "${MODE_ARRAY[@]}"; do
  mode="$(echo "${raw_mode}" | xargs)"
  phase_args=(--phase-mode none)
  case "${mode}" in
    ictc_bridge_u_makefx)
      ;;
    ictc_phase_diagonal_full_l_makefx)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs diagonal
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    ictc_phase_diagonal_full_l_all_layers_softplus_makefx)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs diagonal
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope persistent
      )
      ;;
    ictc_phase_full_l_gated_makefx)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full-gated
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    ictc_phase_full_l_softplus_makefx)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    ictc_phase_full_l_persistent_softplus_makefx)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope persistent
      )
      ;;
    ictc_phase_full_l_nonlinear_makefx)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs full-nonlinear
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    ictc_phase_charge2_full_l_makefx)
      phase_args=(
        --phase-mode final-full-l-residual
        --phase-amplitude softplus
        --phase-coefficient polar
        --phase-context content
        --phase-density-pairs charge2
        --phase-normalization avg-neighbors
        --phase-placement pre-product-full-l
        --phase-scope final
      )
      ;;
    ictc_attention_makefx)
      phase_args=(
        --phase-mode none
        --attn-heads 4
        --attn-mode density-preserving
        --attn-scope final
      )
      ;;
    ictc_attention_legacy_makefx)
      phase_args=(
        --phase-mode none
        --attn-heads 4
        --attn-mode legacy-softmax
        --attn-scope all
      )
      ;;
    *)
      echo "unknown mode: ${mode}" >&2
      exit 2
      ;;
  esac

  job="${JOB_PREFIX}_${mode}_seed${SEED}_epochs${EPOCHS}"
  exec 9>"${OUT_ROOT}/${job}.run.lock"
  flock 9
  if [[ -f "${OUT_ROOT}/${job}.SKIP" ]]; then
    echo "SKIP_REQUESTED ${job}" | tee -a "${OUT_ROOT}/status.log"
    flock -u 9
    exec 9>&-
    continue
  fi
  if [[ -f "${OUT_ROOT}/${job}.DONE" ]]; then
    echo "REUSE_DONE ${job}" | tee -a "${OUT_ROOT}/status.log"
    flock -u 9
    exec 9>&-
    continue
  fi
  echo "START ${mode} $(date)" | tee -a "${OUT_ROOT}/status.log"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" -m chorus.cli.train \
    "${common_args[@]}" "${phase_args[@]}" \
    --checkpoint "${OUT_ROOT}/checkpoints/${job}.pth" \
    >"${OUT_ROOT}/logs/${job}.log" 2>&1
  echo "END ${mode} rc=0 $(date)" | tee -a "${OUT_ROOT}/status.log"
  touch "${OUT_ROOT}/${job}.DONE"
  flock -u 9
  exec 9>&-
done

echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
