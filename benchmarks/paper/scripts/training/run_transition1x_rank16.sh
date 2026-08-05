#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616}"
OUT_ROOT="${OUT_ROOT:-${REPO}/benchmarks/paper/results/phase/transition1x_rank16_steps100000_20260723}"
EVALUATOR="${REPO}/benchmarks/paper/scripts/training/evaluate_xxmd_dft.sh"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"
EMA_DECAY="${EMA_DECAY:-0.0}"
EMA_START_STEP="${EMA_START_STEP:-0}"
EVALUATE_EMA_VALIDATION="${EVALUATE_EMA_VALIDATION:-0}"

SEED=20260616
MAX_STEPS=100000
EPOCHS=32
RANK=16
AVG_NEIGHBORS=10.71685543435131
E0_KEYS="1,6,7,8"
E0_VALUES="-13.62222753701504,-1029.4130839658328,-1484.8710358098756,-2041.8396277138045"
NAME="transition1x_ictc_phase_full_l_softplus_makefx_rank16_seed${SEED}_steps${MAX_STEPS}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}"/{logs,checkpoints,official_test}
cat >"${OUT_ROOT}/protocol.json" <<EOF
{"dataset":"Transition1x reaction-disjoint 50k/10k/10k","seed":${SEED},"optimizer_steps":${MAX_STEPS},"phase_density_rank":${RANK},"reference_rank":8,"change":"phase-density-rank only","loss":"MSE energy:force=1:100","scheduler":"step cosine 1e-3 to 1e-6","makefx":true,"keep_validation_checkpoints":${KEEP_CHECKPOINTS},"ema_decay":${EMA_DECAY},"ema_start_step":${EMA_START_STEP},"evaluate_ema_validation":${EVALUATE_EMA_VALIDATION}}
EOF

checkpoint="${OUT_ROOT}/checkpoints/${NAME}.pth"
log="${OUT_ROOT}/logs/${NAME}.log"
echo "START ${NAME} $(date)" | tee -a "${OUT_ROOT}/status.log"
/usr/bin/time -f 'WALL_SECONDS %e' "${PYTHON_BIN}" -m chorus.cli.train \
  --data-dir "${DATA_DIR}" --train-prefix train --val-prefix val \
  --channels 64 --lmax 1 --max-ell 2 --num-interaction 2 --correlation 2 \
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
  --first-layer-self-connection --mace-compatible-random-init \
  --readout-hidden-channels 64 \
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
  --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}" \
  --atomic-energy-keys "${E0_KEYS}" --atomic-energy-values="${E0_VALUES}" \
  --scaling std_scaling --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}" \
  --batch-size 16 --dtype float32 --device cuda --num-workers 2 \
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
  --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine --lr-scheduler-gamma 0.9993 \
  --optimizer adamw --optimizer-param-groups mace --weight-decay 5e-7 \
  --amsgrad --max-grad-norm 10.0 \
  --ema-decay "${EMA_DECAY}" --ema-start-step "${EMA_START_STEP}" \
  --phase-hidden-channels 32 --phase-scale-init 0.05 --phase-density-rank "${RANK}" \
  --train-makefx-compile --require-train-makefx-compile \
  --makefx-buckets 4 --makefx-max-slots 8 --pad-nodes-to-max --pad-edges-to-max \
  --seed "${SEED}" --log-interval 200 --checkpoint "${checkpoint}" \
  --keep-checkpoints "${KEEP_CHECKPOINTS}" \
  --phase-mode final-full-l-residual --phase-amplitude softplus \
  --phase-coefficient polar --phase-context content --phase-density-pairs full \
  --phase-normalization avg-neighbors --phase-placement pre-product-full-l \
  --phase-scope final >"${log}" 2>&1
echo "END_TRAIN ${NAME} $(date)" | tee -a "${OUT_ROOT}/status.log"

if [[ "${EVALUATE_EMA_VALIDATION}" == "1" ]]; then
  DATA_DIR="${DATA_DIR}" CHECKPOINT_DIR="${OUT_ROOT}/checkpoints" \
  OUT_DIR="${OUT_ROOT}/ema_validation" EVAL_PREFIX=val AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
  E0_KEYS="${E0_KEYS}" E0_VALUES="${E0_VALUES}" BATCH_SIZE=16 FORCE_WEIGHT=100 \
  PHASE_DENSITY_RANK="${RANK}" MACE_ICTC_REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" \
  bash "${EVALUATOR}" >"${OUT_ROOT}/ema_validation_driver.log" 2>&1
  echo "END_EMA_VALIDATION ${NAME} $(date)" | tee -a "${OUT_ROOT}/status.log"
fi

DATA_DIR="${DATA_DIR}" CHECKPOINT_DIR="${OUT_ROOT}/checkpoints" \
OUT_DIR="${OUT_ROOT}/official_test" AVG_NEIGHBORS="${AVG_NEIGHBORS}" \
E0_KEYS="${E0_KEYS}" E0_VALUES="${E0_VALUES}" BATCH_SIZE=16 FORCE_WEIGHT=100 \
PHASE_DENSITY_RANK="${RANK}" MACE_ICTC_REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" \
bash "${EVALUATOR}" >"${OUT_ROOT}/official_test_driver.log" 2>&1
echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
