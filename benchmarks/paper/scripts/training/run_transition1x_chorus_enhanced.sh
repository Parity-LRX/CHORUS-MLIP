#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616}"
MODE="${MODE:-full-nonlinear}"
RANK="${RANK:-8}"
CHANNELS="${CHANNELS:-64}"
LMAX="${LMAX:-1}"
MAX_ELL="${MAX_ELL:-2}"
CORRELATION="${CORRELATION:-2}"
PHASE_SCALE_INIT="${PHASE_SCALE_INIT:-0.05}"
PHASE_HEADS="${PHASE_HEADS:-1}"
NONLINEAR_LAYER_READOUTS="${NONLINEAR_LAYER_READOUTS:-0}"
FINAL_LAYER_READOUT_ONLY="${FINAL_LAYER_READOUT_ONLY:-0}"
READOUT_HIDDEN_CHANNELS="${READOUT_HIDDEN_CHANNELS:-64}"
ELEMENT_ENERGY_CORRECTION="${ELEMENT_ENERGY_CORRECTION:-0}"
FINAL_FIT_ELEMENT_ENERGY_CORRECTION="${FINAL_FIT_ELEMENT_ENERGY_CORRECTION:-0}"
SCALAR_FFN="${SCALAR_FFN:-0}"
SCALING="${SCALING:-std_scaling}"
NO_ATOMIC_INTER_SHIFT="${NO_ATOMIC_INTER_SHIFT:-0}"
ATOMIC_INTER_SCALE="${ATOMIC_INTER_SCALE:-}"
EMA_DECAY="${EMA_DECAY:-0.0}"
EMA_START_STEP="${EMA_START_STEP:-0}"
MAX_STEPS="${MAX_STEPS:-50000}"
EPOCHS="${EPOCHS:-17}"
SEED="${SEED:-20260616}"
OUT_ROOT="${OUT_ROOT:-${REPO}/benchmarks/paper/results/phase/transition1x_chorus_enhanced_20260723/${MODE}_rank${RANK}_steps${MAX_STEPS}_seed${SEED}}"

AVG_NEIGHBORS=10.71685543435131
E0_KEYS="1,6,7,8"
E0_VALUES="-13.62222753701504,-1029.4130839658328,-1484.8710358098756,-2041.8396277138045"
NAME="transition1x_ictc_phase_${MODE}_c${CHANNELS}_l${LMAX}_rank${RANK}_seed${SEED}_steps${MAX_STEPS}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}"/{logs,checkpoints}
ATOMIC_INTER_SCALE_JSON=null
if [[ -n "${ATOMIC_INTER_SCALE}" ]]; then
  ATOMIC_INTER_SCALE_JSON="${ATOMIC_INTER_SCALE}"
fi
cat >"${OUT_ROOT}/protocol.json" <<EOF
{"dataset":"Transition1x reaction-disjoint 50k/10k/10k","seed":${SEED},"optimizer_steps":${MAX_STEPS},"channels":${CHANNELS},"lmax":${LMAX},"max_ell":${MAX_ELL},"correlation":${CORRELATION},"phase_density_rank":${RANK},"phase_density_pairs":"${MODE}","phase_heads":${PHASE_HEADS},"nonlinear_layer_readouts":${NONLINEAR_LAYER_READOUTS},"final_layer_readout_only":${FINAL_LAYER_READOUT_ONLY},"readout_hidden_channels":${READOUT_HIDDEN_CHANNELS},"element_energy_correction":${ELEMENT_ENERGY_CORRECTION},"final_fit_element_energy_correction":${FINAL_FIT_ELEMENT_ENERGY_CORRECTION},"scalar_ffn":${SCALAR_FFN},"scaling":"${SCALING}","atomic_inter_scale":${ATOMIC_INTER_SCALE_JSON},"no_atomic_inter_shift":${NO_ATOMIC_INTER_SHIFT},"phase_scale_init":${PHASE_SCALE_INIT},"ema_decay":${EMA_DECAY},"ema_start_step":${EMA_START_STEP},"loss":"MSE energy:force=1:100","scheduler":"step cosine 1e-3 to 1e-6","batch_size":16,"makefx":true,"selection_split":"validation"}
EOF

checkpoint="${OUT_ROOT}/checkpoints/${NAME}.pth"
log="${OUT_ROOT}/logs/${NAME}.log"
READOUT_ARGS=()
if [[ "${NONLINEAR_LAYER_READOUTS}" == "1" ]]; then
  READOUT_ARGS+=(--nonlinear-layer-readouts)
fi
if [[ "${FINAL_LAYER_READOUT_ONLY}" == "1" ]]; then
  READOUT_ARGS+=(--final-layer-readout-only)
fi
if [[ "${ELEMENT_ENERGY_CORRECTION}" == "1" ]]; then
  READOUT_ARGS+=(--element-energy-correction)
fi
if [[ "${FINAL_FIT_ELEMENT_ENERGY_CORRECTION}" == "1" ]]; then
  READOUT_ARGS+=(--final-fit-element-energy-correction)
fi
if [[ "${SCALAR_FFN}" == "1" ]]; then
  READOUT_ARGS+=(--scalar-ffn)
fi
SCALING_ARGS=(--scaling "${SCALING}")
if [[ -n "${ATOMIC_INTER_SCALE}" ]]; then
  SCALING_ARGS+=(--atomic-inter-scale "${ATOMIC_INTER_SCALE}")
fi
if [[ "${NO_ATOMIC_INTER_SHIFT}" == "1" ]]; then
  SCALING_ARGS+=(--no-atomic-inter-shift)
fi
echo "START ${NAME} $(date)" | tee -a "${OUT_ROOT}/status.log"
/usr/bin/time -f 'WALL_SECONDS %e' "${PYTHON_BIN}" -m chorus.cli.train \
  --data-dir "${DATA_DIR}" --train-prefix train --val-prefix val \
  --channels "${CHANNELS}" --lmax "${LMAX}" --max-ell "${MAX_ELL}" \
  --num-interaction 2 --correlation "${CORRELATION}" \
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
  --first-layer-self-connection --mace-compatible-random-init \
  --readout-hidden-channels "${READOUT_HIDDEN_CHANNELS}" \
  "${READOUT_ARGS[@]}" \
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
  --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}" \
  --atomic-energy-keys "${E0_KEYS}" --atomic-energy-values="${E0_VALUES}" \
  "${SCALING_ARGS[@]}" --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}" \
  --batch-size 16 --dtype float32 --device cuda --num-workers 2 \
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
  --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
  --optimizer adamw --optimizer-param-groups mace --weight-decay 5e-7 \
  --amsgrad --max-grad-norm 10.0 \
  --phase-hidden-channels 32 --phase-heads "${PHASE_HEADS}" \
  --phase-scale-init "${PHASE_SCALE_INIT}" \
  --phase-density-rank "${RANK}" \
  --ema-decay "${EMA_DECAY}" --ema-start-step "${EMA_START_STEP}" \
  --train-makefx-compile --require-train-makefx-compile \
  --makefx-buckets 4 --makefx-max-slots 8 --pad-nodes-to-max --pad-edges-to-max \
  --seed "${SEED}" --log-interval 200 --checkpoint "${checkpoint}" \
  --keep-checkpoints 1 \
  --phase-mode final-full-l-residual --phase-amplitude softplus \
  --phase-coefficient polar --phase-context content \
  --phase-density-pairs "${MODE}" --phase-normalization avg-neighbors \
  --phase-placement pre-product-full-l --phase-scope final >"${log}" 2>&1
echo "ALL_OK ${NAME} $(date)" | tee -a "${OUT_ROOT}/status.log"
