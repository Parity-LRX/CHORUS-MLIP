#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_DIR="${DATA_DIR:-/home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x}"
OUT="${OUT:-${ROOT}/mae_selected_train_only_calibration}"
RAW_EVAL_ONLY="${RAW_EVAL_ONLY:-0}"
CHANNELS="${CHANNELS:-128}"
HIDDEN_LMAX="${HIDDEN_LMAX:-2}"
MAX_ELL="${MAX_ELL:-2}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
CORRELATION="${CORRELATION:-3}"
AVG_NEIGHBORS="${AVG_NEIGHBORS:-10.71685543435131}"
ATOMIC_ENERGY_KEYS="${ATOMIC_ENERGY_KEYS:-1,6,7,8}"
ATOMIC_ENERGY_VALUES="${ATOMIC_ENERGY_VALUES:--13.62222753701504,-1029.4130839658328,-1484.8710358098756,-2041.8396277138045}"
PHASE_DENSITY_RANK="${PHASE_DENSITY_RANK:-16}"
PHASE_DENSITY_PAIRS="${PHASE_DENSITY_PAIRS:-full-nonlinear}"
PHASE_CONTEXT="${PHASE_CONTEXT:-content}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CHORUS="${RUN_CHORUS:-1}"
BASELINE_NAME="${BASELINE_NAME:-baseline}"
CHORUS_NAME="${CHORUS_NAME:-chorus}"
CHORUS_SCOPE="${CHORUS_SCOPE:-final}"

BASELINE_SOURCE="${BASELINE_SOURCE:-${ROOT}/ictc_c128_l2_corr3_phaseoff/checkpoints/t1x_c128_l2_corr3_phaseoff_ictc_bridge_u_makefx_seed20260616_epochs32.e19s62480.pth}"
CHORUS_SOURCE="${CHORUS_SOURCE:-${ROOT}/chorus_c128_l2_corr3_rank16_mae_ckpts_rerun/checkpoints/t1x_c128_l2_corr3_rank16_mae_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs32.e26s84348.pth}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

mkdir -p "${OUT}"

common=(
  --data-dir "${DATA_DIR}" --train-prefix train
  --channels "${CHANNELS}" --lmax "${HIDDEN_LMAX}" --max-ell "${MAX_ELL}"
  --num-interaction "${NUM_INTERACTIONS}" --correlation "${CORRELATION}"
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg
  --first-layer-self-connection --mace-compatible-random-init
  --readout-hidden-channels 64
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6
  --max-radius 5.0 --avg-num-neighbors "${AVG_NEIGHBORS}"
  --atomic-energy-keys "${ATOMIC_ENERGY_KEYS}"
  --atomic-energy-values="${ATOMIC_ENERGY_VALUES}"
  --scaling std_scaling
  --batch-size 16 --dtype float32 --device cuda --num-workers 0
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0
  --optimizer adamw --optimizer-param-groups mace --weight-decay 5e-7
  --amsgrad --max-grad-norm 10
  --phase-hidden-channels 32 --phase-scale-init 0.05
  --phase-density-rank "${PHASE_DENSITY_RANK}"
  --seed 20260616 --log-interval 200
  --element-energy-correction
)

baseline_phase=(--phase-mode none)
chorus_phase=(
  --phase-mode final-full-l-residual
  --phase-amplitude softplus --phase-coefficient polar
  --phase-context "${PHASE_CONTEXT}" --phase-density-pairs "${PHASE_DENSITY_PAIRS}"
  --phase-normalization avg-neighbors
  --phase-placement pre-product-full-l --phase-scope "${CHORUS_SCOPE}"
)

augment_with_zero_correction() {
  local source="$1"
  local augmented="$2"
  "${PYTHON_BIN}" - "${source}" "${augmented}" <<'PY'
import hashlib
import pathlib
import sys

import torch

source = pathlib.Path(sys.argv[1])
augmented = pathlib.Path(sys.argv[2])
ckpt = torch.load(source, map_location="cpu", weights_only=False)
state = ckpt["e3trans_state_dict"]
if "element_energy_correction" in state:
    raise RuntimeError(f"{source} already contains element_energy_correction")

keys = ckpt["atomic_energy_keys"]
state["element_energy_correction"] = torch.zeros(
    int(keys.numel()), dtype=ckpt["atomic_energy_values"].dtype
)
ckpt["model_hyperparameters"]["ictd_fix_element_energy_correction"] = True
augmented.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, augmented)

source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
augmented_sha = hashlib.sha256(augmented.read_bytes()).hexdigest()
print(f"source_sha256={source_sha}")
print(f"augmented_sha256={augmented_sha}")
print("only_added_state_tensor=element_energy_correction")
print("element_energy_correction_init=zeros")
PY
}

run_one() {
  local name="$1"
  local source="$2"
  shift 2
  local phase_args=("$@")
  local model_dir="${OUT}/${name}"
  local augmented="${model_dir}/${name}.zero_correction_source.pth"
  local calibrated="${model_dir}/${name}.train_only_calibrated.pth"
  mkdir -p "${model_dir}"

  augment_with_zero_correction "${source}" "${augmented}" \
    >"${model_dir}/augmentation_audit.txt"

  {
    echo "model=${name}"
    echo "source_checkpoint=${source}"
    echo "checkpoint_selection=minimum validation Force MAE"
    echo "calibration_fit_split=train only"
    echo "energy_residual=one constant per element"
    echo "forces_changed=false"
  } >"${model_dir}/protocol.txt"

  for split in val test; do
    "${PYTHON_BIN}" -m chorus.cli.train \
      "${common[@]}" "${phase_args[@]}" \
      --val-prefix "${split}" --epochs 0 --max-steps 0 --eval-only \
      --resume-checkpoint "${augmented}" \
      --checkpoint "${model_dir}/${name}.raw_${split}.unused.pth" \
      >"${model_dir}/eval_raw_${split}.log" 2>&1
  done

  if (( RAW_EVAL_ONLY > 0 )); then
    return
  fi

  "${PYTHON_BIN}" -m chorus.cli.train \
    "${common[@]}" "${phase_args[@]}" \
    --val-prefix val --epochs 0 --max-steps 0 \
    --final-fit-element-energy-correction \
    --resume-checkpoint "${augmented}" --checkpoint "${calibrated}" \
    >"${model_dir}/calibrate_and_val.log" 2>&1

  for split in val test; do
    "${PYTHON_BIN}" -m chorus.cli.train \
      "${common[@]}" "${phase_args[@]}" \
      --val-prefix "${split}" --epochs 0 --max-steps 0 --eval-only \
      --resume-checkpoint "${calibrated}" \
      --checkpoint "${model_dir}/${name}.${split}.unused.pth" \
      >"${model_dir}/eval_${split}.log" 2>&1
  done

  touch "${model_dir}/DONE"
}

if (( RUN_BASELINE > 0 )); then
  run_one "${BASELINE_NAME}" "${BASELINE_SOURCE}" "${baseline_phase[@]}"
fi
if (( RUN_CHORUS > 0 )); then
  run_one "${CHORUS_NAME}" "${CHORUS_SOURCE}" "${chorus_phase[@]}"
fi
touch "${OUT}/ALL_DONE"
