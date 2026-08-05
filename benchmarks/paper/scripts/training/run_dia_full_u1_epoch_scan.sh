#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/lrx/xxmd/processed_dft_temporal_r5}"
OUT_ROOT="${OUT_ROOT:-${REPO}/benchmarks/paper/results/phase/xxmd_dia_full_u1_epoch_scan_20260723}"
RUNNER="${REPO}/benchmarks/paper/scripts/training/run_xxmd_dft_screen_queued.sh"
EVALUATOR="${REPO}/benchmarks/paper/scripts/training/evaluate_xxmd_dft.sh"

WAIT_SCREENS="${WAIT_SCREENS:-chorus_t1x_rank16_ema}" \
MOLECULES=dia MODES=ictc_phase_full_l_softplus_makefx \
SEED=20260616 MAX_STEPS=150000 BATCH_SIZE=16 FORCE_WEIGHT=100 \
KEEP_CHECKPOINTS=194 SKIP_OFFICIAL_TEST=1 \
EMA_DECAY=0.999 EMA_START_STEP=0 \
MACE_ICTC_REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" \
DATA_ROOT="${DATA_ROOT}" OUT_ROOT="${OUT_ROOT}" \
bash "${RUNNER}"

train_root="${OUT_ROOT}/dia/train"
all_ckpts="${train_root}/checkpoints"
selected_ema="${OUT_ROOT}/dia/selected_checkpoints_ema"
selected_raw="${OUT_ROOT}/dia/selected_checkpoints_raw"
test_ema="${OUT_ROOT}/dia/official_test_scan_ema"
test_raw="${OUT_ROOT}/dia/official_test_scan_raw"
rm -rf "${selected_ema}" "${selected_raw}" "${test_ema}" "${test_raw}"
mkdir -p "${selected_ema}" "${selected_raw}" "${test_ema}" "${test_raw}"

for epoch in 90 100 110 120 130 140 150 160 170 180 190 193; do
  checkpoint="$(find "${all_ckpts}" -maxdepth 1 -name "*.e${epoch}s*.pth" -print -quit)"
  if [[ -n "${checkpoint}" ]]; then
    ln -s "${checkpoint}" "${selected_ema}/$(basename "${checkpoint}")"
  fi
done

"${PYTHON_BIN}" - "${selected_ema}" "${selected_raw}" <<'PY'
import pathlib
import sys
import torch

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
for path in sorted(source.glob("*.pth")):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["default_state_source"] = "raw"
    torch.save(checkpoint, target / path.name)
PY

# shellcheck disable=SC1090
source "${train_root}/metadata/dia.env"
avg_neighbors="$("${PYTHON_BIN}" -c \
  "import json; d=json.load(open('${DATA_ROOT}/dia/metadata.json')); print(d['splits']['train']['mean_directed_neighbors'])")"

DATA_DIR="${DATA_ROOT}/dia" CHECKPOINT_DIR="${selected_ema}" OUT_DIR="${test_ema}" \
AVG_NEIGHBORS="${avg_neighbors}" E0_KEYS="${E0_KEYS}" E0_VALUES="${E0_VALS}" \
BATCH_SIZE=16 FORCE_WEIGHT=100 PHASE_DENSITY_RANK=8 \
MACE_ICTC_REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" \
bash "${EVALUATOR}" >"${OUT_ROOT}/dia_test_scan_ema_driver.log" 2>&1

DATA_DIR="${DATA_ROOT}/dia" CHECKPOINT_DIR="${selected_raw}" OUT_DIR="${test_raw}" \
AVG_NEIGHBORS="${avg_neighbors}" E0_KEYS="${E0_KEYS}" E0_VALUES="${E0_VALS}" \
BATCH_SIZE=16 FORCE_WEIGHT=100 PHASE_DENSITY_RANK=8 \
MACE_ICTC_REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" \
bash "${EVALUATOR}" >"${OUT_ROOT}/dia_test_scan_raw_driver.log" 2>&1

echo "ALL_SCAN_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
