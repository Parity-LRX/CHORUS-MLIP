#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
DATA_ROOT="${DATA_ROOT:-/tmp/mace_ictd_public_md17}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/persistent_diagonal_small_20260802/rmd17}"

mkdir -p "${ROOT}"
exec env \
  PYTHON_BIN="${PYTHON_BIN}" \
  MACE_ICTC_REPO="${REPO}" \
  DATA_ROOT="${DATA_ROOT}" \
  DATASETS=revised_aspirin,revised_benzene,revised_ethanol \
  SEEDS=20260616 \
  MODES=ictc_phase_diagonal_full_l_all_layers_softplus_eager \
  EPOCHS=300 \
  BATCH_SIZE=16 \
  CHANNELS=64 \
  HIDDEN_LMAX=1 \
  MAX_ELL=2 \
  NUM_INTERACTIONS=2 \
  CORRELATION=2 \
  R_MAX=4.5 \
  PHASE_DENSITY_RANK=8 \
  LR_SCHEDULER=exp \
  LR_GAMMA=0.9993 \
  TRAIN_MAKEFX_COMPILE=0 \
  PARALLEL_JOBS=2 \
  NUM_WORKERS=1 \
  OUT_ROOT="${ROOT}" \
  bash "${REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
