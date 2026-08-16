#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/home/ylzhang/CHORUS-MLIP}"
PYTHON="${PYTHON:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/lrx}"
RUN_ROOT="${RUN_ROOT:-/home/ylzhang/chorus_runs/nequip_lmax_controls_c64_r16_20260815/L3}"

export REPO OPERATOR_REPO PYTHON DATA_ROOT RUN_ROOT
export DATASETS="3bpa"
export CHANNELS="64"
export CHORUS_RANK="16"
export CHORUS_SCOPE="persistent"
export NUM_LAYERS="2"
export LMAX="3"
export SEED="20260616"
export EVAL_BATCH_SIZE="25"
export OPENEQUIVARIANCE="0"
export REUSE_COMPLETED_TRAIN="0"

exec bash "${REPO}/scripts/run_extended_variant.sh" e3nn on 0
