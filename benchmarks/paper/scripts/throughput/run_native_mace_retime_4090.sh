#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ylzhang/CHORUS-MLIP-attention-test"
OUT="/home/ylzhang/chorus_runs/cross_model_throughput_20260728"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
PY="/home/ylzhang/micromamba/envs/FSCETP/bin/python"
MACE="/home/ylzhang/chorus_runs/large_scale_main_20260724/baselines/native_mace_t1x/checkpoints/native_mace_t1x_run-20260616_epoch-24.pt"

cd "${REPO}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=20260728
export PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}"

"${PY}" "${BENCH}" --engine native-mace --checkpoint "${MACE}" \
  --sizes 32,64,128,256,512,1024,2048 \
  --output "${OUT}/native_mace.json" > "${OUT}/native_mace.log" 2>&1
