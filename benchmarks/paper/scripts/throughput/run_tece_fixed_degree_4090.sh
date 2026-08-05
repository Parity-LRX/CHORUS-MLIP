#!/usr/bin/env bash
set -euo pipefail

REPO="/home/ylzhang/CHORUS-MLIP-attention-test"
OUT="/home/ylzhang/chorus_runs/cross_model_throughput_20260728"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
PY="/home/ylzhang/tace_chorus_venv/bin/python"
SIZES="32,64,128,256,512,1024,2048"
TECE36="/home/ylzhang/tace_chorus_runs/large_missing_seed20260616/t1x/checkpoints_epoch/TECE-028-90625.ckpt"
TECE48="/home/ylzhang/tace_chorus_runs/large_tece_c48_seed20260616/t1x/checkpoints_epoch/TECE-028-90625.ckpt"

cd "${REPO}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=20260728
export PYTHONPATH="/home/ylzhang/tace_chorus_benchmark:${PYTHONPATH:-}"

"${PY}" "${BENCH}" --engine tece --checkpoint "${TECE36}" \
  --sizes "${SIZES}" --output "${OUT}/tece_c36.json" \
  > "${OUT}/tece_c36.log" 2>&1
"${PY}" "${BENCH}" --engine tece --checkpoint "${TECE48}" \
  --sizes "${SIZES}" --output "${OUT}/tece_c48.json" \
  > "${OUT}/tece_c48.log" 2>&1
