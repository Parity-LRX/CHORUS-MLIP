#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
OUT="${OUT:-/home/ylzhang/chorus_runs/tece_openeq_throughput_20260728}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/tace_chorus_venv/bin/python}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
SIZES="${SIZES:-32,64,128,256,512,1024,2048}"
TECE36="/home/ylzhang/tace_chorus_runs/large_missing_seed20260616/t1x/checkpoints_epoch/TECE-028-90625.ckpt"
TECE48="/home/ylzhang/tace_chorus_runs/large_tece_c48_seed20260616/t1x/checkpoints_epoch/TECE-028-90625.ckpt"

mkdir -p "${OUT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CPATH="/home/ylzhang/micromamba/envs/FSCETP/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="/home/ylzhang/micromamba/envs/FSCETP/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="/home/ylzhang/micromamba/envs/FSCETP/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

run_one() {
  local label="$1"
  local checkpoint="$2"
  printf 'START %s %s\n' "${label}" "$(date -Is)" | tee -a "${OUT}/status.log"
  env PYTHONPATH="/home/ylzhang/tace_chorus_benchmark:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${BENCH}" --engine tece \
    --tece-backend openeq --checkpoint "${checkpoint}" \
    --sizes "${SIZES}" --output "${OUT}/${label}.json" \
    >"${OUT}/${label}.log" 2>&1
  printf 'DONE %s %s\n' "${label}" "$(date -Is)" | tee -a "${OUT}/status.log"
}

run_one tece_c36_openeq "${TECE36}"
run_one tece_c48_openeq "${TECE48}"
touch "${OUT}/DONE"
