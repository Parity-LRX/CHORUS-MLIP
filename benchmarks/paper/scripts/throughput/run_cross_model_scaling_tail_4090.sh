#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
OUT="${OUT:-/home/ylzhang/chorus_runs/cross_model_throughput_20260728}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
FSCETP="/home/ylzhang/micromamba/envs/FSCETP/bin/python"
TECE_PY="/home/ylzhang/tace_chorus_venv/bin/python"
ALL_SIZES="32,64,128,256,512,1024,2048"
LARGE_SIZES="1024,2048"

R8="/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x/chorus_c128_l2_corr3_rank8_mae_ckpts/checkpoints/t1x_c128_l2_corr3_rank8_mae_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs32.e20s65604.pth"
R16="/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x/chorus_c128_l2_corr3_rank16_mae_ckpts_rerun/checkpoints/t1x_c128_l2_corr3_rank16_mae_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs32.e26s84348.pth"
R32="/home/ylzhang/chorus_runs/large_scale_main_20260724/rank32_pilot/t1x_c128_l2_corr3_rank32/checkpoints/t1x_c128_l2_corr3_rank32_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs32.e20s65604.pth"
ICTC="/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x/ictc_c128_l2_corr3_phaseoff/checkpoints/t1x_c128_l2_corr3_phaseoff_ictc_bridge_u_makefx_seed20260616_epochs32.e19s62480.pth"
MACE="/home/ylzhang/chorus_runs/large_scale_main_20260724/baselines/native_mace_t1x/checkpoints/native_mace_t1x_run-20260616_epoch-24.pt"
TECE36="/home/ylzhang/tace_chorus_runs/large_missing_seed20260616/t1x/checkpoints_epoch/TECE-028-90625.ckpt"
TECE48="/home/ylzhang/tace_chorus_runs/large_tece_c48_seed20260616/t1x/checkpoints_epoch/TECE-028-90625.ckpt"

mkdir -p "${OUT}"
status="${OUT}/tail_status.log"

run() {
  local name="$1"
  shift
  echo "START ${name} $(date -Is)" | tee -a "${status}"
  "$@" > "${OUT}/${name}.log" 2>&1
  echo "END ${name} $(date -Is)" | tee -a "${status}"
}

cd "${REPO}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=20260728

# R8 and R32 need the full curve. R16 32--512 already exists in the main run.
run chorus_r8 \
  env PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}" \
  "${FSCETP}" "${BENCH}" --engine chorus-r8 --checkpoint "${R8}" \
  --sizes "${ALL_SIZES}" --output "${OUT}/chorus_r8.json"

run chorus_r32 \
  env PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}" \
  "${FSCETP}" "${BENCH}" --engine chorus-r32 --checkpoint "${R32}" \
  --sizes "${ALL_SIZES}" --output "${OUT}/chorus_r32.json"

run chorus_r16_large \
  env PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}" \
  "${FSCETP}" "${BENCH}" --engine chorus-r16 --checkpoint "${R16}" \
  --sizes "${LARGE_SIZES}" --output "${OUT}/chorus_r16_large.json"

run native_mace_large \
  env PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}" \
  "${FSCETP}" "${BENCH}" --engine native-mace --checkpoint "${MACE}" \
  --sizes "${LARGE_SIZES}" --output "${OUT}/native_mace_large.json"

run mace_ictc_baseline_large \
  env PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}" \
  "${FSCETP}" "${BENCH}" --engine ictc-baseline --checkpoint "${ICTC}" \
  --sizes "${LARGE_SIZES}" --output "${OUT}/mace_ictc_baseline_large.json"

run tece_c36_large \
  env PYTHONPATH="/home/ylzhang/tace_chorus_benchmark:${PYTHONPATH:-}" \
  "${TECE_PY}" "${BENCH}" --engine tece --checkpoint "${TECE36}" \
  --sizes "${LARGE_SIZES}" --output "${OUT}/tece_c36_large.json"

run tece_c48_large \
  env PYTHONPATH="/home/ylzhang/tace_chorus_benchmark:${PYTHONPATH:-}" \
  "${TECE_PY}" "${BENCH}" --engine tece --checkpoint "${TECE48}" \
  --sizes "${LARGE_SIZES}" --output "${OUT}/tece_c48_large.json"

echo "ALL_OK $(date -Is)" | tee -a "${status}"
