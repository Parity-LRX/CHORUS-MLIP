#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
OUT="${OUT:-/home/ylzhang/chorus_runs/cross_model_throughput_20260728}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
FSCETP="/home/ylzhang/micromamba/envs/FSCETP/bin/python"
DPA_PY="/home/ylzhang/venvs/dpa4-master/bin/python"
TECE_PY="/home/ylzhang/tace_chorus_venv/bin/python"
SIZES="${SIZES:-32,64,128,256,512,1024,2048}"

CHORUS="/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x/chorus_c128_l2_corr3_rank16_mae_ckpts_rerun/checkpoints/t1x_c128_l2_corr3_rank16_mae_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs32.e26s84348.pth"
ICTC="/home/ylzhang/chorus_runs/large_scale_main_20260724/t1x/ictc_c128_l2_corr3_phaseoff/checkpoints/t1x_c128_l2_corr3_phaseoff_ictc_bridge_u_makefx_seed20260616_epochs32.e19s62480.pth"
MACE="/home/ylzhang/chorus_runs/large_scale_main_20260724/baselines/native_mace_t1x/checkpoints/native_mace_t1x_run-20260616_epoch-24.pt"
DPA32_ROOT="/home/ylzhang/chorus_runs/large_scale_main_20260724/external/dpa4_t1x_large/c32_mix3"
DPA48_ROOT="/home/ylzhang/chorus_runs/dpa4_c48_scaling_20260727/t1x_c48_mix3"
TECE36="/home/ylzhang/tace_chorus_runs/large_missing_seed20260616/t1x/checkpoints_epoch/TECE-028-90625.ckpt"
TECE48="/home/ylzhang/tace_chorus_runs/large_tece_c48_seed20260616/t1x/checkpoints_epoch/TECE-028-90625.ckpt"

mkdir -p "${OUT}"
status="${OUT}/status.log"

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

run native_mace \
  env PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}" \
  "${FSCETP}" "${BENCH}" --engine native-mace --checkpoint "${MACE}" \
  --sizes "${SIZES}" --output "${OUT}/native_mace.json"

run mace_ictc_baseline \
  env PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}" \
  "${FSCETP}" "${BENCH}" --engine ictc-baseline --checkpoint "${ICTC}" \
  --sizes "${SIZES}" --output "${OUT}/mace_ictc_baseline.json"

run chorus_r16 \
  env PYTHONPATH="${REPO}:/tmp/mace_torch_0_3_16:${PYTHONPATH:-}" \
  "${FSCETP}" "${BENCH}" --engine chorus-r16 --checkpoint "${CHORUS}" \
  --sizes "${SIZES}" --output "${OUT}/chorus_r16.json"

run tece_c36 \
  env PYTHONPATH="/home/ylzhang/tace_chorus_benchmark:${PYTHONPATH:-}" \
  "${TECE_PY}" "${BENCH}" --engine tece --checkpoint "${TECE36}" \
  --sizes "${SIZES}" --output "${OUT}/tece_c36.json"

run tece_c48 \
  env PYTHONPATH="/home/ylzhang/tace_chorus_benchmark:${PYTHONPATH:-}" \
  "${TECE_PY}" "${BENCH}" --engine tece --checkpoint "${TECE48}" \
  --sizes "${SIZES}" --output "${OUT}/tece_c48.json"

if [[ ! -f "${OUT}/dpa4_torch212_rebuild.exit" ]] || [[ "$(cat "${OUT}/dpa4_torch212_rebuild.exit")" != "0" ]]; then
  echo "ERROR dpa4 torch2.12 rebuild incomplete" | tee -a "${status}"
  exit 2
fi

run dpa4_c32_compiled \
  "${DPA_PY}" "${BENCH}" --engine dpa4 \
  --config "${DPA32_ROOT}/input.json" \
  --checkpoint "${DPA32_ROOT}/ckpt_steps100000/model.ckpt-84375.pt" \
  --sizes "${SIZES}" --output "${OUT}/dpa4_c32_compiled.json"

run dpa4_c48_compiled \
  "${DPA_PY}" "${BENCH}" --engine dpa4 \
  --config "${DPA48_ROOT}/input.json" \
  --checkpoint "${DPA48_ROOT}/ckpt_steps100000/model.ckpt-90625.pt" \
  --sizes "${SIZES}" --output "${OUT}/dpa4_c48_compiled.json"

echo "ALL_OK $(date -Is)" | tee -a "${status}"
