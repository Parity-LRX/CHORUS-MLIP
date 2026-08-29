#!/usr/bin/env bash
#SBATCH --job-name=nequip-oeq-cost-ext
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%1
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/nequip_oeq_depth_lmax_scaling_extended_a100_20260818/logs/%A_%a.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
NEQUIP_REPO="${NEQUIP_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS-OEQ-OPT}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
OEQ_SITE="${OEQ_SITE:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages}"
TORCH211_SITE="${TORCH211_SITE:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/dpa4_torch211_overlay}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/nequip_oeq_depth_lmax_scaling_extended_a100_20260818/results}"
LOG_DIR="${LOG_DIR:-/public/home/sps-xia/rxlin/chorus_runs/nequip_oeq_depth_lmax_scaling_extended_a100_20260818/logs}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_nequip_atom_scaling.py"
L1_ROOT="/public/home/sps-xia/rxlin/nequip_chorus_runs/lmax_controls_c64_r16_20260815/L1/transition1x"
L2_ROOT="/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_scope_c64_r16_20260801/transition1x"
L3_ROOT="/public/home/sps-xia/rxlin/nequip_chorus_runs/lmax_controls_c64_r16_20260815/L3/transition1x"
C72_ROOT="/public/home/sps-xia/rxlin/nequip_chorus_runs/extended_c72_r16_20260729/transition1x"
SIZES="${SIZES:-128,256,512,1024,2048,4096}"

mkdir -p "${OUT}" "${LOG_DIR}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -f "${TORCH211_SITE}/torch/__init__.py" ]]
[[ -f "${OEQ_SITE}/openequivariance/__init__.py" ]]
TORCH_LIB="${TORCH211_SITE}/torch/lib"
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}${libdir}"
done < <(find "${TORCH211_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

extra_args=()
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    label="nequip_oeq_l1_off_c64"
    config="${L1_ROOT}/e3nn_chorus_off/train.yaml"
    ;;
  1)
    label="nequip_oeq_l1_final_c64_r16"
    config="${L1_ROOT}/e3nn_chorus_on/train.yaml"
    ;;
  2)
    label="nequip_oeq_l1_persistent_c64_r16"
    config="${L1_ROOT}/e3nn_chorus_on_persistent/train.yaml"
    ;;
  3)
    label="nequip_oeq_lmax3_final_c64_r16"
    config="${L3_ROOT}/e3nn_chorus_on/train.yaml"
    ;;
  4)
    label="nequip_oeq_lmax3_persistent_c64_r16"
    config="${L3_ROOT}/e3nn_chorus_on_persistent/train.yaml"
    ;;
  5)
    label="nequip_oeq_l2_depth3_final_c64_r16"
    config="${L2_ROOT}/e3nn_chorus_on/train.yaml"
    extra_args+=(--num-layers 3)
    ;;
  6)
    label="nequip_oeq_l2_depth3_persistent_c64_r16"
    config="${L2_ROOT}/e3nn_chorus_on_persistent/train.yaml"
    extra_args+=(--num-layers 3)
    ;;
  7)
    label="nequip_oeq_l2_depth3_widened_c72_off"
    config="${C72_ROOT}/e3nn_chorus_off/train.yaml"
    extra_args+=(--num-layers 3)
    ;;
  *)
    exit 2
    ;;
esac

[[ -f "${config}" ]]
PYTHONPATH="${TORCH211_SITE}:${NEQUIP_REPO}:${REPO}:${OEQ_SITE}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${BENCH}" \
  --config "${config}" \
  --label "${label}" \
  --sizes "${SIZES}" \
  --max-train-atoms 2048 \
  --enable-openequivariance \
  --require-openequivariance \
  --output "${OUT}/${label}.json" \
  "${extra_args[@]}"

touch "${OUT}/${label}.DONE"
