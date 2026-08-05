#!/usr/bin/env bash
#SBATCH --job-name=nequip_a100_scale
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs/nequip-%A_%a.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
NEQUIP_REPO="${NEQUIP_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS-OEQ-OPT}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
OEQ_SITE="${OEQ_SITE:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages}"
TORCH211_SITE="${TORCH211_SITE:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/dpa4_torch211_overlay}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/nequip_family}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_nequip_atom_scaling.py"
ROOT="/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_c84_r16_matched_oeq_20260803/transition1x"
SIZES="${SIZES:-128,256,512,1024,2048,4096}"

mkdir -p "${OUT}" /public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
[[ -f "${TORCH211_SITE}/torch/__init__.py" ]]
TORCH_LIB="${TORCH211_SITE}/torch/lib"
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}${libdir}"
done < <(find "${TORCH211_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) label="nequip_sh_c84"; config="${ROOT}/e3nn_chorus_off_oeq/train.yaml" ;;
  1) label="nequip_chorus_final_c84_r16"; config="${ROOT}/e3nn_chorus_on_oeq/train.yaml" ;;
  2) label="nequip_chorus_persistent_c84_r16"; config="${ROOT}/e3nn_chorus_on_persistent_oeq/train.yaml" ;;
  *) exit 2 ;;
esac

PYTHONPATH="${TORCH211_SITE}:${NEQUIP_REPO}:${REPO}:${OEQ_SITE}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${BENCH}" \
  --config "${config}" --label "${label}" --sizes "${SIZES}" \
  --require-openequivariance \
  --output "${OUT}/${label}.json"
touch "${OUT}/${label}.DONE"
