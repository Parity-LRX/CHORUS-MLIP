#!/usr/bin/env bash
#SBATCH --job-name=dpa4_a100_scale
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs/dpa4-%A_%a.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
ASSETS="${ASSETS:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python}"
DPA211_SITE="${DPA211_SITE:-${ASSETS}/dpa4_torch211_overlay}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/dpa4_family}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
SIZES="${SIZES:-128,256,512,1024,2048,4096}"

mkdir -p "${OUT}" /public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1
[[ -f "${DPA211_SITE}/torch/__init__.py" ]]
[[ -f "${DPA211_SITE}/deepmd/__init__.py" ]]
TORCH_LIB="${DPA211_SITE}/torch/lib"
CUDA_LIBS=""
while IFS= read -r libdir; do
  CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}${libdir}"
done < <(find "${DPA211_SITE}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="${ASSETS}/mpi_libs:${TORCH_LIB}:${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) label="dpa4_c32_fp32_compiled"; root="${ASSETS}/dpa4/c32" ;;
  1) label="dpa4_c48_fp32_compiled"; root="${ASSETS}/dpa4/c48" ;;
  *) exit 2 ;;
esac

[[ -f "${PYTHON_BIN%/bin/python}/READY" ]]
PYTHONPATH="${DPA211_SITE}:${REPO}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" - <<'PY'
import torch
import deepmd

if not torch.__version__.startswith("2.11."):
    raise RuntimeError(f"DPA-4 compiled benchmark requires torch 2.11.x, found {torch.__version__}")
print(f"DPA-4 runtime: torch={torch.__version__}, deepmd={deepmd.__version__}")
PY
PYTHONPATH="${DPA211_SITE}:${REPO}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${BENCH}" --engine dpa4 --dpa-amp off \
  --config "${root}/input.json" --checkpoint "${root}/model.pt" \
  --sizes "${SIZES}" --output "${OUT}/${label}.json"
touch "${OUT}/${label}.DONE"
