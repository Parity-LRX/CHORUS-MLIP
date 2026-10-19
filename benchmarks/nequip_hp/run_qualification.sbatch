#!/usr/bin/env bash
#SBATCH --job-name=nequip-chorus-oeq-qualify
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/nequip_chorus_oeq_native_20261018
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_oeq_native_20261018/logs/qualification_%j.log
set -euo pipefail
ulimit -c 0
ROOT=/public/home/sps-xia/rxlin/nequip_chorus_oeq_native_20261018
cd "$ROOT"
export SOURCE="${SOURCE:?immutable source required}"
export RUN="${RUN:-baseline_v1}"
export OUT="$ROOT/results/$RUN"
TOOLS="$SOURCE/tools"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
TORCH_SITE="$ASSETS/dpa4_torch211_overlay"
OEQ_SITE=/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages
CUDA_LIBS=""
while IFS= read -r libdir; do CUDA_LIBS="${CUDA_LIBS:+$CUDA_LIBS:}$libdir"; done < <(
  find "$TORCH_SITE/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$TORCH_SITE/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TOOLS/runtime_guard:$TOOLS:$TORCH_SITE:$SOURCE/nequip_root:$SOURCE/chorus:$SOURCE/chorus/benchmarks/hp:$OEQ_SITE"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0
CACHE="$ROOT/cache/$RUN"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp" CUDA_CACHE_PATH="$CACHE/cuda_driver"
mkdir -p "$OUT" "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME/torch/kernels"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
set -x
PY=/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python
if [[ "${MODE:-environment}" == environment ]]; then
  "$PY" "$TOOLS/qualify_environment.py"
elif [[ "$MODE" == benchmark ]]; then
  "$PY" "$TOOLS/benchmark.py"
elif [[ "$MODE" == unit ]]; then
  "$PY" "$TOOLS/run_unit.py"
elif [[ "$MODE" == report ]]; then
  "$PY" "$TOOLS/report.py"
else
  "$PY" "$TOOLS/qualify_optimization.py"
fi
printf 'DONE\n'
