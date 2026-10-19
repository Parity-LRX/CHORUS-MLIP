#!/usr/bin/env bash
#SBATCH --job-name=nequip-native-dataflow
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/nequip_chorus_dataflow_20261018
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_dataflow_20261018/logs/round_%j.log
set -euo pipefail
ulimit -c 0
ROOT=/public/home/sps-xia/rxlin/nequip_chorus_dataflow_20261018
export ROOT SOURCE="${SOURCE:?immutable source required}"
TOOLS="$SOURCE/tools"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
TORCH_SITE="$ASSETS/dpa4_torch211_overlay"
OEQ_SITE=/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages
export NEQUIP_PY=/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python
export DPA_PY=/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python
BEFORE=/public/home/sps-xia/rxlin/nequip_chorus_dpa4_pair_20261018/source_v2/nequip_root
CHORUS_COMMON=/public/home/sps-xia/rxlin/nequip_chorus_oeq_native_20261018/source_v9/chorus
export NEQUIP_PATH="$TOOLS/runtime_guard:$TOOLS:$TORCH_SITE:$SOURCE/nequip_root:$CHORUS_COMMON:$OEQ_SITE"
export NEQUIP_BEFORE_PATH="$TOOLS/runtime_guard:$TOOLS:$TORCH_SITE:$BEFORE:$CHORUS_COMMON:$OEQ_SITE"
export DPA_PATH="$TOOLS/runtime_guard:$TOOLS:$TORCH_SITE:$SOURCE/dpa_root"
CUDA_LIBS=""
while IFS= read -r libdir; do CUDA_LIBS="${CUDA_LIBS:+$CUDA_LIBS:}$libdir"; done < <(
  find "$TORCH_SITE/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$TORCH_SITE/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0
export DP_INTERFACE_PREC=low DP_COMPILE_INFER=1 DP_TRITON_INFER=2 DP_TF32_INFER=0 DP_CUTE_INFER=0
CACHE="$ROOT/cache/${RUN:-${MODE:-baseline}}"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp" CUDA_CACHE_PATH="$CACHE/cuda_driver"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME/torch/kernels"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
if [[ "${MODE:-baseline}" == unit ]]; then
  PYTHONPATH="$NEQUIP_PATH" OUT="$ROOT/results/$RUN" "$NEQUIP_PY" "$TOOLS/run_unit.py"
elif [[ "$MODE" == qualify ]]; then
  PYTHONPATH="$NEQUIP_PATH" OUT="$ROOT/results/$RUN" QUAL_VARIANTS=optimized "$NEQUIP_PY" "$TOOLS/qualify_optimization.py"
elif [[ "$MODE" == report ]]; then
  PYTHONPATH="$NEQUIP_PATH" "$NEQUIP_PY" "$TOOLS/dataflow/report.py" --root "$ROOT" --run "$RUN"
else
  PYTHONPATH="$NEQUIP_PATH" "$NEQUIP_PY" "$TOOLS/dataflow/driver.py" --root "$ROOT" --run "$RUN" \
    --case "${CASE:?case required}" --scope "${SCOPE:?scope required}" --mode "$MODE"
fi
printf 'DONE\n'
