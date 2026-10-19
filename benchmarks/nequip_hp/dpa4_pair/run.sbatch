#!/usr/bin/env bash
#SBATCH --job-name=nequip-dpa4-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/nequip_chorus_dpa4_pair_20261018
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_dpa4_pair_20261018/logs/pair_%j.log
set -euo pipefail
ulimit -c 0
ROOT=/public/home/sps-xia/rxlin/nequip_chorus_dpa4_pair_20261018
export ROOT SOURCE="${SOURCE:?immutable source required}"
TOOLS="$SOURCE/tools"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
TORCH_SITE="$ASSETS/dpa4_torch211_overlay"
OEQ_SITE=/public/home/sps-xia/rxlin/venvs/nequip-chorus-oeq/lib/python3.11/site-packages
export TORCH_SITE OEQ_SITE
export NEQUIP_PY=/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python
export DPA_PY=/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python
export NEQUIP_PATH="$TOOLS/runtime_guard:$TOOLS:$TORCH_SITE:$SOURCE/nequip_root:$OEQ_SITE"
export DPA_PATH="$TOOLS/runtime_guard:$TOOLS:$TORCH_SITE:$SOURCE/dpa_root"
CUDA_LIBS=""
while IFS= read -r libdir; do CUDA_LIBS="${CUDA_LIBS:+$CUDA_LIBS:}$libdir"; done < <(
  find "$TORCH_SITE/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$TORCH_SITE/torch/lib:$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=8.0
export DP_INTERFACE_PREC=low DP_COMPILE_INFER=1 DP_TRITON_INFER=2 DP_TF32_INFER=0 DP_CUTE_INFER=0
CACHE="$ROOT/cache/${RUN:-${MODE:-prepare}}"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp" CUDA_CACHE_PATH="$CACHE/cuda_driver"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME/torch/kernels"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv
set -x
if [[ "${MODE:-prepare}" == prepare ]]; then
  PYTHONPATH="$NEQUIP_PATH" "$NEQUIP_PY" "$TOOLS/pair/prepare.py" nequip_table --root "$ROOT"
  PYTHONPATH="$DPA_PATH" "$DPA_PY" "$TOOLS/pair/prepare.py" dpa_pairs --root "$ROOT" --template "$SOURCE/dpa_template.json"
  PYTHONPATH="$NEQUIP_PATH" "$NEQUIP_PY" "$TOOLS/pair/prepare.py" save_nequip --root "$ROOT"
elif [[ "$MODE" == report ]]; then
  PYTHONPATH="$NEQUIP_PATH" "$NEQUIP_PY" "$TOOLS/pair/report.py" --root "$ROOT" --run "$RUN"
else
  PILOT=()
  if [[ "$MODE" == pilot ]]; then PILOT=(--pilot); fi
  PYTHONPATH="$NEQUIP_PATH" "$NEQUIP_PY" "$TOOLS/pair/driver.py" --root "$ROOT" --run "$RUN" \
    --case "${CASE:?case required}" --scope "${SCOPE:?scope required}" "${PILOT[@]}"
fi
printf 'DONE\n'
