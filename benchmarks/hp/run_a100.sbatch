#!/usr/bin/env bash
#SBATCH --job-name=chorus-o3-hp
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_o3_hp_20261017/logs/a100_%A_%a.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_o3_hp_20261017
SOURCE="$ROOT/source_v2/chorus"
cd "$ROOT"
export PYTHONPATH="$SOURCE:$ROOT/source/ictc_o3_hp"
export TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=1 MAX_JOBS=4
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME="$ROOT/cache/$SLURM_ARRAY_TASK_ID/xdg"
export FSCETP_ICTD_CACHE_DIR="$ROOT/cache/$SLURM_ARRAY_TASK_ID/ictd"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/$SLURM_ARRAY_TASK_ID/extensions"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/$SLURM_ARRAY_TASK_ID/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/$SLURM_ARRAY_TASK_ID/triton"
export PYTHONPYCACHEPREFIX="$ROOT/cache/$SLURM_ARRAY_TASK_ID/pycache"
export TMPDIR="$ROOT/cache/$SLURM_ARRAY_TASK_ID/tmp"
mkdir -p "$TMPDIR"
PYTHON=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
set -x
case "$SLURM_ARRAY_TASK_ID" in
  0)
    "$PYTHON" -m pytest -q "$SOURCE/chorus/test/test_hp_backend.py" --tb=short
    "$PYTHON" "$SOURCE/benchmarks/hp/qualify_and_bench.py" --scope final --channels 64 --lmax 2 --out results/a100_final_c64_l2.json
    ;;
  1)
    "$PYTHON" "$SOURCE/benchmarks/hp/qualify_and_bench.py" --scope persistent --channels 64 --lmax 2 --out results/a100_persistent_c64_l2.json
    ;;
  2)
    "$PYTHON" "$SOURCE/benchmarks/hp/qualify_and_bench.py" --scope persistent --channels 96 --lmax 3 --layers 3 --atoms 128 --out results/a100_persistent_c96_l3_d3.json
    ;;
esac
printf 'DONE\n'
