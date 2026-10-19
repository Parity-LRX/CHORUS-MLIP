#!/usr/bin/env bash
#SBATCH --job-name=chorus-radial-first
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/radial_first_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="${TOOLS:-$ROOT/tools_v64}"
export SOURCE="$ROOT/source_v40/chorus" HP_SOURCE="$ROOT/source_v40/ictc_o3_hp"
CACHE="$ROOT/cache/native"
export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE"
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
if [[ "${SKIP_RAW:-0}" != 1 ]]; then
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python -m pytest -q \
  "$SOURCE/chorus/test/test_hp_backend.py" -k live_radial_first_projected_targets \
  --maxfail=1 --junitxml="$ROOT/results/radial_first_${SLURM_JOB_ID}.xml"
fi
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_mixed_sweep.py" \
  --root "$ROOT" --tools "$TOOLS" --case l3c256d4 --atoms 128 \
  --result-set radial_first_qual_v14 --qualify --repeats 1 --stop-on-error \
  --scopes persistent final --variants radial_joint_32k radial_bound_8k radial_bound_32k radial_bound_64k
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/run_training_mixed_sweep.py" \
  --root "$ROOT" --tools "$TOOLS" --case l3c256d4 --atoms 1024 \
  --result-set radial_first_screen_v14 --repeats 16 --profile \
  --scopes persistent final --variants radial_joint_32k radial_bound_8k radial_bound_32k radial_bound_64k
printf 'DONE\n'
