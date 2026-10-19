#!/usr/bin/env bash
#SBATCH --job-name=chorus-hp-release
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/chorus_hp_release_20261019
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_release_20261019/logs/release_%j.log
set -euo pipefail
export ROOT=/public/home/sps-xia/rxlin/chorus_hp_release_20261019
export SOURCE_ROOT="$ROOT/${SOURCE_VERSION:-source_v3}"
export WHEELHOUSE="$ROOT/wheels_source_v3"
SOURCE="$SOURCE_ROOT/chorus"
HP_SOURCE="$SOURCE_ROOT/ictc_o3_hp"
TOOLS="$SOURCE/benchmarks/hp/release"
BASE=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export XDG_CACHE_HOME="$ROOT/cache/xdg" TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/extensions_source_v3" FSCETP_ICTD_CACHE_DIR="$ROOT/cache/ictd"
export PYTHONPYCACHEPREFIX="$ROOT/cache/pycache" CUDA_CACHE_PATH="$ROOT/cache/cuda"
export PIP_CACHE_DIR="$ROOT/cache/pip" TMPDIR="$ROOT/cache/tmp"
mkdir -p "$TMPDIR" "$WHEELHOUSE" "$ROOT/sdists_source_v3" "$ROOT/results" "$CUDA_CACHE_PATH"
unset PYTHONPATH
date -Is
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.used,memory.total --format=csv
set -x
"$BASE" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$WHEELHOUSE" "$HP_SOURCE" "$SOURCE"
for package in "$HP_SOURCE" "$SOURCE"; do
  ( cd "$package"; "$BASE" -c 'import os,setuptools.build_meta as b; b.build_sdist(os.environ["ROOT"]+"/sdists_source_v3")' )
done
"$BASE" "$TOOLS/check_wheels.py"
# Ordinary build tooling is staged separately; neither shared environment is changed.
"$BASE" -m pip download --only-binary=:all: --no-deps --dest "$WHEELHOUSE" ninja
"$BASE" -m venv --system-site-packages --without-pip "$ROOT/env_reference_source_v3"
"$BASE" -m venv --system-site-packages --without-pip "$ROOT/env_hp_source_v3"
REF="$ROOT/env_reference_source_v3/bin/python"
HP="$ROOT/env_hp_source_v3/bin/python"
export HP_PYTHON="$HP" GIT_TERMINAL_PROMPT=0
"$REF" -m pip install --no-index --find-links "$WHEELHOUSE" --report "$ROOT/results/install_reference.json" 'chorus-mlip==0.1.0'
PYTHONPATH="$ROOT/reference_before" "$REF" "$TOOLS/check_reference.py" --write --oracle "$ROOT/results/reference_before.pth"
"$REF" "$TOOLS/check_reference.py" --oracle "$ROOT/results/reference_before.pth" --report "$ROOT/results/reference_compatibility.json"
"$HP" -m pip install --no-index --no-build-isolation --find-links "$WHEELHOUSE" --report "$ROOT/results/install_hp.json" 'chorus-mlip[hp]==0.1.0'
if "$HP" -m pip install --dry-run --no-index --no-build-isolation --find-links "$WHEELHOUSE" \
    'chorus-mlip[hp]==0.1.0' 'ictc-o3-hp!=0.2.12' > "$ROOT/results/incompatible_resolver.log" 2>&1; then
  printf 'ERROR: incompatible HP requirement was accepted\n'; exit 1
fi
"$BASE" "$TOOLS/check_install.py"
PYTHONPATH="$SOURCE/benchmarks/hp" "$HP" "$TOOLS/check_native.py"
printf 'RELEASE_QUALIFICATION_DONE\n'
