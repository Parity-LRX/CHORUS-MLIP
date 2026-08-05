#!/usr/bin/env bash
set -euo pipefail

VENV="/home/ylzhang/venvs/dpa4-master"
SOURCE="/home/ylzhang/external/deepmd-kit-dpa4-master"
REPO="/home/ylzhang/CHORUS-MLIP-attention-test"
OUT="/home/ylzhang/chorus_runs/cross_model_throughput_20260728"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
SIZES="32,64,128,256,512,1024,2048"

source "${VENV}/bin/activate"
if ! python - <<'PY'
import torch
raise SystemExit(not (
    torch.__version__.startswith("2.11.0") and torch.version.cuda == "12.6"
))
PY
then
  python -m pip install --force-reinstall \
    torch==2.11.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
fi

python - <<'PY'
import torch
assert torch.__version__.startswith("2.11.0")
assert torch.version.cuda == "12.6"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY

CUDA_HOME="/home/ylzhang/toolchains/cuda126"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  /home/ylzhang/bin/micromamba create -y -p "${CUDA_HOME}" \
    -c nvidia -c conda-forge cuda-nvcc=12.6.85
fi
if [[ ! -f "${CUDA_HOME}/targets/x86_64-linux/include/cublas_v2.h" ]]; then
  /home/ylzhang/bin/micromamba install -y -p "${CUDA_HOME}" \
    -c nvidia -c conda-forge cuda-libraries-dev=12.6
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_HOME
export CUDA_BIN_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export DP_ENABLE_TENSORFLOW=0
export DP_ENABLE_PYTORCH=1
export DP_VARIANT=cuda
export CMAKE_BUILD_PARALLEL_LEVEL=8
export CMAKE_ARGS="-DENABLE_TENSORFLOW=OFF -DUSE_TF_PYTHON_LIBS=OFF -DENABLE_PYTORCH=ON -DCMAKE_CUDA_COMPILER=${CUDACXX} -DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}"

cd "${SOURCE}"
python -m pip install --no-build-isolation --force-reinstall --no-deps \
  -C build-dir=/home/ylzhang/external/dpa4-master-build/torch211-cu126b .

cd "${REPO}"
python "${BENCH}" --engine dpa4 \
  --config /home/ylzhang/chorus_runs/large_scale_main_20260724/external/dpa4_t1x_large/c32_mix3/input.json \
  --checkpoint /home/ylzhang/chorus_runs/large_scale_main_20260724/external/dpa4_t1x_large/c32_mix3/ckpt_steps100000/model.ckpt-84375.pt \
  --sizes "${SIZES}" --output "${OUT}/dpa4_c32_compiled.json" \
  > "${OUT}/dpa4_c32_compiled.log" 2>&1
python "${BENCH}" --engine dpa4 \
  --config /home/ylzhang/chorus_runs/dpa4_c48_scaling_20260727/t1x_c48_mix3/input.json \
  --checkpoint /home/ylzhang/chorus_runs/dpa4_c48_scaling_20260727/t1x_c48_mix3/ckpt_steps100000/model.ckpt-90625.pt \
  --sizes "${SIZES}" --output "${OUT}/dpa4_c48_compiled.json" \
  > "${OUT}/dpa4_c48_compiled.log" 2>&1
