#!/usr/bin/env bash
#SBATCH --job-name=build-mfftorch-lmp
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/build_mfftorch_lmp_%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/build_mfftorch_lmp_%j.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
BUILD_ROOT="${BUILD_ROOT:-${ROOT}/deploy/mfftorch_lammps_20260924}"
TAG="${LAMMPS_TAG:-stable_22Jul2025}"
TARBALL="${BUILD_ROOT}/lammps-${TAG}.tar.gz"
SRC_PARENT="${BUILD_ROOT}/source"
DONE="${BUILD_ROOT}/BUILD_DONE"

mkdir -p "${BUILD_ROOT}" "${SRC_PARENT}"
rm -f "${DONE}"

# The CPU-only XHPC nodes do not expose the CUDA toolkit required by the CUDA
# PyTorch CMake package.  This build runs on the A100 host CPUs without asking
# Slurm for a GPU and discovers the node-local toolkit explicitly.
for cuda_root in /usr/local/cuda-12.8 /usr/local/cuda-12 /usr/local/cuda /usr/local/cuda-*; do
  if [[ -x "${cuda_root}/bin/nvcc" ]]; then
    export CUDAToolkit_ROOT="${cuda_root}"
    export CUDA_HOME="${cuda_root}"
    export CUDACXX="${cuda_root}/bin/nvcc"
    export PATH="${cuda_root}/bin:${PATH}"
    break
  fi
done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA nvcc is unavailable on $(hostname)" >&2
  exit 3
fi
nvcc --version

if [[ ! -s "${TARBALL}" ]] || ! tar tzf "${TARBALL}" >/dev/null 2>&1; then
  rm -f "${TARBALL}"
  "${PYTHON_BIN}" - "${TARBALL}" "${TAG}" <<'PY'
import pathlib
import sys
import urllib.request

target = pathlib.Path(sys.argv[1])
tag = sys.argv[2]
url = f"https://codeload.github.com/lammps/lammps/tar.gz/refs/tags/{tag}"
with urllib.request.urlopen(url, timeout=120) as source, target.open("wb") as sink:
    while chunk := source.read(8 * 1024 * 1024):
        sink.write(chunk)
PY
fi
tar tzf "${TARBALL}" >/dev/null

top="$(${PYTHON_BIN} - "${TARBALL}" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:gz") as archive:
    print(archive.getmembers()[0].name.split("/", 1)[0])
PY
)"
LAMMPS_SRC="${SRC_PARENT}/${top}"
if [[ ! -d "${LAMMPS_SRC}/cmake" ]]; then
  tar xzf "${TARBALL}" -C "${SRC_PARENT}"
fi

rm -rf "${LAMMPS_SRC}/src/USER-MFFTORCH"
cp -a "${REPO}/lammps_user_mfftorch/src/USER-MFFTORCH" "${LAMMPS_SRC}/src/"
mkdir -p "${LAMMPS_SRC}/cmake/Packages" "${LAMMPS_SRC}/cmake/Modules/Packages"
# Keep both LAMMPS CMake layouts on the same authoritative package definition.
cp "${REPO}/lammps_user_mfftorch/cmake/Packages/USER-MFFTORCH.cmake" \
  "${LAMMPS_SRC}/cmake/Packages/USER-MFFTORCH.cmake"
cp "${REPO}/lammps_user_mfftorch/cmake/Packages/USER-MFFTORCH.cmake" \
  "${LAMMPS_SRC}/cmake/Modules/Packages/USER-MFFTORCH.cmake"
"${PYTHON_BIN}" \
  "${REPO}/lammps_user_mfftorch/test_cudagraph_repro/patch_lammps_cmake.py" \
  "${LAMMPS_SRC}/cmake/CMakeLists.txt"

LIBTORCH_PREFIX="$(${PYTHON_BIN} -c 'import torch; print(torch.utils.cmake_prefix_path)')"
TORCH_LIB="$(${PYTHON_BIN} -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
NVIDIA_LIB_DIRS="$(${PYTHON_BIN} - <<'PY'
import pathlib
import site

root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(path) for path in sorted(root.glob("*/lib")) if path.is_dir()))
PY
)"
NCCL_LIB="$(${PYTHON_BIN} - <<'PY'
import pathlib
import site

path = pathlib.Path(site.getsitepackages()[0]) / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
if not path.is_file():
    raise SystemExit(f"missing NCCL library: {path}")
print(path)
PY
)"
MKL_INCLUDE_DIR="$(${PYTHON_BIN} -c 'import os, sys; print(os.path.join(sys.prefix, "include"))')"
[[ -d "${MKL_INCLUDE_DIR}" ]] || MKL_INCLUDE_DIR=/usr/include
export LD_LIBRARY_PATH="${TORCH_LIB}:${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"

BUILD_DIR="${BUILD_ROOT}/build-mfftorch"
rm -rf "${BUILD_DIR}"
cmake -S "${LAMMPS_SRC}/cmake" -B "${BUILD_DIR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=/usr/bin/g++ \
  -D CMAKE_CXX_FLAGS=-fpermissive \
  -D CMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed ${NCCL_LIB}" \
  -D CMAKE_PREFIX_PATH="${LIBTORCH_PREFIX}" \
  -D MKL_INCLUDE_DIR="${MKL_INCLUDE_DIR}" \
  -D BUILD_MPI=OFF \
  -D BUILD_OMP=OFF \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_ENABLE_VIRIAL=ON
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK:-32}"

"${BUILD_DIR}/lmp" -h > "${BUILD_ROOT}/lmp_help.txt"
grep -q "mff/torch" "${BUILD_ROOT}/lmp_help.txt"

{
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'lammps_tag=%s\n' "${TAG}"
  printf 'lammps_executable=%s\n' "${BUILD_DIR}/lmp"
  printf 'torch_version=%s\n' "$(${PYTHON_BIN} -c 'import torch; print(torch.__version__)')"
  printf 'torch_cmake_prefix=%s\n' "${LIBTORCH_PREFIX}"
  printf 'source_commit=%s\n' "$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || printf snapshot)"
  sha256sum \
    "${REPO}/lammps_user_mfftorch/src/USER-MFFTORCH/pair_mff_torch.cpp" \
    "${REPO}/lammps_user_mfftorch/src/USER-MFFTORCH/mff_torch_engine.cpp"
} > "${BUILD_ROOT}/build_manifest.txt"

touch "${DONE}"
echo "MFFTORCH_LAMMPS_BUILD_DONE executable=${BUILD_DIR}/lmp"
