#!/usr/bin/env bash
#SBATCH --job-name=chorus-force-ref-audit
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/force_reference_audit_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:?set immutable CHORUS snapshot}"
HP_SOURCE="${HP_SOURCE:?set immutable ICTC-O3-HP snapshot}"
CACHE="$ROOT/cache/native_v1_1"
export PYTHONPATH="$SOURCE:$HP_SOURCE" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$CACHE/tmp"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$SOURCE/benchmarks/hp/audit_force_reference.py" \
 --checkpoint /public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth \
 --reference-cache "$ROOT/cache/verification/final_n10000_k50_fp32_v61.pt" \
 --candidate "$ROOT/results/final_network_v62/chorus_final_dest1atomicfactorcoalescenetworkgeom_n10000_k50.qualification.pt" \
 ${AUDIT_FLAGS:-} \
 --out "$ROOT/results/${RESULT_SET:-force_reference_audit_v63}/audit.json"
printf 'DONE\n'
