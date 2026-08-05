#!/usr/bin/env bash
#SBATCH --job-name=ictc_nequip_match
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/mace-ictc-nequip-matched-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
NEQUIP_REPO="${NEQUIP_REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS-OEQ-OPT}"
OUT="${OUT:-/public/home/sps-xia/rxlin/chorus_runs/mace_ictc_nequip_matched_${SLURM_JOB_ID}}"
SCRIPT="${REPO}/benchmarks/paper/scripts/throughput/bench_mace_ictc_nequip_matched.py"
NEQUIP_CONFIG="${NEQUIP_CONFIG:-/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_c84_r16_matched_oeq_20260803/xxmd_mal/e3nn_chorus_on_oeq/train.yaml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints}"
CHECKPOINT_OFF="${CHECKPOINT_OFF:-${CHECKPOINT_ROOT}/phase_off.pth}"
CHECKPOINT_FINAL="${CHECKPOINT_FINAL:-${CHECKPOINT_ROOT}/chorus_final_r16.pth}"
CHECKPOINT_PERSISTENT="${CHECKPOINT_PERSISTENT:-/public/home/sps-xia/rxlin/chorus_persistent_runs/large_c128_r16_20260730/mal_bucky/mal/checkpoints/mal_persistent_c128_l2_corr3_rank16_final.e45s40250.pth}"

mkdir -p "${OUT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH="${REPO}:${NEQUIP_REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${SCRIPT}" \
  --warmup 5 --iterations 20 --repeats 3 --output-dir "${OUT}" \
  --checkpoint-off "${CHECKPOINT_OFF}" \
  --checkpoint-final "${CHECKPOINT_FINAL}" \
  --checkpoint-persistent "${CHECKPOINT_PERSISTENT}" \
  --nequip-config "${NEQUIP_CONFIG}"
