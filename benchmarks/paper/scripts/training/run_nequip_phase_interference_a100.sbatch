#!/usr/bin/env bash
#SBATCH --job-name=nequip_phase_diag
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/nequip_phase_interference_20260804/slurm-%A_%a.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/NequIP-CHORUS}"
OPERATOR_REPO="${OPERATOR_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-operator}"
PYTHON="${PYTHON:-/public/home/sps-xia/rxlin/venvs/nequip-chorus-mff/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/nequip_phase_interference_20260804}"
DIAGNOSTIC="${OPERATOR_REPO}/benchmarks/paper/scripts/training/diagnose_nequip_phase_interference.py"

export PYTHONPATH="${REPO}:${OPERATOR_REPO}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=4
export NEQUIP_NUM_TASKS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_ROOT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    depth=2
    final_root="/public/home/sps-xia/rxlin/nequip_chorus_runs/depth2_scope_c64_r16_20260801"
    persistent_root="${final_root}"
    final_variant="e3nn_chorus_on"
    persistent_variant="e3nn_chorus_on_persistent"
    ;;
  1)
    depth=3
    final_root="/public/home/sps-xia/rxlin/nequip_chorus_runs/extended_c64_r16_20260729"
    persistent_root="/public/home/sps-xia/rxlin/nequip_chorus_runs/persistent_c64_r16_20260801"
    final_variant="e3nn_chorus_on"
    persistent_variant="e3nn_chorus_on_persistent"
    ;;
  *)
    echo "unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

datasets=(transition1x xxmd_mal xxmd_sti md22_buckyball 3bpa)
labels=("Transition1x-50k" "xxMD MAL" "xxMD STI" "MD22 Buckyball" "3BPA 300 K")

for scope in final persistent; do
  if [[ "${scope}" == "final" ]]; then
    root="${final_root}"
    variant="${final_variant}"
  else
    root="${persistent_root}"
    variant="${persistent_variant}"
  fi
  for index in "${!datasets[@]}"; do
    dataset="${datasets[$index]}"
    train_dir="${root}/${dataset}/${variant}/run"
    output_dir="${OUTPUT_ROOT}/depth${depth}/${scope}/${dataset}"
    mkdir -p "${output_dir}"
    if [[ -f "${output_dir}/DONE" ]]; then
      echo "SKIP depth=${depth} scope=${scope} dataset=${dataset}"
      continue
    fi
    if [[ ! -f "${train_dir}/best_model.pth" ]]; then
      echo "missing validation-selected checkpoint ${train_dir}/best_model.pth" >&2
      exit 3
    fi
    "${PYTHON}" "${DIAGNOSTIC}" \
      --train-dir "${train_dir}" \
      --dataset-label "${labels[$index]}" \
      --batch-size 4 \
      --max-frames 256 \
      --intervention-batches 8 \
      --bins 72 \
      --device cuda \
      --output-dir "${output_dir}" \
      >"${output_dir}/diagnostic.log" 2>&1
    touch "${output_dir}/DONE"
    echo "DONE depth=${depth} scope=${scope} dataset=${dataset}"
  done
done

touch "${OUTPUT_ROOT}/depth${depth}/DONE"
