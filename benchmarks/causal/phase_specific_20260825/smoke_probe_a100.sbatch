#!/usr/bin/env bash
#SBATCH --job-name=phase-probe-smoke
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/smoke/probe-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/smoke/probe-%j.err

set -euo pipefail
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-phase-causal-20260825}"
PY="/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python"
PRIMARY="/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/runs/mal/seed20260616/full"
OUT="/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/smoke/probe"
export PYTHONPATH="${REPO}:/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16"
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
"${PY}" "${REPO}/benchmarks/causal/phase_specific_20260825/evaluate_phase_interventions.py" \
  --checkpoint "${PRIMARY}/checkpoints/model.pth" \
  --data-dir /public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728/xxmd/processed_dft_temporal_r5/mal \
  --split test --elements H,C,O --dataset mal --seed 20260616 \
  --test-label smoke --batch-size 4 --permutation-repeats 2 --max-batches 1 \
  --device cuda --output-dir "${OUT}"
"${PY}" - "${OUT}/metrics.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert x["frames"] == 4
assert x["max_batches"] == 1
assert x["metrics"]["common_shift"]["maximum_absolute_force_change_ev_per_angstrom"] < 5e-5
assert x["metrics"]["aligned"]["force_component_count"] > 0
PY
