#!/usr/bin/env bash
#SBATCH --job-name=phase-aligned-smoke
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/smoke/aligned-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/smoke/aligned-%j.err

set -euo pipefail
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-phase-causal-20260825}"
exec env REPO="${REPO}" \
  RUN_ROOT=/public/home/sps-xia/rxlin/chorus_phase_causal_20260825/smoke/aligned_runs \
  PRIMARY_ROOT=/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/runs \
  DATASET=mal SEED=20260616 MAX_STEPS=20 \
  bash "${REPO}/benchmarks/causal/phase_specific_20260825/run_aligned_one.sh"
