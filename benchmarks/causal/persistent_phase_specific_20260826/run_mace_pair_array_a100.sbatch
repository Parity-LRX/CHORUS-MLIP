#!/usr/bin/env bash
#SBATCH --job-name=mace-persist-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-19%1
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_persistent_causal_20260826/slurm/mace_pair_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_persistent_causal_20260826/slurm/mace_pair_%A_%a.err

set -euo pipefail
REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_persistent_causal_20260826}"
datasets=(mal bucky 3bpa zro)
seeds=(20260616 20260617 20260618 20260619 20260620)
dataset="${datasets[$((SLURM_ARRAY_TASK_ID / 5))]}"
seed="${seeds[$((SLURM_ARRAY_TASK_ID % 5))]}"
protocol="${REPO}/benchmarks/causal/persistent_phase_specific_20260826/protocol.json"
source_state="${REPO}/benchmarks/causal/persistent_phase_specific_20260826/SOURCE_STATE"
mkdir -p "${ROOT}/slurm"

env REPO="${REPO}" RUN_ROOT="${ROOT}/mace/runs" DATASET="${dataset}" \
  MODE=full SEED="${seed}" MAX_STEPS=45000 PHASE_SCOPE=persistent \
  PROTOCOL_PATH="${protocol}" SOURCE_STATE_PATH="${source_state}" \
  bash "${REPO}/benchmarks/causal/u1_hermitian_20260824/run_one.sh"

env REPO="${REPO}" RUN_ROOT="${ROOT}/mace/runs" \
  PRIMARY_ROOT="${ROOT}/mace/runs" DATASET="${dataset}" SEED="${seed}" \
  MAX_STEPS=45000 PHASE_SCOPE=persistent \
  PROTOCOL_PATH="${protocol}" SOURCE_STATE_PATH="${source_state}" \
  bash "${REPO}/benchmarks/causal/phase_specific_20260825/run_aligned_one.sh"

env REPO="${REPO}" PRIMARY_ROOT="${ROOT}/mace/runs" \
  PROBE_ROOT="${ROOT}/mace/probes" DATASET="${dataset}" SEED="${seed}" \
  PROTOCOL_PATH="${protocol}" SOURCE_STATE_PATH="${source_state}" \
  bash "${REPO}/benchmarks/causal/phase_specific_20260825/run_phase_probe_one.sh"

python3 - "${ROOT}/mace/runs/${dataset}/seed${seed}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
full = json.loads((root / "full" / "metrics.json").read_text())
aligned = json.loads((root / "aligned" / "metrics.json").read_text())
if full["parameter_count"] != aligned["parameter_count"]:
    raise SystemExit(
        f"parameter mismatch full={full['parameter_count']} aligned={aligned['parameter_count']}"
    )
(root / "PERSISTENT_PAIR_VERIFIED").write_text(
    f"parameter_count={full['parameter_count']}\n"
)
PY
