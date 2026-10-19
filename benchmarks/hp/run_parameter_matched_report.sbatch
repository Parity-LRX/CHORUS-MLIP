#!/usr/bin/env bash
#SBATCH --job-name=chorus-matched-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/matched_report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v79"
export PYTHONPATH="$TOOLS"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native/pycache"
date -Is
scontrol show job -d "$SLURM_JOB_ID"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/report_training_large_matrix.py" \
  --root "$ROOT" --result-set highl_parameter_matched_v5 \
  --cases l3 l3c192d3 l3c256d4 --target-speedup 1.5 \
  --case-result-set l3=highl_parameter_matched_v6 l3c192d3=highl_parameter_matched_v6 \
  --qualification-root "$ROOT/results/parameter_matched_oracle_v3" \
  --qualification-root "$ROOT/results/parameter_matched_oracle_v2" \
  --qualification-root "$ROOT/results/parameter_matched_oracle_v1" \
  --out "$ROOT/reports_parameter_matched_v8"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python "$TOOLS/report_training_highl_candidates.py" \
  --root "$ROOT" --out "$ROOT/reports_first_radial_candidates_v3"
printf 'DONE\n'
