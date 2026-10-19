#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-selected-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/selected_report_v2_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
export PYTHONPYCACHEPREFIX="$ROOT/cache/report_pycache"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$ROOT/tools_v32/report_selected_training.py" \
  --root "$ROOT/results/selected_training_v2" --out "$ROOT/reports_selected_training_v2"
printf 'DONE\n'
