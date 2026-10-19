#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-capacity-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/capacity_report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
export PYTHONPYCACHEPREFIX="$ROOT/cache/report_pycache"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$ROOT/tools_v25/report_training_capacity_screen.py" \
  --root "$ROOT/results/training_capacity_screen" --log "$ROOT/logs/training_screen_92263.log" \
  --out "$ROOT/reports_training_capacity_screen"
printf 'DONE\n'
