#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-scale-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/scale_report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
export PYTHONPYCACHEPREFIX="$ROOT/cache/report_pycache"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$ROOT/tools_v27/report_native_training_release.py" \
  --root "$ROOT/results/native_scale_expandable_v1" --sizes 4096 --out "$ROOT/reports_native_scale4096"
printf 'DONE\n'
