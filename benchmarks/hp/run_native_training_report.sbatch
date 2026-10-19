#!/usr/bin/env bash
#SBATCH --job-name=chorus-native-training-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/native_report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
export PYTHONPYCACHEPREFIX="$ROOT/cache/report_pycache"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$ROOT/tools_v14/report_native_training_release.py" \
  --root "$ROOT/results/native_release_v1" --out "$ROOT/reports_native_release"
printf 'DONE\n'
