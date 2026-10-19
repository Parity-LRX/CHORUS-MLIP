#!/usr/bin/env bash
#SBATCH --job-name=chorus-large-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_large_20261017/logs/report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_large_20261017
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "${REPORT_SOURCE:-$ROOT/report_tools_v1}/summarize_capacity.py" --root "$ROOT" --out "$ROOT/reports/${REPORT_TAG:-capacity_v1}"
printf 'DONE\n'
