#!/usr/bin/env bash
#SBATCH --job-name=chorus-architecture-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/logs/report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$ROOT/report_tools/summarize_architecture.py" --root "$ROOT" \
  --result-set results_v2 results_large_v2 \
  --slurm-accounting-file "$ROOT/report_tools/slurm_accounting_snapshot.txt"
printf 'DONE\n'
