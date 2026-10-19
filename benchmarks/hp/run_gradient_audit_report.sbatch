#!/usr/bin/env bash
#SBATCH --job-name=chorus-gradient-audit-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/gradient_report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
export PYTHONPYCACHEPREFIX="$ROOT/cache/report_pycache"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$ROOT/tools_v8/summarize_training_gradient_audits.py" \
  --inputs "$ROOT/results/gradient_audit_v1/n128.json" \
    "$ROOT/results/gradient_audit_v1/n512.json" \
    "$ROOT/results/gradient_audit_literal_v2/n128.json" \
    "$ROOT/results/gradient_audit_literal_v2/n512.json" \
  --out "$ROOT/reports_gradient/gradient_audit_summary.json"
printf 'DONE\n'
