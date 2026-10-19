#!/usr/bin/env bash
#SBATCH --job-name=chorus-native-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018/logs/native_report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
SOURCE="${SOURCE:?immutable CHORUS source required}"
export PYTHONPYCACHEPREFIX="$ROOT/cache/native_report_pycache"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/summarize_inference_native_nograph.py" \
  --source "$ROOT/report_combined_v15/inference_summary.json" --out "$ROOT/native_report_v16"
