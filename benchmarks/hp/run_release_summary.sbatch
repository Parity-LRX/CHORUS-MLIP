#!/usr/bin/env bash
#SBATCH --job-name=chorus-release-summary
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/release_summary_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
PY=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
SCRIPT="$ROOT/report_tools/summarize_large50_pair.py"
"$PY" "$SCRIPT" --root "$ROOT/results" --model final \
  --sets final_repeat_v64_a final_repeat_v64_b final_repeat_v64_c final_default_v66 final_release_v70 \
  --training-set training_paired_v70 --out "$ROOT/results/summary_final_release_v70.json"
"$PY" "$SCRIPT" --root "$ROOT/results" --model persistent \
  --sets persistent_default_v67 persistent_release_v70 \
  --out "$ROOT/results/summary_persistent_release_v70.json"
printf 'DONE\n'
