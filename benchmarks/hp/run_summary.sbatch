#!/usr/bin/env bash
#SBATCH --job-name=chorus-level2-summary
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/summary_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
SOURCE="${SOURCE:-$ROOT/native_v10/chorus}"
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/${SUMMARY_SCRIPT:-summarize_throughput.py}"
printf 'DONE\n'
