#!/usr/bin/env bash
#SBATCH --job-name=chorus-infer-report
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018/logs/report_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_inference_aoti_20261018
SOURCE="${SOURCE:?immutable source required}"
export PYTHONPYCACHEPREFIX="$ROOT/cache/report_pycache"
RUN="${RUN:-prepared_t211_v7}"
L3_RUN="${L3_RUN:-prepared_csr_t211_v9}"
MAX_FINAL_RUN="${MAX_FINAL_RUN:-prepared_csr_t211_v10}"
MAX_PERSISTENT_RUN="${MAX_PERSISTENT_RUN:-prepared_csr_t211_v13}"
REPORT_RUN="${REPORT_RUN:-report_combined_v13}"
set -x
/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python \
  "$SOURCE/benchmarks/hp/summarize_inference_aoti_nograph.py" --root "$ROOT" \
  --run "$RUN" --case-run "l3=$L3_RUN" \
  --scope-run "l3c256d4/final=$MAX_FINAL_RUN" "l3c256d4/persistent=$MAX_PERSISTENT_RUN" \
  --out "$ROOT/$REPORT_RUN"
