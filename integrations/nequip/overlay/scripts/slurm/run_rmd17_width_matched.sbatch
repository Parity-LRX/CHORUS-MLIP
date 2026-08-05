#!/usr/bin/env bash
#SBATCH --job-name=nequip_width_match
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/nequip_chorus_runs/width-matched-%j.log

set -euo pipefail

REPO=/public/home/sps-xia/rxlin/NequIP-CHORUS
RUN_ROOT="${RUN_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_runs/rmd17_c72_width_matched_20260729}"
CHANNELS=72
EPOCHS=180
MOLECULES="revised_aspirin revised_ethanol revised_benzene"
device="${CUDA_VISIBLE_DEVICES}"

mkdir -p "${RUN_ROOT}"
cat >"${RUN_ROOT}/protocol.json" <<'EOF'
{
  "purpose": "parameter-matched capacity control for C64/R16 CHORUS",
  "channels": 72,
  "chorus": false,
  "backends": ["e3nn", "ictc"],
  "trainable_parameters": 768356,
  "reference_chorus_trainable_parameters": 757325,
  "relative_parameter_difference": 0.014566,
  "seed": 20260616,
  "epochs": 180,
  "selection_metric": "validation_f_mae",
  "tf32": false,
  "ema": false
}
EOF

launch() {
  local backend="$1"
  env REPO="${REPO}" RUN_ROOT="${RUN_ROOT}" CHANNELS="${CHANNELS}" \
    EPOCHS="${EPOCHS}" MOLECULES="${MOLECULES}" \
    bash "${REPO}/scripts/run_rmd17_variant.sh" \
      "${backend}" off "${device}" \
      >"${RUN_ROOT}/${backend}_c72.driver.log" 2>&1 &
}

launch e3nn
pid_e3nn=$!
launch ictc
pid_ictc=$!

status=0
if ! wait "${pid_e3nn}"; then status=1; fi
if ! wait "${pid_ictc}"; then status=1; fi
if (( status == 0 )); then
  touch "${RUN_ROOT}/ALL_DONE"
fi
exit "${status}"
