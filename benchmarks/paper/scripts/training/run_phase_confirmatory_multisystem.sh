#!/usr/bin/env bash
set -euo pipefail

# Confirmatory phase-mechanism matrix:
#   revised_{aspirin,ethanol,benzene} x seeds {20260616,17,18} x seven modes.
#
# The completed revised_aspirin/20260616 chunk is reused from the screening run.
# Every other system/seed pair is an independent restartable chunk.  A chunk is
# skipped only after its child matrix exits successfully and writes .complete.

PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP}"
ROOT="${PHASE_CONFIRMATORY_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/mechanism_confirmatory_multisystem_3seed_20260719}"
SCREENING_ASPIRIN_SEED16="${SCREENING_ASPIRIN_SEED16:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/mechanism_controls_aspirin_seed16_formal_20260719}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"

MODES="${MODES:-ictc_phase_positive_full_l_eager,ictc_phase_signed_full_l_eager,ictc_phase_cartesian_full_l_eager,ictc_phase_radial_full_l_eager,ictc_phase_full_l_softplus_eager,ictc_phase_diagonal_full_l_eager,ictc_attention_eager}"

mkdir -p "${ROOT}"/{chunks,analysis}

cat > "${ROOT}/protocol.json" <<EOF
{
  "protocol": "confirmatory_multisystem_three_seed_phase_mechanisms",
  "datasets": ["revised_aspirin", "revised_ethanol", "revised_benzene"],
  "seeds": [20260616, 20260617, 20260618],
  "modes": "${MODES}",
  "checkpoint_selection": "minimum validation loss",
  "primary_reporting": "force and energy MAE from the same minimum-validation-loss checkpoint",
  "epochs": 300,
  "parallel_jobs": 1,
  "reused_chunk": "${SCREENING_ASPIRIN_SEED16}"
}
EOF

sha256sum \
  "${MACE_ICTC_REPO}/chorus/models/pure_cartesian_ictd_fix.py" \
  "${MACE_ICTC_REPO}/chorus/cli/train.py" \
  "${MACE_ICTC_REPO}/chorus/training/train_loop.py" \
  "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh" \
  "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_confirmatory_multisystem.sh" \
  "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/analyze_md17_convergence.py" \
  > "${ROOT}/code_manifest.sha256"

run_chunk() {
  local dataset="$1"
  local seed="$2"
  local name="${dataset}_seed${seed}"
  local out="${ROOT}/chunks/${name}"
  if [[ -f "${out}/.complete" ]]; then
    echo "SKIP ${name} $(date)" | tee -a "${ROOT}/status.log"
    return
  fi
  echo "START ${name} $(date)" | tee -a "${ROOT}/status.log"
  OUT_ROOT="${out}" \
  DATASETS="${dataset}" \
  SEEDS="${seed}" \
  MODES="${MODES}" \
  EPOCHS=300 \
  PARALLEL_JOBS=1 \
  NUM_WORKERS=2 \
  PYTHON_BIN="${PYTHON_BIN}" \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  bash "${RUN_MATRIX}"
  touch "${out}/.complete"
  echo "OK ${name} $(date)" | tee -a "${ROOT}/status.log"
}

# Interleave systems so cross-system evidence arrives before the whole matrix ends.
# revised_aspirin/20260616 is the completed screening run referenced above.
run_chunk revised_aspirin 20260617
run_chunk revised_benzene 20260616
run_chunk revised_ethanol 20260616
run_chunk revised_aspirin 20260618
run_chunk revised_benzene 20260617
run_chunk revised_ethanol 20260617
run_chunk revised_benzene 20260618
run_chunk revised_ethanol 20260618

log_dirs=("${SCREENING_ASPIRIN_SEED16}/logs")
while IFS= read -r log_dir; do
  log_dirs+=("${log_dir}")
done < <(find "${ROOT}/chunks" -mindepth 2 -maxdepth 2 -type d -name logs | sort)

"${PYTHON_BIN}" \
  "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/analyze_md17_convergence.py" \
  "${log_dirs[@]}" \
  --out-dir "${ROOT}/analysis" \
  --target-epoch 300 \
  --plots

echo "ALL_OK $(date)" | tee -a "${ROOT}/status.log"
echo "results: ${ROOT}"
