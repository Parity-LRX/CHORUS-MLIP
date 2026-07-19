#!/usr/bin/env bash
set -euo pipefail

# Confirmatory phase-mechanism matrix on the public Cheng liquid-water split.
# This campaign reuses the archived matched MACE/MACE-ICTC baselines and trains
# only the seven phase-mechanism controls.  Runs are serialized for a single
# RTX 4090 and split by seed so a stopped campaign can resume safely.

PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_ICTC_REPO="${MACE_ICTC_REPO:-/home/ylzhang/CHORUS-MLIP}"
DATA_ROOT="${DATA_ROOT:-/tmp/mace_ictd_public_water}"
ROOT="${PHASE_WATER_ROOT:-${MACE_ICTC_REPO}/benchmarks/paper/results/phase/mechanism_confirmatory_water_3seed_20260719}"
BASELINE_RESULTS="${BASELINE_RESULTS:-${MACE_ICTC_REPO}/benchmarks/paper/results/training/water_cheng_convergence_20260618}"
RUN_MATRIX="${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh"
WAIT_PID="${WAIT_PID:-}"

MODES="${MODES:-ictc_phase_positive_full_l_eager,ictc_phase_signed_full_l_eager,ictc_phase_cartesian_full_l_eager,ictc_phase_radial_full_l_eager,ictc_phase_full_l_softplus_eager,ictc_phase_diagonal_full_l_eager,ictc_attention_eager}"
SEEDS="${SEEDS:-20260616,20260617,20260618}"

mkdir -p "${ROOT}"/{chunks,analysis}

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    echo "WAIT existing_campaign_pid=${WAIT_PID} $(date)" | tee -a "${ROOT}/queue.log"
    sleep 60
  done
fi

data="${DATA_ROOT}/cheng_water"
for required in train.extxyz val.extxyz test.extxyz processed_train.h5 processed_val.h5 processed_test.h5; do
  if [[ ! -f "${data}/${required}" ]]; then
    echo "missing Cheng-water input: ${data}/${required}" >&2
    exit 2
  fi
done

cat > "${ROOT}/protocol.json" <<EOF
{
  "protocol": "confirmatory_cheng_liquid_water_three_seed_phase_mechanisms",
  "dataset": "cheng_water",
  "source_split": "${data}",
  "archived_baseline_results": "${BASELINE_RESULTS}",
  "seeds": "${SEEDS}",
  "modes": "${MODES}",
  "checkpoint_selection": "minimum validation loss",
  "primary_reporting": "force and energy MAE from the same minimum-validation-loss checkpoint",
  "epochs": 300,
  "batch_size": 16,
  "average_num_neighbors": 34.0,
  "r_max_angstrom": 4.5,
  "parallel_jobs": 1
}
EOF

sha256sum \
  "${MACE_ICTC_REPO}/mace_ictc/models/pure_cartesian_ictd_fix.py" \
  "${MACE_ICTC_REPO}/mace_ictc/cli/train.py" \
  "${MACE_ICTC_REPO}/mace_ictc/training/train_loop.py" \
  "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_md17_matrix.sh" \
  "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/run_phase_confirmatory_water.sh" \
  "${MACE_ICTC_REPO}/benchmarks/paper/scripts/training/analyze_md17_convergence.py" \
  > "${ROOT}/code_manifest.sha256"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
for raw_seed in "${SEED_ARRAY[@]}"; do
  seed="$(echo "${raw_seed}" | xargs)"
  name="cheng_water_seed${seed}"
  out="${ROOT}/chunks/${name}"
  if [[ -f "${out}/.complete" ]]; then
    echo "SKIP ${name} $(date)" | tee -a "${ROOT}/status.log"
    continue
  fi
  echo "START ${name} $(date)" | tee -a "${ROOT}/status.log"
  OUT_ROOT="${out}" \
  DATA_ROOT="${DATA_ROOT}" \
  DATASETS=cheng_water \
  SEEDS="${seed}" \
  MODES="${MODES}" \
  EPOCHS=300 \
  BATCH_SIZE=16 \
  AVG_NEIGHBORS=34.0 \
  PARALLEL_JOBS=1 \
  NUM_WORKERS=2 \
  PYTHON_BIN="${PYTHON_BIN}" \
  MACE_ICTC_REPO="${MACE_ICTC_REPO}" \
  bash "${RUN_MATRIX}"
  touch "${out}/.complete"
  echo "OK ${name} $(date)" | tee -a "${ROOT}/status.log"
done

log_dirs=()
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
