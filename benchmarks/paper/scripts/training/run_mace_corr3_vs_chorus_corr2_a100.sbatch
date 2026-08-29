#!/usr/bin/env bash
#SBATCH --job-name=chorus-corr2-control
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_corr2_control_20260826/slurm-%j.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_corr2_control_20260826/slurm-%j.err

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-main-17ccf67}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_corr2_control_20260826/chorus_final_c128_l2_corr2_r16}"
BASE_QUEUE="${REPO}/benchmarks/paper/scripts/training/queue_chorus_c64_rank16_all.sh"
STATUS="${ROOT}/queue_status.log"

mkdir -p "${ROOT}"
exec 9>"${ROOT}/run.lock"
if ! flock -n 9; then
  printf 'SKIP_LOCKED %s\n' "$(date -Is)" | tee -a "${STATUS}"
  exit 0
fi

if [[ -f "${ROOT}/ALL_DONE" ]]; then
  printf 'SKIP_ALL_DONE %s\n' "$(date -Is)" | tee -a "${STATUS}"
  exit 0
fi

cat >"${ROOT}/protocol.env" <<'EOF'
comparison=MACE-ICTC CHORUS-Off correlation=3 (existing) versus CHORUS-Final correlation=2 (this run)
channels=128
hidden_lmax=2
max_ell=2
num_interactions=2
correlation=2
phase_density_rank=16
phase_scope=final
phase_density_pairs=full-nonlinear
seed=20260616
precision=float32
tf32=false
ema=false
energy_force_weight=1:100
checkpoint_selection=minimum validation Force MAE; earliest step breaks ties
test_used_for_selection=false
EOF

printf 'START job=%s host=%s commit=%s %s\n' \
  "${SLURM_JOB_ID:-none}" "$(hostname)" "$(git -C "${REPO}" rev-parse --short HEAD)" \
  "$(date -Is)" | tee -a "${STATUS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "${STATUS}"

run_one() {
  local system="$1"
  rm -f "${ROOT}/ALL_DONE"
  REPO="${REPO}" PYTHON_BIN="${PYTHON_BIN}" MACE_TORCH_PATH="${MACE_TORCH_PATH}" \
    ROOT="${ROOT}" STATUS="${STATUS}" SYSTEMS="${system}" \
    CHANNELS=128 RANK=16 CORRELATION=2 NUM_INTERACTIONS=2 \
    HIDDEN_LMAX=2 MAX_ELL=2 MODE=chorus PHASE_CONTEXT=content \
    SEED=20260616 \
    T1X_DATA_DIR="${DATA_ROOT}/transition1x/chorus_reaction_id_50k_seed20260616" \
    MAL_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal" \
    STI_DATA_DIR="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/sti" \
    BUCKY_DATA_DIR="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed" \
    bash "${BASE_QUEUE}"
  rm -f "${ROOT}/ALL_DONE"
}

# MAL is assigned to the 4090; this card runs the disjoint remainder.
for system in bucky sti t1x; do
  run_one "${system}"
done

touch "${ROOT}/ALL_DONE"
printf 'ALL_CORR2_CONTROL_TASKS_DONE %s\n' "$(date -Is)" | tee -a "${STATUS}"
