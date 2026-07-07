#!/bin/bash
# LR throughput pipeline re-run (current code): re-export 4 cores + deployed-MD timing + rRESPA conservation.
# Sequential to keep fig_sweep timing clean (no concurrent GPU work).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MACE_ICTC_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
LRX="${LRX:-/home/ylzhang/lrx}"
PY="${PY:-python3}"
cd "$LRX"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
LOG="${LR_PIPELINE_LOG:-${LRX}/lr_pipeline_rerun.log}"
: > "$LOG"
echo "PIPELINE START $(date)" >> "$LOG"

ISO="${LR_ISO_PT2:-${LRX}/mff_md_work/disp_path1_mbdslq_c64_cut9_iso.pt2}"
ANI="${LR_ANI_PT2:-${LRX}/mff_md_work/disp_path1_mbdslq_c64_cut9_aniso.pt2}"
C6="${LR_C6_PT2:-${LRX}/mff_md_work/disp_c6_c64_cut9.pt2}"
RESPA="${LR_RESPA_PT2:-${LRX}/respa_trained/disp_path1_trained.pt2}"

# 0) delete stale cores (force fresh re-export with current code)
for c in "$ISO" "$ANI" "$C6" "$RESPA"; do rm -f "$c" "$c.json" "$c.meta"; done
echo "deleted stale cores" >> "$LOG"

# 1) ISO + ANISO cores
echo "=== [1] export ISO/ANISO $(date) ===" >> "$LOG"
bash _respa_export_c64cut9.sh >> "$LOG" 2>&1
echo "rc_iso=$?" >> "$LOG"; ls -la "$ISO" "$ANI" >> "$LOG" 2>&1

# 2) C6 core (pairwise-c6; mirror of ISO export, only --dispersion-mode differs)
echo "=== [2] export C6 $(date) ===" >> "$LOG"
$PY -m mace_ictc.cli.export_aoti_core --route baseline --atoms 128 --degree 40 --channels 64 \
  --lmax 2 --num-interaction 2 --contraction-order 3 --attn-heads 0 --dtype float32 --device cuda \
  --dynamic --dispersion-mode pairwise-c6 --dispersion-cutoff 9.0 --out "$C6" >> "$LOG" 2>&1
echo "rc_c6=$?" >> "$LOG"; ls -la "$C6" >> "$LOG" 2>&1

# 3) rRESPA trained core
echo "=== [3] export rRESPA trained $(date) ===" >> "$LOG"
bash respa_trained/do_export.sh >> "$LOG" 2>&1
echo "rc_respa_export=$?" >> "$LOG"; ls -la "$RESPA" >> "$LOG" 2>&1

# 4) deployed-MD timing (fig_sweep) — clean GPU, timing-critical
echo "=== [4] fig_sweep deployed-MD $(date) ===" >> "$LOG"
rm -f fig_respa_c6/SWEEP_DONE
bash "$SCRIPT_DIR/fig_sweep_deployed_md.sh" >> "$LOG" 2>&1
echo "rc_figsweep=$?" >> "$LOG"

# 5) rRESPA conservation sweeps
echo "=== [5] rRESPA sweep (K=2..50) $(date) ===" >> "$LOG"
bash respa_trained/sweep.sh >> "$LOG" 2>&1
echo "rc_respa_sweep=$?" >> "$LOG"
echo "=== [5b] rRESPA sweep2 (high-K 75/100/150) $(date) ===" >> "$LOG"
bash respa_trained/sweep2.sh >> "$LOG" 2>&1
echo "rc_respa_sweep2=$?" >> "$LOG"

echo "PIPELINE DONE $(date)" >> "$LOG"
