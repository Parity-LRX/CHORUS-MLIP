#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
STATUS="${STATUS:-/home/ylzhang/chorus_runs/dpa4_master_286f12d_buckyball_20260724/queue_status.log}"

mkdir -p "$(dirname "${STATUS}")"
mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

mark "WAIT_INSTALL"
while screen -list | grep -q '\.dpa4_install'; do
  sleep 20
done
"${DPA_ENV}/bin/python" -c \
  'import deepmd, torch, vesin, vesin.torch, vesin_torch; assert torch.cuda.is_available(); print(deepmd.__version__, "vesin", vesin.__version__, "vesin-torch", vesin_torch.__version__)'
mark "INSTALL_OK"

mark "WAIT_CHORUS_GPU"
while screen -list | grep -q '\.full_nonlinear_screen'; do
  sleep 60
done

mark "START_SMOKE_20"
STEPS=20 REPO="${REPO}" DPA_ENV="${DPA_ENV}" \
  bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_master_buckyball.sh"
mark "SMOKE_OK"

mark "START_FULL_45000"
STEPS=45000 REPO="${REPO}" DPA_ENV="${DPA_ENV}" \
  bash "${REPO}/benchmarks/paper/scripts/training/run_dpa4_master_buckyball.sh"
mark "FULL_DONE"

# DPA-4 is intentionally first. Resume the short, already-prepared CHORUS
# calibration only after the external-model benchmark releases the GPU.
mark "START_PLAIN_FULL_CALIBRATION"
REPO="${REPO}" \
  bash "${REPO}/benchmarks/paper/scripts/training/calibrate_transition1x_plain_full.sh"
mark "PLAIN_FULL_CALIBRATION_DONE"
