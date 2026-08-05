#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
DATA="${DATA:-/home/ylzhang/lrx/md22/dpa4_buckyball_matched_20260724}"
OUT="${OUT:-/home/ylzhang/chorus_runs/dpa4_master_286f12d_buckyball_20260724}"
STEPS="${STEPS:-45000}"
CHANNELS="${CHANNELS:-32}"
MIXING_LAYERS="${MIXING_LAYERS:-3}"
RCUT="${RCUT:-6.0}"
ENABLE_EMA="${ENABLE_EMA:-true}"

mkdir -p "${OUT}"
exec 9>"${OUT}/run.lock"
flock 9
if [[ -f "${OUT}/DONE" ]]; then
  echo "REUSE_DONE ${OUT}"
  exit 0
fi

cp "${REPO}/benchmarks/paper/external/dpa4/buckyball_master_286f12d.json" \
  "${OUT}/input.base.json"
"${DPA_ENV}/bin/python" - \
  "${OUT}/input.base.json" \
  "${OUT}/input.json" \
  "${STEPS}" \
  "${CHANNELS}" \
  "${MIXING_LAYERS}" \
  "${RCUT}" \
  "${ENABLE_EMA}" <<'PY'
import json
import sys

source, destination, steps, channels, mixing_layers, rcut, enable_ema = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)
config["training"]["numb_steps"] = int(steps)
config["training"]["save_dir"] = f"ckpt_steps{steps}"
config["training"]["disp_file"] = f"lcurve_steps{steps}.out"
config["validating"]["save_best_dir"] = f"ckpt_best_steps{steps}"
config["model"]["descriptor"]["channels"] = int(channels)
config["model"]["descriptor"]["mixing_layers"] = int(mixing_layers)
config["model"]["descriptor"]["rcut"] = float(rcut)
config["training"]["enable_ema"] = enable_ema.lower() in {"1", "true", "yes", "on"}
config["validating"]["ema_full_validation"] = False
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY

{
  echo "DPA4_MASTER_SHA 286f12d3e3cb69eefb33d4716267f6457dffa80b"
  echo "START $(date -Is)"
  echo "STEPS ${STEPS}"
  echo "CHANNELS ${CHANNELS}"
  echo "MIXING_LAYERS ${MIXING_LAYERS}"
  echo "RCUT ${RCUT}"
  echo "ENABLE_EMA ${ENABLE_EMA}"
  echo "TF32 false"
  "${DPA_ENV}/bin/python" -c \
    'import deepmd, torch, vesin, vesin.torch, vesin_torch; print("deepmd", deepmd.__version__); print("torch", torch.__version__, "cuda", torch.version.cuda); print("vesin", vesin.__version__, "vesin-torch", vesin_torch.__version__)'
} | tee "${OUT}/protocol.log"

cd "${OUT}"
/usr/bin/time -f "WALL_SECONDS %e" \
  "${DPA_ENV}/bin/dp" --pt train input.json \
  >"${OUT}/train_steps${STEPS}.log" 2>&1
echo "DONE $(date -Is)" | tee -a "${OUT}/protocol.log"
touch "${OUT}/DONE"
