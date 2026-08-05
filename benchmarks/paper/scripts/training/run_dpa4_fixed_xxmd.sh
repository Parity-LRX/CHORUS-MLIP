#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
DPA_ENV="${DPA_ENV:-/home/ylzhang/venvs/dpa4-master}"
BASE_CONFIG="${BASE_CONFIG:-${REPO}/benchmarks/paper/external/dpa4/buckyball_master_286f12d.json}"
DATA="${DATA:?set DATA to a converted DeepMD system root}"
OUT="${OUT:?set OUT}"
STEPS="${STEPS:-45000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHANNELS="${CHANNELS:-32}"
MIXING_LAYERS="${MIXING_LAYERS:-3}"
RCUT="${RCUT:-5.0}"
SEED="${SEED:-20260616}"
USE_COMPILE="${USE_COMPILE:-0}"
USE_AMP="${USE_AMP:-1}"

mkdir -p "${OUT}"
exec 9>"${OUT}/run.lock"
flock 9
if [[ -f "${OUT}/DONE" ]]; then
  echo "REUSE_DONE ${OUT}"
  exit 0
fi

cp "${BASE_CONFIG}" "${OUT}/input.base.json"
"${DPA_ENV}/bin/python" - \
  "${OUT}/input.base.json" "${OUT}/input.json" "${DATA}" "${STEPS}" \
  "${BATCH_SIZE}" "${CHANNELS}" "${MIXING_LAYERS}" "${RCUT}" "${SEED}" \
  "${USE_COMPILE}" "${USE_AMP}" <<'PY'
import json
import math
import sys
from pathlib import Path

import numpy as np

(
    source,
    destination,
    data_root,
    steps,
    batch_size,
    channels,
    mixing_layers,
    rcut,
    seed,
    use_compile,
    use_amp,
) = sys.argv[1:]
data_root = Path(data_root)
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)

train_type_maps = sorted((data_root / "train").rglob("type_map.raw"))
val_type_maps = sorted((data_root / "val").rglob("type_map.raw"))
if not train_type_maps or not val_type_maps:
    raise RuntimeError(f"no DeepMD systems found below {data_root}")
type_map = train_type_maps[0].read_text().split()
for path in train_type_maps + val_type_maps:
    candidate = path.read_text().split()
    if candidate != type_map:
        raise RuntimeError(
            f"inconsistent type_map: {path} has {candidate}, expected {type_map}"
        )
train_energy_files = sorted((data_root / "train").rglob("set.*/energy.npy"))
val_energy_files = sorted((data_root / "val").rglob("set.*/energy.npy"))
n_train = sum(int(np.load(path, mmap_mode="r").shape[0]) for path in train_energy_files)
n_val = sum(int(np.load(path, mmap_mode="r").shape[0]) for path in val_energy_files)
if n_train == 0 or n_val == 0:
    raise RuntimeError(f"empty DeepMD split below {data_root}")
batch_size = int(batch_size)
epoch_steps = math.ceil(n_train / batch_size)

config["_comment"] = (
    "DPA-4 master 286f12d; official split represented as one or more "
    "fixed-topology DeepMD systems; "
    "strict FP32, no EMA, validation checkpoints retained for Force-MAE selection."
)
config["model"]["type_map"] = type_map
descriptor = config["model"]["descriptor"]
descriptor["rcut"] = float(rcut)
descriptor["channels"] = int(channels)
descriptor["mixing_layers"] = int(mixing_layers)
descriptor["seed"] = int(seed)
config["model"]["fitting_net"]["seed"] = int(seed)
config["model"]["enable_tf32"] = False
config["model"]["use_compile"] = bool(int(use_compile))
config["model"]["descriptor"]["use_amp"] = bool(int(use_amp))

training = config["training"]
training["training_data"]["systems"] = [str(data_root / "train")]
training["training_data"]["batch_size"] = batch_size
training["validation_data"]["systems"] = [str(data_root / "val")]
training["validation_data"]["batch_size"] = batch_size
training["validation_data"]["numb_batch"] = math.ceil(n_val / batch_size)
training["numb_steps"] = int(steps)
training["save_freq"] = epoch_steps
training["disp_freq"] = epoch_steps
training["save_dir"] = f"ckpt_steps{steps}"
training["max_ckpt_keep"] = math.ceil(int(steps) / epoch_steps) + 2
training["disp_file"] = f"lcurve_steps{steps}.out"
training["enable_ema"] = False
training["seed"] = int(seed)
config["validating"]["save_best_dir"] = f"ckpt_best_steps{steps}"
config["validating"]["ema_full_validation"] = False
config["validating"]["tf32_infer"] = False

with open(destination, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
print(
    json.dumps(
        {
            "type_map": type_map,
            "n_train": n_train,
            "n_val": n_val,
            "train_systems": len(train_type_maps),
            "val_systems": len(val_type_maps),
            "epoch_steps": epoch_steps,
            "saved_checkpoints": training["max_ckpt_keep"],
        }
    )
)
PY

{
  echo "DPA4_MASTER_SHA 286f12d3e3cb69eefb33d4716267f6457dffa80b"
  echo "START $(date -Is)"
  echo "DATA ${DATA}"
  echo "STEPS ${STEPS}"
  echo "BATCH_SIZE ${BATCH_SIZE}"
  echo "CHANNELS ${CHANNELS}"
  echo "MIXING_LAYERS ${MIXING_LAYERS}"
  echo "RCUT ${RCUT}"
  echo "ENABLE_EMA false"
  echo "TF32 false"
  echo "USE_COMPILE ${USE_COMPILE}"
  echo "DESCRIPTOR_USE_AMP ${USE_AMP}"
  "${DPA_ENV}/bin/python" -c \
    'import deepmd, torch; print("deepmd", deepmd.__version__); print("torch", torch.__version__, "cuda", torch.version.cuda)'
} | tee "${OUT}/protocol.log"

cd "${OUT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
restart_args=()
latest_checkpoint="$(
  { find "${OUT}/ckpt_steps${STEPS}" -maxdepth 1 -type f \
      -name 'model.ckpt-*.pt' -print 2>/dev/null || true; } \
    | sed -E 's/.*model\.ckpt-([0-9]+)\.pt/\1 &/' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
)"
if [[ -n "${latest_checkpoint}" ]]; then
  restart_prefix="${latest_checkpoint%.pt}"
  restart_args=(--restart "${restart_prefix}")
  {
    echo "RESTART $(date -Is)"
    echo "RESTART_CHECKPOINT ${latest_checkpoint}"
  } | tee -a "${OUT}/protocol.log"
fi
/usr/bin/time -f "WALL_SECONDS %e" \
  "${DPA_ENV}/bin/dp" --pt train input.json "${restart_args[@]}" \
  >>"${OUT}/train_steps${STEPS}.log" 2>&1
echo "DONE $(date -Is)" | tee -a "${OUT}/protocol.log"
touch "${OUT}/DONE"
