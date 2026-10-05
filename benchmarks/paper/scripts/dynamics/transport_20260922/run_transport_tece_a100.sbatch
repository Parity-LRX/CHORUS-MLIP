#!/usr/bin/env bash
#SBATCH --job-name=transport-tece
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/tece_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/tece_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
REPO="${REPO:-${ROOT}/source/CHORUS-MLIP}"
TACE_REPO="${TACE_REPO:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/tace_chorus_benchmark}"
TACE_BIN="${TACE_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin}"
TACE_OVERLAY="${TACE_OVERLAY:-/public/home/sps-xia/rxlin/a100_scaling_assets_20260804/tace_chorus_venv_overlay/site-packages}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/prepared}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/tece}"
STEPS="${STEPS:-45000}"
EPOCHS="${EPOCHS:-180}"
RUN_EVAL="${RUN_EVAL:-1}"

datasets=(water lips)
channels_list=(36 48)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dataset="${datasets[$((task_id / 2))]}"
channels="${channels_list[$((task_id % 2))]}"
data="${DATA_ROOT}/${dataset}"
run="${RUN_ROOT}/${dataset}/c${channels}"

[[ -f "${ROOT}/DATA_READY" ]]
mkdir -p "${run}"
cd "${run}"
export PYTHONPATH="${TACE_OVERLAY}:${TACE_REPO}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

atomic_energy="$(${TACE_BIN}/python - "${data}/metadata.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print("{" + ", ".join(
    f"{z}: {energy:.17g}"
    for z, energy in zip(x["atomic_numbers"], x["element_reference_energies_ev"])
) + "}")
PY
)"

if [[ ! -f "${run}/TRAIN_DONE" ]]; then
  env TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=0 \
    "${TACE_BIN}/python" -m tace.scripts.train \
      --config-path "${TACE_REPO}/example/train" -cn tace_buckyball_tece_s.yaml \
      "misc.project_name=TECE_C${channels}_${dataset}_Seed20260616" \
      misc.global_seed=20260616 misc.allow_tf32=false \
      dataset.split_seed=20260616 \
      "dataset.train_file=${data}/extxyz/train.extxyz" \
      "dataset.valid_file=${data}/extxyz/val.extxyz" \
      dataset.keys.energy_key=energy dataset.keys.forces_key=forces \
      dataset.train_dataloader.batch_size=4 \
      dataset.valid_dataloader.batch_size=4 \
      "trainer.max_steps=${STEPS}" "trainer.max_epochs=${EPOCHS}" \
      "scheduler.T_max=${STEPS}" \
      model.config.fidelity.0.name=DFT \
      "model.config.fidelity.0.atomic_energy=${atomic_energy}" \
      "model.config.num_channel=${channels}" \
      model.config.mmax=2 model.config.Lmax=2 model.config.lmax=2 \
      model.config.num_layers=2 model.config.product_basis.correlation=3 \
      synth_metric.monitor_metric_name=val/forces_mae \
      callbacks.checkpoint_epoch.monitor=val/forces_mae \
      callbacks.checkpoint_epoch.save_top_k=-1 \
      '~callbacks.ema' '~callbacks.early_stopping' >train.log 2>&1
  touch "${run}/TRAIN_DONE"
fi

if [[ "${RUN_EVAL}" == "1" && ! -f "${run}/DONE" ]]; then
  selected="$(${TACE_BIN}/python - <<'PY'
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

versions = sorted(
    Path("lightning_logs").glob("version_*"),
    key=lambda path: int(path.name.split("_")[-1]),
)
if not versions:
    raise SystemExit("no TensorBoard validation log")
events = EventAccumulator(str(versions[-1]))
events.Reload()
points = events.Scalars("val/forces_mae")
best = min(points, key=lambda point: (point.value, point.step))
checkpoint_step = best.step + 1
candidates = sorted(Path("checkpoints_epoch").glob(f"TECE-*-{checkpoint_step}.ckpt"))
if len(candidates) != 1:
    raise SystemExit(f"checkpoint mismatch at {checkpoint_step}: {candidates}")
Path("selection.json").write_text(json.dumps({
    "selection_rule": "minimum full-validation Force MAE; test never selects",
    "tensorboard_step": best.step,
    "checkpoint_step": checkpoint_step,
    "validation_force_mae_ev_per_angstrom": best.value,
    "checkpoint": str(candidates[0]),
}, indent=2) + "\n")
print(candidates[0])
PY
)"
  "${TACE_BIN}/python" -m tace.scripts.eval \
    -i "${data}/extxyz/val.extxyz" -m "${selected}" -t 1 -e 1 -b 4 \
    --device cuda --dtype float32 --nl_backend matscipy \
    --energy_key energy --forces_key forces >val.log 2>&1
  "${TACE_BIN}/python" -m tace.scripts.eval \
    -i "${data}/extxyz/test.extxyz" -m "${selected}" -t 1 -e 1 -b 4 \
    --device cuda --dtype float32 --nl_backend matscipy \
    --energy_key energy --forces_key forces >test.log 2>&1
  touch "${run}/DONE"
fi

echo "TRANSPORT_TECE_DONE dataset=${dataset} channels=${channels} $(date -Is)"
