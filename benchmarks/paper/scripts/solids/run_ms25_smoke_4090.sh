#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-ms25-20260818}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/home/ylzhang/chorus_data/ms25_20260818/prepared}"
OUT="${OUT:-/home/ylzhang/chorus_runs/ms25_smoke_20260818/hea_persistent}"
metadata="${DATA_ROOT}/hea/set1/metadata.json"
data_dir="${DATA_ROOT}/hea/set1/chorus"

mkdir -p "${OUT}/checkpoints" "${OUT}/logs"
export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

readarray -t values < <("${PYTHON_BIN}" - "${metadata}" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
print(x["cutoff_angstrom"])
print(x["average_train_neighbours"])
print(",".join(map(str,x["atomic_numbers"])))
print(",".join(f"{v:.17g}" for v in x["element_reference_energies_ev"]))
PY
)

"${PYTHON_BIN}" -m chorus.cli.train \
  --data-dir "${data_dir}" --train-prefix train --val-prefix val \
  --channels 128 --lmax 2 --max-ell 2 \
  --num-interaction 2 --correlation 3 \
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
  --first-layer-self-connection --mace-compatible-random-init \
  --readout-hidden-channels 64 \
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
  --max-radius "${values[0]}" --avg-num-neighbors "${values[1]}" \
  --atomic-energy-keys "${values[2]}" --atomic-energy-values="${values[3]}" \
  --scaling std_scaling --epochs 1 --max-steps 20 --batch-size 1 \
  --dtype float32 --device cuda --num-workers 0 \
  --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
  --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
  --optimizer adamw --optimizer-param-groups mace \
  --weight-decay 5e-7 --amsgrad --max-grad-norm 10 \
  --phase-mode final-full-l-residual \
  --phase-amplitude softplus --phase-coefficient polar \
  --phase-context content --phase-density-pairs full-nonlinear \
  --phase-normalization avg-neighbors \
  --phase-placement pre-product-full-l --phase-scope persistent \
  --phase-hidden-channels 32 --phase-scale-init 0.05 --phase-density-rank 16 \
  --train-makefx-compile --require-train-makefx-compile \
  --makefx-buckets 4 --makefx-max-slots 8 \
  --pad-nodes-to-max --pad-edges-to-max \
  --seed 20260616 --log-interval 1 --keep-checkpoints 2 \
  --checkpoint "${OUT}/checkpoints/smoke.pth" >"${OUT}/logs/train.log" 2>&1

touch "${OUT}/DONE"
echo "MS25_SMOKE_DONE $(date -Is)"
