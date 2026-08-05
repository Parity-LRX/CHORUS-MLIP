#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
RAW_ROOT="${RAW_ROOT:-/home/ylzhang/lrx/md22/all_raw}"
OUT_ROOT="${OUT_ROOT:-/home/ylzhang/lrx/md22/all_lowdata600_test1000_20260723}"
PREPARE_NPZ="${REPO}/benchmarks/paper/scripts/training/prepare_md22_npz_lowdata.py"
PREPARE_TEST="${REPO}/benchmarks/paper/scripts/training/prepare_md22_test_h5.py"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}"
for source in "${RAW_ROOT}"/md22_*.npz; do
  name="$(basename "${source}" .npz)"
  tag="${name#md22_}"
  tag="${tag//-/_}"
  out="${OUT_ROOT}/${tag}"
  processed="${out}/processed"
  if [[ -f "${out}/.complete" ]]; then
    echo "SKIP ${tag}"
    continue
  fi
  echo "START ${tag} $(date)" | tee -a "${OUT_ROOT}/status.log"
  "${PYTHON_BIN}" "${PREPARE_NPZ}" --input "${source}" --output-dir "${out}" \
    --candidate-size 1200 --test-size 1000 --energy-bins 20 --seed 20260616

  mapfile -t keys < <("${PYTHON_BIN}" - "${out}/split_metadata.json" <<'PY'
import json, sys
for value in json.load(open(sys.argv[1]))["atomic_numbers"]:
    print(value)
PY
  )
  "${PYTHON_BIN}" -m chorus.cli.preprocess \
    --input-file "${out}/candidate_1200.extxyz" --output-dir "${processed}" \
    --train-ratio 0.5 --seed 20260616 --atomic-energy-keys "${keys[@]}" \
    --max-radius 5.0 --num-workers 8

  eval "$("${PYTHON_BIN}" - "${processed}/fitted_E0.csv" "${processed}/processed_train.h5.counts.npz" <<'PY'
import csv, sys, numpy as np
rows=list(csv.DictReader(open(sys.argv[1])))
counts=np.load(sys.argv[2])
avg=float(counts["edge_counts"].sum()/counts["node_counts"].sum())
print("E0_KEYS=" + ",".join(row["Atom"] for row in rows))
print("E0_VALUES=" + ",".join(row["E0"] for row in rows))
print("AVG_NEIGHBORS=" + repr(avg))
PY
  )"
  "${PYTHON_BIN}" "${PREPARE_TEST}" \
    --input "${out}/heldout_test.extxyz" --output-dir "${processed}" \
    --atomic-energy-keys "${E0_KEYS}" --atomic-energy-values="${E0_VALUES}" \
    --max-radius 5.0 --num-workers 8
  {
    echo "E0_KEYS=${E0_KEYS}"
    echo "E0_VALUES=${E0_VALUES}"
    echo "AVG_NEIGHBORS=${AVG_NEIGHBORS}"
  } >"${out}/training.env"
  touch "${out}/.complete"
  echo "END ${tag} $(date)" | tee -a "${OUT_ROOT}/status.log"
done
echo "ALL_OK $(date)" | tee -a "${OUT_ROOT}/status.log"
