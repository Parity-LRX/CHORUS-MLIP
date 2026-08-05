#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
OUT="${OUT:-/home/ylzhang/chorus_runs/dpa4_amp_audit_20260728}"
DPA_PY="${DPA_PY:-/home/ylzhang/venvs/dpa4-master/bin/python}"
BENCH="${REPO}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
SIZES="${SIZES:-32,64,128,256,512,1024,2048}"

DPA32_ROOT="/home/ylzhang/chorus_runs/large_scale_main_20260724/external/dpa4_t1x_large/c32_mix3"
DPA48_ROOT="/home/ylzhang/chorus_runs/dpa4_c48_scaling_20260727/t1x_c48_mix3"

mkdir -p "${OUT}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export DP_AMP_INFER=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=20260728

run_one() {
  local name="$1"
  local root="$2"
  local checkpoint="$3"
  local amp="$4"
  printf 'START %s %s\n' "${name}" "$(date -Is)" | tee -a "${OUT}/status.log"
  "${DPA_PY}" "${BENCH}" --engine dpa4 \
    --config "${root}/input.json" --checkpoint "${checkpoint}" \
    --dpa-amp "${amp}" --sizes "${SIZES}" \
    --output "${OUT}/${name}.json" >"${OUT}/${name}.log" 2>&1
  printf 'DONE %s %s\n' "${name}" "$(date -Is)" | tee -a "${OUT}/status.log"
}

run_one dpa4_c32_bf16_amp "${DPA32_ROOT}" \
  "${DPA32_ROOT}/ckpt_steps100000/model.ckpt-84375.pt" on
run_one dpa4_c32_strict_fp32 "${DPA32_ROOT}" \
  "${DPA32_ROOT}/ckpt_steps100000/model.ckpt-84375.pt" off
run_one dpa4_c48_bf16_amp "${DPA48_ROOT}" \
  "${DPA48_ROOT}/ckpt_steps100000/model.ckpt-90625.pt" on
run_one dpa4_c48_strict_fp32 "${DPA48_ROOT}" \
  "${DPA48_ROOT}/ckpt_steps100000/model.ckpt-90625.pt" off

"${DPA_PY}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = ("c32", "c48")
summary = {"protocol": "same checkpoint and compiled DPA-4 model; AMP is the only variable"}
for model in models:
    bf16 = json.loads((root / f"dpa4_{model}_bf16_amp.json").read_text())
    fp32 = json.loads((root / f"dpa4_{model}_strict_fp32.json").read_text())
    rows = {}
    for task in ("inference", "train"):
        left = {
            int(row["natoms"]): row
            for row in bf16["rows"]
            if row["task"] == task and row["status"] == "ok"
        }
        right = {
            int(row["natoms"]): row
            for row in fp32["rows"]
            if row["task"] == task and row["status"] == "ok"
        }
        rows[task] = [
            {
                "natoms": natoms,
                "bf16_amp_atoms_per_second": left[natoms]["atoms_per_second"],
                "strict_fp32_atoms_per_second": right[natoms]["atoms_per_second"],
                "bf16_over_fp32_speedup": (
                    left[natoms]["atoms_per_second"]
                    / right[natoms]["atoms_per_second"]
                ),
                "bf16_amp_peak_memory_gib": left[natoms]["peak_memory_gib"],
                "strict_fp32_peak_memory_gib": right[natoms]["peak_memory_gib"],
            }
            for natoms in sorted(set(left) & set(right))
        ]
    summary[model] = rows
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

touch "${OUT}/DONE"
printf 'ALL_DONE %s\n' "$(date -Is)" | tee -a "${OUT}/status.log"
