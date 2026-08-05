#!/usr/bin/env bash
#SBATCH --job-name=a100_scale_summary
#SBATCH --partition=CPU-INT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804/logs/summary-%j.log

set -euo pipefail

REPO="${REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730}"
ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_runs/a100_scaling_20260804}"
OUT="${OUT:-${ROOT}/aggregate}"
RAW="${OUT}/raw"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"

rm -rf "${RAW}"
mkdir -p "${RAW}" "${ROOT}/logs"
for filename in \
  mace_sh_c128.json \
  mace_ictc_c128.json \
  chorus_final_c128_r8.json \
  chorus_final_c128_r16.json \
  chorus_final_c128_r32.json \
  chorus_persistent_c128_r16.json; do
  cp "${ROOT}/mace_family/${filename}" "${RAW}/${filename}"
done
for filename in \
  nequip_sh_c84.json \
  nequip_chorus_final_c84_r16.json \
  nequip_chorus_persistent_c84_r16.json; do
  cp "${ROOT}/nequip_family/${filename}" "${RAW}/${filename}"
done
for filename in dpa4_c32_fp32_compiled.json dpa4_c48_fp32_compiled.json; do
  cp "${ROOT}/dpa4_family/${filename}" "${RAW}/${filename}"
done
for filename in tece_c36_openeq.json tece_c48_openeq.json; do
  cp "${ROOT}/tece_family/${filename}" "${RAW}/${filename}"
done

"${PYTHON_BIN}" - "${RAW}" <<'PY'
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
expected_sizes = {128, 256, 512, 1024, 2048, 4096}
for path in sorted(raw.glob("*.json")):
    payload = json.loads(path.read_text())
    protocol = payload["protocol"]
    if protocol.get("dtype") != "float32" or protocol.get("tf32") is not False:
        raise RuntimeError(f"{path.name}: not strict FP32")
    if "A100" not in str(protocol.get("gpu", "")):
        raise RuntimeError(f"{path.name}: not measured on A100")
    for task in ("inference", "train"):
        sizes = {int(row["natoms"]) for row in payload["rows"] if row["task"] == task}
        if sizes != expected_sizes:
            raise RuntimeError(f"{path.name}: incomplete {task} sizes {sorted(sizes)}")
    invalid = [
        (row["task"], row["natoms"], row.get("status"))
        for row in payload["rows"]
        if row.get("status") not in {"ok", "oom"}
    ]
    if invalid:
        raise RuntimeError(f"{path.name}: invalid benchmark rows {invalid}")
    if path.name.startswith("nequip_"):
        oeq = payload.get("openequivariance", {})
        if not oeq.get("precompiled_extension") or int(oeq.get("module_count", 0)) <= 0:
            raise RuntimeError(f"{path.name}: optimized OpenEquivariance audit failed")
    if path.name.startswith("dpa4_"):
        if payload.get("dpa_descriptor_amp") is not False:
            raise RuntimeError(f"{path.name}: DPA descriptor AMP was not disabled")
        if not str(protocol.get("torch", "")).startswith("2.11."):
            raise RuntimeError(f"{path.name}: DPA runtime is not torch 2.11")
print(f"validated {len(list(raw.glob('*.json')))} model payloads")
PY

"${PYTHON_BIN}" "${REPO}/benchmarks/paper/scripts/throughput/summarize_a100_scaling.py" \
  --source "${RAW}" --output "${OUT}"
touch "${OUT}/DONE"
