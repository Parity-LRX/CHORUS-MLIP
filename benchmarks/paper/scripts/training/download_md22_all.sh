#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-/home/ylzhang/lrx/md22/all_raw}"
mkdir -p "${OUT}"

files=(
  md22_Ac-Ala3-NHMe.npz
  md22_DHA.npz
  md22_stachyose.npz
  md22_AT-AT.npz
  md22_AT-AT-CG-CG.npz
  md22_buckyball-catcher.npz
  md22_double-walled_nanotube.npz
)

download_one() {
  local name="$1"
  local target="${OUT}/${name}"
  local direct="https://www.quantum-machine.org/gdml/data/npz/${name}"
  local proxy="https://sgdml.org/secure_proxy.php?file=repo/datasets/${name}"
  if [[ -s "${target}" ]]; then
    echo "EXISTS ${name}"
    return
  fi
  if ! curl -fL --retry 5 --retry-delay 5 -C - -o "${target}" "${direct}"; then
    rm -f "${target}"
    curl -fL --retry 5 --retry-delay 5 -o "${target}" "${proxy}"
  fi
  echo "DOWNLOADED ${name}"
}
export OUT
export -f download_one
printf '%s\n' "${files[@]}" | xargs -n1 -P3 bash -c 'download_one "$1"' _

python_bin="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
"${python_bin}" - "${OUT}" <<'PY'
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
for path in sorted(root.glob("md22_*.npz")):
    with np.load(path) as data:
        required = {"R", "z", "F", "E"}
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"{path}: missing {sorted(missing)}")
        print(path.name, data["R"].shape, data["z"].shape, data["F"].shape, data["E"].shape)
PY

sha256sum "${OUT}"/md22_*.npz >"${OUT}/SHA256SUMS"
echo "ALL_OK $(date)"
