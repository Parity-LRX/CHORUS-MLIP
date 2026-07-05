#!/usr/bin/env bash
set -euo pipefail

# Recompute convergence summaries from finished or in-progress apple-to-apple
# MD17 training logs. This is log-only analysis and does not launch training.

PY="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
REPO="${MACE_ICTC_REPO:-/home/ylzhang/lrx/MACE-ICTC}"
TARGET_EPOCH="${TARGET_EPOCH:-300}"

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 /path/to/benchmark_out_dir [more_out_dirs...]" >&2
  exit 2
fi

cd "${REPO}"

for out in "$@"; do
  log_dir="${out}/logs"
  if [[ ! -d "${log_dir}" ]]; then
    echo "missing log directory: ${log_dir}" >&2
    exit 3
  fi
  analysis_dir="${out}/convergence_analysis"
  "${PY}" benchmarks/paper/scripts/training/analyze_md17_convergence.py \
    "${log_dir}" \
    --out-dir "${analysis_dir}" \
    --target-epoch "${TARGET_EPOCH}" \
    --plots
  echo "wrote ${analysis_dir}"
done
