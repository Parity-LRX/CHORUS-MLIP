#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ylzhang/CHORUS-MLIP-attention-test}"
PYTHON_BIN="${PYTHON_BIN:-/home/ylzhang/micromamba/envs/FSCETP/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/tmp/mace_torch_0_3_16}"
ROOT="${ROOT:-/home/ylzhang/chorus_runs/large_scale_main_20260724/rank32_pilot}"
STATUS="${STATUS:-${ROOT}/queue_status.log}"

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
mkdir -p "${ROOT}"

mark() {
  printf '%s %s\n' "$1" "$(date -Is)" | tee -a "${STATUS}"
}

run_rank32() {
  local tag="$1"
  local data_dir="$2"
  local out_dir="$3"
  local avg_neighbors="$4"
  local e0_keys="$5"
  local e0_values="$6"
  local epochs="$7"
  local max_steps="$8"
  local keep_checkpoints="$9"
  local batch_size="${10:-16}"

  local checkpoint="${out_dir}/checkpoints/${tag}_ictc_phase_full_l_nonlinear_makefx_seed20260616_epochs${epochs}.pth"
  if [[ -f "${out_dir}/DONE" ]]; then
    mark "SKIP_${tag}_DONE"
  else
    mkdir -p "${out_dir}/checkpoints" "${out_dir}/logs"
    mark "START_${tag}"
    "${PYTHON_BIN}" -m chorus.cli.train \
      --data-dir "${data_dir}" --train-prefix train --val-prefix val \
      --channels 128 --lmax 2 --max-ell 2 \
      --num-interaction 2 --correlation 3 \
      --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
      --first-layer-self-connection --mace-compatible-random-init \
      --readout-hidden-channels 64 \
      --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
      --max-radius 5.0 --avg-num-neighbors "${avg_neighbors}" \
      --atomic-energy-keys "${e0_keys}" --atomic-energy-values="${e0_values}" \
      --scaling std_scaling \
      --epochs "${epochs}" --max-steps "${max_steps}" \
      --batch-size "${batch_size}" \
      --dtype float32 --device cuda --num-workers 0 \
      --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
      --lr 0.001 --min-lr 1e-6 --lr-scheduler cosine \
      --optimizer adamw --optimizer-param-groups mace \
      --weight-decay 5e-7 --amsgrad --max-grad-norm 10 \
      --phase-hidden-channels 32 --phase-scale-init 0.05 \
      --phase-density-rank 32 \
      --train-makefx-compile --require-train-makefx-compile \
      --makefx-buckets 4 --makefx-max-slots 8 \
      --pad-nodes-to-max --pad-edges-to-max \
      --seed 20260616 --log-interval 20 \
      --keep-checkpoints "${keep_checkpoints}" \
      --phase-mode final-full-l-residual \
      --phase-amplitude softplus --phase-coefficient polar \
      --phase-context content --phase-density-pairs full-nonlinear \
      --phase-normalization avg-neighbors \
      --phase-placement pre-product-full-l --phase-scope final \
      --checkpoint "${checkpoint}" >"${out_dir}/logs/train.log" 2>&1
    touch "${out_dir}/DONE"
    mark "DONE_${tag}"
  fi

  local eval_dir="${out_dir}/validation_force_selected_eval"
  if [[ -f "${eval_dir}/DONE" ]]; then
    mark "SKIP_${tag}_EVAL_DONE"
    return
  fi
  mkdir -p "${eval_dir}"
  local selected
  selected="$("${PYTHON_BIN}" - \
    "${out_dir}" "${eval_dir}/selection.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
destination = Path(sys.argv[2])
with (run_dir / "checkpoints" / "loss.csv").open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row["kind"] == "epoch"]
if not rows:
    raise RuntimeError(f"no validation rows in {run_dir}")
best = min(rows, key=lambda row: (float(row["val_force_mae"]), int(row["step"])))
epoch, step = int(best["epoch"]), int(best["step"])
candidates = list((run_dir / "checkpoints").glob(f"*.e{epoch}s{step}.pth"))
if len(candidates) != 1:
    raise RuntimeError(
        f"expected one checkpoint for epoch={epoch}, step={step}, got {candidates}"
    )
result = {
    "selection_rule": "minimum validation Force MAE; earliest step breaks ties",
    "test_used_for_selection": False,
    "selected_checkpoint": str(candidates[0]),
    "step": step,
    "epoch": epoch,
    "validation": {
        key: float(best[key])
        for key in (
            "val_energy_mae",
            "val_energy_rmse",
            "val_force_mae",
            "val_force_rmse",
        )
    },
    "validation_checkpoint_count": len(rows),
}
destination.write_text(json.dumps(result, indent=2) + "\n")
print(candidates[0])
PY
)"
  mark "START_${tag}_TEST_EVAL"
  "${PYTHON_BIN}" -m chorus.cli.train \
    --data-dir "${data_dir}" --train-prefix train --val-prefix test \
    --channels 128 --lmax 2 --max-ell 2 \
    --num-interaction 2 --correlation 3 \
    --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
    --first-layer-self-connection --mace-compatible-random-init \
    --readout-hidden-channels 64 \
    --function-type bessel --num-basis 8 --polynomial-cutoff-p 6 \
    --max-radius 5.0 --avg-num-neighbors "${avg_neighbors}" \
    --atomic-energy-keys "${e0_keys}" --atomic-energy-values="${e0_values}" \
    --scaling std_scaling \
    --epochs 1 --batch-size "${batch_size}" \
    --dtype float32 --device cuda --num-workers 0 \
    --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
    --phase-hidden-channels 32 --phase-scale-init 0.05 \
    --phase-density-rank 32 \
    --seed 20260616 \
    --phase-mode final-full-l-residual \
    --phase-amplitude softplus --phase-coefficient polar \
    --phase-context content --phase-density-pairs full-nonlinear \
    --phase-normalization avg-neighbors \
    --phase-placement pre-product-full-l --phase-scope final \
    --resume-checkpoint "${selected}" --eval-only \
    --checkpoint "${eval_dir}/unused.pth" >"${eval_dir}/test.log" 2>&1
  "${PYTHON_BIN}" - \
    "${eval_dir}/selection.json" "${eval_dir}/test.log" \
    "${eval_dir}/metrics.json" <<'PY'
import json
import re
import sys
from pathlib import Path

selection_path, log_path, destination = map(Path, sys.argv[1:])
text = log_path.read_text(errors="replace")
patterns = {
    "energy_mae_ev_per_atom": r"MAE:\s+Fmae=[0-9.eE+-]+\s+eV/A\s+Emae=([0-9.eE+-]+)\s+eV/atom",
    "energy_rmse_ev_per_atom": r"RMSE:\s+Frmse=[0-9.eE+-]+\s+eV/A\s+Ermse=([0-9.eE+-]+)\s+eV/atom",
    "force_mae_ev_per_angstrom": r"MAE:\s+Fmae=([0-9.eE+-]+)\s+eV/A",
    "force_rmse_ev_per_angstrom": r"RMSE:\s+Frmse=([0-9.eE+-]+)\s+eV/A",
}
test = {}
for key, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"missing {key} in {log_path}")
    test[key] = float(matches[-1])
result = json.loads(selection_path.read_text())
result["test"] = test
destination.write_text(json.dumps(result, indent=2) + "\n")
PY
  touch "${eval_dir}/DONE"
  mark "DONE_${tag}_TEST_EVAL"
}

SYSTEMS="${SYSTEMS:-t1x mal sti dia bucky}"
for system in ${SYSTEMS}; do
  case "${system}" in
    t1x)
      run_rank32 \
        t1x_c128_l2_corr3_rank32 \
        /home/ylzhang/lrx/transition1x/chorus_reaction_id_50k_seed20260616 \
        "${ROOT}/t1x_c128_l2_corr3_rank32" \
        10.71685543435131 \
        1,6,7,8 \
        -13.622227668762207,-1029.4130859375,-1484.87109375,-2041.839599609375 \
        32 100000 40 16
      ;;
    mal)
      run_rank32 \
        xxmd_mal_c128_l2_corr3_rank32 \
        /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/mal \
        "${ROOT}/xxmd_mal_c128_l2_corr3_rank32" \
        7.99384126984127 \
        1,6,8 \
        -1001.3306884765625,-750.9979858398438,-500.66534423828125 \
        52 45000 100 16
      ;;
    sti)
      run_rank32 \
        xxmd_sti_c128_l2_corr3_rank32 \
        /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/sti \
        "${ROOT}/xxmd_sti_c128_l2_corr3_rank32" \
        16.62877403846154 \
        1,6 \
        -518.6243286132812,-605.061767578125 \
        57 45000 100 16
      ;;
    dia)
      run_rank32 \
        xxmd_dia_c128_l2_corr3_rank32 \
        /home/ylzhang/lrx/xxmd/processed_dft_temporal_r5/dia \
        "${ROOT}/xxmd_dia_c128_l2_corr3_rank32" \
        14.036274193548387 \
        1,6,16 \
        -1531.021284830928,-1913.77660603866,-382.755321207732 \
        59 45000 100 16
      ;;
    bucky)
      run_rank32 \
        md22_buckyball_catcher_c128_l2_corr3_rank32 \
        /home/ylzhang/lrx/md22/chorus_lowdata600_20260720/processed \
        "${ROOT}/md22_buckyball_catcher_c128_l2_corr3_rank32" \
        30.3929 \
        1,6 \
        -230.09867339,-986.13717166 \
        300 45000 300 4
      ;;
    *)
      echo "unknown system: ${system}" >&2
      exit 2
      ;;
  esac
done

mark "ALL_REQUESTED_RANK32_DONE systems=${SYSTEMS}"
