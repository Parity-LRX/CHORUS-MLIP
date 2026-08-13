#!/usr/bin/env bash
set -euo pipefail

# Matched many-element CHORUS comparison for the scalable Hermitian-density
# writeback.  The onehot-full reference uses the same protocol; only
# PHASE_DENSITY_SPECIES_MODE, SPECIES_EMBEDDING_DIM, and SPECIES_RANK differ.

chorus_repo="${CHORUS_REPO:-/public/home/sps-xia/rxlin/CHORUS-MLIP-main-720c6ee}"
python_bin="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
mace_torch_path="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
data_dir="${DATA_DIR:-/public/home/sps-xia/rxlin/data/mptrj_proc45_full}"
e0_csv="${E0_CSV:-${data_dir}/fitted_E0.csv}"
out_root="${OUT_ROOT:?set OUT_ROOT to a new result directory}"

species_mode="${PHASE_DENSITY_SPECIES_MODE:-embedded-lowrank}"
embedding_dim="${SPECIES_EMBEDDING_DIM:-16}"
species_rank="${SPECIES_RANK:-16}"
seed="${SEED:-20260616}"

if [[ ! -f "${data_dir}/processed_train.h5" ]]; then
  echo "missing ${data_dir}/processed_train.h5" >&2
  exit 2
fi
if [[ ! -f "${data_dir}/processed_valsub.h5" ]]; then
  echo "missing ${data_dir}/processed_valsub.h5" >&2
  exit 2
fi
if [[ ! -f "${e0_csv}" ]]; then
  echo "missing fitted atomic energies: ${e0_csv}" >&2
  exit 2
fi

readarray -t chemistry < <(
  "${python_bin}" - "${e0_csv}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
atomic_numbers = sorted(int(row["Atom"]) for row in rows)
if not atomic_numbers or len(atomic_numbers) != len(set(atomic_numbers)):
    raise SystemExit("fitted E0 CSV must contain a unique, non-empty Atom column")
e0 = {int(row["Atom"]): float(row["E0"]) for row in rows}
print(",".join(str(z) for z in atomic_numbers))
print(",".join(repr(e0[z]) for z in atomic_numbers))
PY
)
if [[ "${#chemistry[@]}" -ne 2 ]]; then
  echo "failed to infer atomic numbers and fitted E0 values" >&2
  exit 2
fi

run_name="chorus_persistent_l4_c64_${species_mode}_d${embedding_dim}_r${species_rank}"
run_dir="${out_root}/mptraj_full_chorus/seed${seed}/${run_name}"
checkpoint="${run_dir}/model.pth"
if [[ -e "${checkpoint}" ]]; then
  echo "refusing to overwrite existing checkpoint: ${checkpoint}" >&2
  exit 2
fi
mkdir -p "${run_dir}"

args=(
  --data-dir "${data_dir}" --train-prefix train --val-prefix valsub
  --checkpoint "${checkpoint}"
  --channels 64 --lmax 4 --max-ell 4
  --num-interaction 3 --correlation 2
  --product-backend ictd-bridge-u --angular-basis ictd
  --use-reduced-cg --first-layer-self-connection
  --readout-hidden-channels 64
  --phase-hidden-channels 32 --phase-scale-init 0.05
  --phase-density-rank 8 --phase-mode final-full-l-residual
  --phase-amplitude softplus --phase-coefficient polar
  --phase-context content --phase-density-pairs full
  --phase-placement pre-product-full-l --phase-scope persistent
  --phase-density-species-mode "${species_mode}"
  --phase-density-species-embedding-dim "${embedding_dim}"
  --phase-density-species-rank "${species_rank}"
  --function-type bessel --num-basis 8 --polynomial-cutoff-p 6
  --max-radius 4.5 --avg-num-neighbors 25.5799
  --atomic-energy-keys "${chemistry[0]}"
  "--atomic-energy-values=${chemistry[1]}"
  --scaling std_scaling
  --atomic-inter-scale 0.783438899813
  --atomic-inter-shift 0.154031775279
  --epochs 4 --max-steps 100000
  --batch-size 1 --num-workers 2 --gradient-accumulation-steps 2
  --dtype float32 --device cuda
  --loss mse --energy-weight 1000 --force-weight 100 --stress-weight 0
  --lr 0.001 --min-lr 1e-6
  --lr-scheduler exp --lr-scheduler-gamma 0.9993
  --optimizer adamw --optimizer-param-groups mace
  --weight-decay 5e-7 --amsgrad --max-grad-norm 10
  --seed "${seed}" --keep-checkpoints 20
  --checkpoint-interval-steps 10000 --validation-interval-steps 5000
  --ema-decay 0 --ema-start-step 0 --checkpoint-state-source raw
  --log-interval 200 --swa-start-step 80000 --swa-lr 1e-4
  --train-makefx-compile --makefx-buckets 16 --makefx-max-slots 20
  --pad-nodes-to-max --pad-edges-to-max
)

printf 'cd %q\n' "${chorus_repo}" >"${run_dir}/command.sh"
printf '%q ' env "PYTHONPATH=${chorus_repo}:${mace_torch_path}" \
  "${python_bin}" -m chorus.cli.train "${args[@]}" \
  >>"${run_dir}/command.sh"
printf '\n' >>"${run_dir}/command.sh"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  cat "${run_dir}/command.sh"
  exit 0
fi

(
  cd "${chorus_repo}"
  env PYTHONPATH="${chorus_repo}:${mace_torch_path}" \
    "${python_bin}" -m chorus.cli.train "${args[@]}"
) >"${run_dir}/train.log" 2>&1

# model.pth tracks the best validation point across all 5k-step validations.
(
  cd "${chorus_repo}"
  env PYTHONPATH="${chorus_repo}:${mace_torch_path}" \
    "${python_bin}" -m chorus.cli.train "${args[@]}" \
    --resume-checkpoint "${checkpoint}" --eval-only
) >"${run_dir}/eval_best_valsub.log" 2>&1
