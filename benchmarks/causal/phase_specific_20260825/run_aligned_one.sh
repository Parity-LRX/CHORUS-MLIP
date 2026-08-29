#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:?set REPO to the phase-specific deployment tree}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python}"
MACE_TORCH_PATH="${MACE_TORCH_PATH:-/public/home/sps-xia/rxlin/ablation_core_vs_periphery_run/mace_torch_0_3_16}"
DATA_ROOT="${DATA_ROOT:-/public/home/sps-xia/rxlin/nequip_chorus_data/extended_20260728}"
SOLID_DATA_ROOT="${SOLID_DATA_ROOT:-/public/home/sps-xia/rxlin/chorus_solid_data/ms25_20260818/prepared}"
PRIMARY_ROOT="${PRIMARY_ROOT:-/public/home/sps-xia/rxlin/chorus_u1_causal_20260824/runs}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
MAX_STEPS="${MAX_STEPS:-45000}"
TOOLS="${REPO}/benchmarks/causal/phase_specific_20260825"
PHASE_SCOPE="${PHASE_SCOPE:-final}"
PROTOCOL_PATH="${PROTOCOL_PATH:-${TOOLS}/protocol.json}"
SOURCE_STATE_PATH="${SOURCE_STATE_PATH:-${TOOLS}/SOURCE_STATE}"

case "${DATASET}" in
  mal)
    data_dir="${DATA_ROOT}/xxmd/processed_dft_temporal_r5/mal"
    avg_neighbors="7.99384126984127"
    e0_keys="1,6,8"
    e0_values="-1001.3306884765625,-750.9979858398438,-500.66534423828125"
    cutoff="5.0"; epochs="52"; batch_size="16"; tests=(test)
    ;;
  bucky)
    data_dir="${DATA_ROOT}/md22/chorus_lowdata600_20260720/processed"
    avg_neighbors="30.3929"
    e0_keys="1,6"
    e0_values="-230.09867339,-986.13717166"
    cutoff="5.0"; epochs="300"; batch_size="4"; tests=(test)
    ;;
  3bpa)
    data_dir="${DATA_ROOT}/3bpa/standard_450_50_seed20260616_r5"
    # shellcheck disable=SC1090
    source "${data_dir}/training.env"
    avg_neighbors="${AVG_NEIGHBORS}"
    e0_keys="${E0_KEYS}"
    e0_values="${E0_VALUES}"
    cutoff="5.0"; epochs="1600"; batch_size="16"; tests=(300K 600K 1200K)
    ;;
  zro)
    data_dir="${SOLID_DATA_ROOT}/zr-o/set1"
    readarray -t zro_meta < <("${PYTHON_BIN}" - "${data_dir}/metadata.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(x["average_train_neighbours"])
print(",".join(map(str, x["atomic_numbers"])))
print(",".join(f"{v:.17g}" for v in x["element_reference_energies_ev"]))
print(x["cutoff_angstrom"])
PY
    )
    avg_neighbors="${zro_meta[0]}"
    e0_keys="${zro_meta[1]}"
    e0_values="${zro_meta[2]}"
    cutoff="${zro_meta[3]}"; epochs="180"; batch_size="4"; tests=(test)
    ;;
  *) echo "unknown DATASET=${DATASET}" >&2; exit 2 ;;
esac

for required in processed_train.h5 processed_val.h5; do
  [[ -f "${data_dir}/${required}" ]] || { echo "missing ${data_dir}/${required}" >&2; exit 3; }
done

out="${RUN_ROOT}/${DATASET}/seed${SEED}/aligned"
checkpoint="${out}/checkpoints/model.pth"
recovery="${out}/checkpoints/model.recovery.pth"
train_log="${out}/logs/train.log"
protocol="${PROTOCOL_PATH}"
mkdir -p "${out}/checkpoints" "${out}/logs" "${out}/eval"
rm -f "${out}/FAILED"
touch "${out}/RUNNING"

mark() { printf '%s %s\n' "$1" "$(date --iso-8601=seconds)" | tee -a "${out}/status.log"; }
on_exit() {
  rc=$?; trap - EXIT; rm -f "${out}/RUNNING"
  if (( rc != 0 )); then printf 'exit_code=%s\n' "${rc}" >"${out}/FAILED"; mark "FAILED exit=${rc}"; fi
  exit "${rc}"
}
trap on_exit EXIT

export PYTHONPATH="${REPO}:${MACE_TORCH_PATH}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

"${PYTHON_BIN}" - "${protocol}" "${out}/run_manifest.json" <<PY
import hashlib, json, pathlib, platform
p = pathlib.Path("${protocol}")
source = pathlib.Path("${TOOLS}/SOURCE_STATE")
payload = {
    "dataset": "${DATASET}", "mode": "aligned", "seed": int("${SEED}"),
    "max_steps": int("${MAX_STEPS}"), "protocol": str(p),
    "protocol_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "source_state": pathlib.Path("${SOURCE_STATE_PATH}").read_text().strip(),
    "phase_scope": "${PHASE_SCOPE}", "host": platform.node(),
    "paired_full_run": "${PRIMARY_ROOT}/${DATASET}/seed${SEED}/full",
}
pathlib.Path("${out}/run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

phase_args=(
  --phase-mode final-full-l-residual --phase-amplitude softplus
  --phase-coefficient aligned --phase-context content --phase-density-pairs full
  --phase-normalization avg-neighbors --phase-placement pre-product-full-l
  --phase-scope "${PHASE_SCOPE}"
)
common=(
  --data-dir "${data_dir}" --train-prefix train --val-prefix val
  --channels 64 --lmax 1 --max-ell 2 --num-interaction 2 --correlation 2
  --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg
  --first-layer-self-connection --mace-compatible-random-init
  --readout-hidden-channels 64 --function-type bessel --num-basis 8
  --polynomial-cutoff-p 6 --max-radius "${cutoff}"
  --avg-num-neighbors "${avg_neighbors}" --atomic-energy-keys "${e0_keys}"
  --atomic-energy-values="${e0_values}" --scaling std_scaling
  --epochs "${epochs}" --max-steps "${MAX_STEPS}" --batch-size "${batch_size}"
  --dtype float32 --device cuda --num-workers 0 --loss mse
  --energy-weight 1 --force-weight 100 --stress-weight 0 --lr 0.001
  --min-lr 1e-6 --lr-scheduler cosine --optimizer adamw
  --optimizer-param-groups mace --weight-decay 5e-7 --amsgrad --max-grad-norm 10
  --phase-hidden-channels 32 --phase-scale-init 0.05 --phase-density-rank 8
  --train-makefx-compile --require-train-makefx-compile
  --makefx-buckets 4 --makefx-max-slots 8 --pad-nodes-to-max --pad-edges-to-max
  --checkpoint-selection-metric force_mae --checkpoint-interval-steps 1000
  --checkpoint-state-source raw --ema-decay 0 --seed "${SEED}" --log-interval 20
  "${phase_args[@]}"
)

write_command() {
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf '%q ' env PYTHONPATH="${PYTHONPATH}" NVIDIA_TF32_OVERRIDE=0 \
      TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 "${PYTHON_BIN}" -m chorus.cli.train "$@"
    printf '\n'
  } >"${out}/command.sh"
  chmod +x "${out}/command.sh"
}

if [[ ! -f "${out}/TRAIN_DONE" ]]; then
  resume=()
  if [[ -f "${recovery}" ]]; then
    resume=(--resume-checkpoint "${recovery}" --resume-training-state)
    mark "RESUME_FROM_RECOVERY"
  fi
  write_command "${common[@]}" "${resume[@]}" --checkpoint "${checkpoint}"
  mark "TRAIN_START"
  "${PYTHON_BIN}" -m chorus.cli.train "${common[@]}" "${resume[@]}" \
    --checkpoint "${checkpoint}" >"${train_log}" 2>&1
  touch "${out}/TRAIN_DONE"; mark "TRAIN_DONE"
fi
[[ -f "${checkpoint}" ]] || { echo "missing selected checkpoint ${checkpoint}" >&2; exit 4; }

eval_one() {
  local test_name="$1" eval_dir="${out}/eval/$1" eval_data="${data_dir}" val_prefix="test"
  mkdir -p "${eval_dir}"
  if [[ "${DATASET}" == "3bpa" ]]; then
    eval_data="${eval_dir}/data"; mkdir -p "${eval_data}"
    ln -sfn "${data_dir}/processed_train.h5" "${eval_data}/processed_train.h5"
    [[ ! -f "${data_dir}/processed_train.h5.counts.npz" ]] || \
      ln -sfn "${data_dir}/processed_train.h5.counts.npz" "${eval_data}/processed_train.h5.counts.npz"
    ln -sfn "${data_dir}/processed_test_${test_name}.h5" "${eval_data}/processed_test.h5"
    [[ ! -f "${data_dir}/processed_test_${test_name}.h5.counts.npz" ]] || \
      ln -sfn "${data_dir}/processed_test_${test_name}.h5.counts.npz" "${eval_data}/processed_test.h5.counts.npz"
  fi
  if [[ ! -f "${eval_dir}/DONE" ]]; then
    mark "EVAL_START_${test_name}"
    "${PYTHON_BIN}" -m chorus.cli.train \
      --data-dir "${eval_data}" --train-prefix train --val-prefix "${val_prefix}" \
      --channels 64 --lmax 1 --max-ell 2 --num-interaction 2 --correlation 2 \
      --product-backend ictd-bridge-u --angular-basis ictd --use-reduced-cg \
      --first-layer-self-connection --mace-compatible-random-init \
      --readout-hidden-channels 64 --function-type bessel --num-basis 8 \
      --polynomial-cutoff-p 6 --max-radius "${cutoff}" \
      --avg-num-neighbors "${avg_neighbors}" --atomic-energy-keys "${e0_keys}" \
      --atomic-energy-values="${e0_values}" --scaling std_scaling --epochs 1 \
      --batch-size "${batch_size}" --dtype float32 --device cuda --num-workers 0 \
      --loss mse --energy-weight 1 --force-weight 100 --stress-weight 0 \
      --phase-hidden-channels 32 --phase-scale-init 0.05 --phase-density-rank 8 \
      --checkpoint-state-source raw --seed "${SEED}" "${phase_args[@]}" \
      --resume-checkpoint "${checkpoint}" --eval-only \
      --checkpoint "${eval_dir}/unused.pth" >"${eval_dir}/test.log" 2>&1
    touch "${eval_dir}/DONE"; mark "EVAL_DONE_${test_name}"
  fi
}
for test_name in "${tests[@]}"; do eval_one "${test_name}"; done

"${PYTHON_BIN}" "${REPO}/benchmarks/causal/u1_hermitian_20260824/parse_run.py" \
  --run-dir "${out}" --dataset "${DATASET}" --mode aligned --seed "${SEED}" \
  --output "${out}/metrics.json"

"${PYTHON_BIN}" - "${out}/metrics.json" "${PRIMARY_ROOT}/${DATASET}/seed${SEED}/full/metrics.json" <<'PY'
import json, pathlib, sys
aligned = json.loads(pathlib.Path(sys.argv[1]).read_text())
full = json.loads(pathlib.Path(sys.argv[2]).read_text())
if aligned["parameter_count"] != full["parameter_count"]:
    raise SystemExit(
        f"parameter mismatch aligned={aligned['parameter_count']} full={full['parameter_count']}"
    )
PY

touch "${out}/DONE"; rm -f "${out}/RUNNING" "${out}/FAILED"; trap - EXIT; mark "DONE"
