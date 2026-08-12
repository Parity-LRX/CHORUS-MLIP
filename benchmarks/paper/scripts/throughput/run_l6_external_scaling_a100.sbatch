#!/usr/bin/env bash
#SBATCH --job-name=l6-ext-scale
#SBATCH --partition=GPU-A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%2
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_runs/l6_cross_model_20260812/logs/scale-%A_%a.log

set -euo pipefail

repo="/public/home/sps-xia/rxlin/CHORUS-MLIP-persistent-formal-20260730"
assets="/public/home/sps-xia/rxlin/a100_scaling_assets_20260804"
out="/public/home/sps-xia/rxlin/chorus_runs/l6_cross_model_20260812/raw"
bench="${repo}/benchmarks/paper/scripts/throughput/bench_cross_model_scaling.py"
mkdir -p "${out}" /public/home/sps-xia/rxlin/chorus_runs/l6_cross_model_20260812/logs

index="${SLURM_ARRAY_TASK_ID}"
engine_index="$((index / 8))"
within_engine="$((index % 8))"
size_index="$((within_engine / 2))"
task_index="$((within_engine % 2))"
sizes=(256 512 1024 2048)
tasks=(inference train)
natoms="${sizes[${size_index}]}"
task="${tasks[${task_index}]}"

export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ "${engine_index}" == 0 ]]; then
  label="tece_c36_l6_m2"
  python_bin="/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python"
  tace_repo="${assets}/tace_chorus_benchmark"
  tace_site="${assets}/tace_chorus_venv_overlay/site-packages"
  PYTHONPATH="${tace_site}:${tace_repo}:${repo}:${PYTHONPATH:-}" \
    "${python_bin}" "${bench}" --engine tece --tece-backend openeq \
    --checkpoint "${assets}/tece_c36.ckpt" --sizes "${natoms}" --tasks "${task}" \
    --lmax-override 6 --mmax-override 2 --random-init \
    --output "${out}/${label}_N${natoms}_${task}.json"
else
  label="dpa4_c32_l6_m1"
  python_bin="/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python"
  dpa_site="${assets}/dpa4_torch211_overlay"
  torch_lib="${dpa_site}/torch/lib"
  cuda_libs=""
  while IFS= read -r libdir; do
    cuda_libs="${cuda_libs:+${cuda_libs}:}${libdir}"
  done < <(find "${dpa_site}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
  export LD_LIBRARY_PATH="${assets}/mpi_libs:${torch_lib}:${cuda_libs}:${LD_LIBRARY_PATH:-}"
  export DP_INTRA_OP_PARALLELISM_THREADS=1
  export DP_INTER_OP_PARALLELISM_THREADS=1
  export DP_TF32_INFER=0
  export DP_AMP_INFER=0
  PYTHONPATH="${dpa_site}:${repo}:${PYTHONPATH:-}" \
    "${python_bin}" "${bench}" --engine dpa4 --dpa-amp off \
    --config "${assets}/dpa4/c32/input.json" --sizes "${natoms}" --tasks "${task}" \
    --lmax-override 6 --mmax-override 1 --random-init \
    --output "${out}/${label}_N${natoms}_${task}.json"
fi
