#!/usr/bin/env bash
#SBATCH --job-name=chorus-training-6144-pair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/logs/training_6144_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018
TOOLS="$ROOT/tools_v33"
export SOURCE="$ROOT/source_v20/chorus" HP_SOURCE="$ROOT/source_v20/ictc_o3_hp"
ASSETS=/public/home/sps-xia/rxlin/a100_scaling_assets_20260804
OVERLAY="$ASSETS/dpa4_torch211_overlay"
PY=/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python
NATIVE_LIBS="${LD_LIBRARY_PATH:-}"
CUDA_LIBS=""
while IFS= read -r libdir; do CUDA_LIBS="${CUDA_LIBS:+${CUDA_LIBS}:}$libdir"; done \
  < <(find "$OVERLAY/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
RESULT="$ROOT/results/selected_training_6144/n6144"
mkdir -p "$RESULT"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.0
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
export DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
export CUDA_CACHE_PATH="$ROOT/cache/cuda_driver" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
native_env() {
  export PYTHONPATH="$TOOLS:$SOURCE/benchmarks/hp:$SOURCE:$HP_SOURCE" LD_LIBRARY_PATH="$NATIVE_LIBS"
  CACHE="$ROOT/cache/native"
  export XDG_CACHE_HOME="$CACHE/xdg" TORCH_EXTENSIONS_DIR="$CACHE/extensions"
  export FSCETP_ICTD_CACHE_DIR="$CACHE/ictd" TRITON_CACHE_DIR="$CACHE/triton"
  export PYTHONPYCACHEPREFIX="$CACHE/pycache" TMPDIR="$ROOT/cache/tmp"
}
scontrol show job -d "$SLURM_JOB_ID"
set -x
for ROUND in 0 1; do
  ORDER="fused dpa"
  if [[ "$ROUND" == 1 ]]; then ORDER="dpa fused"; fi
  for VERSION in $ORDER; do
    if [[ "$VERSION" == fused ]]; then
      native_env
      "$PY" "$TOOLS/profile_training_nograph.py" --checkpoint "$ASSETS/persistent_t1x.pth" \
        --atoms 6144 --neighbors 50 --rounds 1 --warmup 2 --repeats 20 --measurement-only \
        --training-species-fusion --radial-coefficient-fusion --phase-coefficient-mixed-fusion \
        --training-mixed-destination --tag "fused_r${ROUND}" --out "$RESULT/fused_r${ROUND}.json"
    else
      export PYTHONPATH="$TOOLS:$OVERLAY:$SOURCE"
      export LD_LIBRARY_PATH="$ASSETS/mpi_libs:$OVERLAY/torch/lib:$CUDA_LIBS:$NATIVE_LIBS"
      CACHE="$ROOT/cache/dpa"
      export XDG_CACHE_HOME="$CACHE/xdg" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
      export TRITON_CACHE_DIR="$CACHE/triton" PYTHONPYCACHEPREFIX="$CACHE/pycache"
      mkdir -p "$CACHE/xdg/torch/kernels"
      /public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python \
        "$TOOLS/profile_training_dpa_nograph.py" --atoms 6144 --neighbors 50 \
        --checkpoint "$ASSETS/dpa4/c32/model.pt" --config "$ASSETS/dpa4/c32/input.json" \
        --repeats 20 --out "$RESULT/dpa_r${ROUND}.json"
    fi
  done
done
native_env
# Qualification is separate from all paired timing, with original full E/F and
# every parameter gradient. Its one diagnostic timed step is not in the report.
"$PY" "$TOOLS/profile_training_nograph.py" --checkpoint "$ASSETS/persistent_t1x.pth" \
  --atoms 6144 --neighbors 50 --rounds 1 --warmup 1 --repeats 1 \
  --training-species-fusion --radial-coefficient-fusion --phase-coefficient-mixed-fusion \
  --training-mixed-destination --reference-tp-chunk-edges 1024 --reference-save-on-cpu \
  --tag qualification --out "$RESULT/qualification.json"
"$PY" "$TOOLS/report_selected_training.py" --root "$ROOT/results/selected_training_6144" \
  --sizes 6144 --qualification-name qualification --out "$ROOT/reports_selected_training_6144"
printf 'DONE\n'
