#!/usr/bin/env bash
#SBATCH --job-name=chorus-native-cache-repair
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/logs/cache_repair_%j.log
set -euo pipefail
ROOT=/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017
NAME=ictc_coherent_multiout_v2_128_1152_4480_h1_a1
CACHE="$ROOT/cache/native_v1_1/extensions/$NAME"
ps -u rxlin -ww -o pid,ppid,etime,args
ls -la "$CACHE"
for PID in $(pgrep -u rxlin -x ninja || true); do
  if [[ "$(readlink /proc/$PID/cwd)" == "$CACHE" ]]; then
    printf 'Refusing repair: active builder %s owns the exact cache directory\n' "$PID"
    exit 2
  fi
done
test -f "$CACHE/lock"
test -f "$CACHE/build.ninja"
export PATH=/public/home/sps-xia/rxlin/.conda/envs/mff/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/public/home/sps-xia/rxlin/.conda/envs/mff/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
# Finish the exact build left by cancelled 91944 before waking its waiters.
# No source, weights or compiler options are replaced.
ninja -C "$CACHE" -v -j4
test -s "$CACHE/$NAME.so"
mv "$CACHE/lock" "$CACHE/lock.abandoned_91944"
printf 'DONE: existing waiters may import the completed original extension\n'
