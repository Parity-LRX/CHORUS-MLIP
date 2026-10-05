#!/usr/bin/env bash
#SBATCH --job-name=lips-restart-stability
#SBATCH --partition=GPU-A100
#SBATCH --nodelist=XHPC-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%1
#SBATCH --output=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/lips_restart_stability_%A_%a.out
#SBATCH --error=/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922/slurm/lips_restart_stability_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/public/home/sps-xia/rxlin/chorus_transport_lips_water_20260922}"
BUILD_ROOT="${ROOT}/deploy/mfftorch_lammps_20260924"
LMP="${BUILD_ROOT}/build-mfftorch/lmp"
SOURCE="${ROOT}/md/carnet_protocol_lammps/mace_family/lips/persistent/seed20260616"
MODEL="${BUILD_ROOT}/models/lips_persistent_dynamic.pt2"
BASE="${ROOT}/diagnostics/lips_instability_20260926/restart_controls"

time_steps=(0.001 0.0005 0.001 0.0005)
run_steps=(10000 20000 10000 20000)
dampings=(0.02 0.02 0.1 0.1)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
dt="${time_steps[${task_id}]}"
nsteps="${run_steps[${task_id}]}"
damping="${dampings[${task_id}]}"
tag="dt${dt}_damp${damping}"
work="${BASE}/${tag}"
mkdir -p "${work}"

cat > "${work}/in.restart_control" <<EOF
units metal
atom_style atomic
boundary p p p
atom_modify map yes

read_restart ${SOURCE}/restart.95000.bin
reset_timestep 95000
timestep ${dt}

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
pair_style mff/torch 5 cuda
pair_coeff * * ${MODEL} Li P S

thermo_style custom step atoms temp pe ke etotal press fmax
thermo 10
thermo_modify flush yes
dump dtraj all custom 100 ${work}/trajectory.lammpstrj id type xu yu zu vx vy vz
dump_modify dtraj sort id element Li P S
restart 5000 ${work}/restart.*.bin

fix thermostat all nvt temp 520 520 ${damping} tchain 1
run ${nsteps}
write_data ${work}/final_state.data nocoeff
print "LIPS_RESTART_CONTROL_DONE tag=${tag} step=\$(step)"
EOF

torch_lib="/public/home/sps-xia/rxlin/.conda/envs/mff/lib/python3.11/site-packages/torch/lib"
nvidia_lib_dirs="$(find /public/home/sps-xia/rxlin/.conda/envs/mff/lib/python3.11/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
export LD_LIBRARY_PATH="${torch_lib}:${nvidia_lib_dirs}:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export OMP_NUM_THREADS=4

"${LMP}" -in "${work}/in.restart_control" > "${work}/lammps.log" 2>&1
touch "${work}/DONE"
echo "LIPS_RESTART_STABILITY_DONE ${tag} $(date -Is)"
