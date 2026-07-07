#!/bin/bash
# Full deployed-MD throughput sweep for the figure additions, synthetic rho=0.04 system.
# Series: dispevery (path-1 MBD matrix-free, every step) | disprespa (MBD + rRESPA K=20)
#         | c6 (pairwise-C6, every step). Pure-torch env (path-1 MBD C++ autograd needs no cueq vars).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MACE_ICTC_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
LRX="${LRX:-/home/ylzhang/lrx}"
PY="${PY:-python3}"
LMP="${LMP:-lmp}"
DIR="${LR_FIG_DIR:-${LRX}/fig_respa_c6}"
mkdir -p "$DIR"
ISO="${LR_ISO_PT2:-${LRX}/mff_md_work/disp_path1_mbdslq_c64_cut9_iso.pt2}"
C6="${LR_C6_PT2:-${LRX}/mff_md_work/disp_c6_c64_cut9.pt2}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$($PY -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

MAIN_CUT=5.0; DISP_CUT=9.0; RHO=0.04
emit_in () { # 1=core 2=runstyle 3=dt 4=nsteps 5=thermo 6=N 7=outfile
  local N=$6
  local BOX; BOX=$($PY -c "c=max($MAIN_CUT,$DISP_CUT);print(max(2.0*c+0.1,($N/$RHO)**(1.0/3.0)))")
  local N1=$((N/4)) N2=$((N/4)) N3=$((N/4)) N4=$((N-3*(N/4)))
  cat > "$7" <<LMPIN
units metal
atom_style atomic
atom_modify map yes
boundary p p p
region box block 0 ${BOX} 0 ${BOX} 0 ${BOX}
create_box 4 box
create_atoms 1 random ${N1} 12345 box
create_atoms 2 random ${N2} 12346 box
create_atoms 3 random ${N3} 12347 box
create_atoms 4 random ${N4} 12348 box
mass 1 1.008
mass 2 12.011
mass 3 14.007
mass 4 15.999
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
pair_style mff/torch ${MAIN_CUT} cuda dispersion ${DISP_CUT}
pair_coeff * * ${1} H C N O
velocity all create 300.0 42 mom yes dist gaussian
fix 1 all nve
${2}
timestep ${3}
thermo ${5}
thermo_style custom step temp pe etotal
run ${4}
LMPIN
}
run_one () { # 1=label 2=core 3=runstyle 4=dt 5=nsteps 6=thermo 7=N 8=innerper
  local IN="$DIR/in.$1" OUT="$DIR/out.$1.log"
  emit_in "$2" "$3" "$4" "$5" "$6" "$7" "$IN"
  "$LMP" -in "$IN" > "$OUT" 2>&1; local rc=$?
  local loop; loop=$(grep -oE "Loop time of [0-9.eE+-]+" "$OUT" | grep -oE "[0-9][0-9.eE+-]*" | head -1)
  local inner=$(( $5 * $8 ))
  local ats="NA"
  if [ -n "${loop:-}" ]; then ats=$($PY -c "print(f'{$7*$inner/$loop:.1f}')"); fi
  echo "RESULT $1 N=$7 rc=$rc loop=${loop:-NONE} inner=$inner atoms_s=$ats"
}
echo "===== SWEEP START $(date) ====="
for N in 128 256 512 1024 2048; do
  run_one "dispevery_N${N}" "$ISO" "run_style verlet"                          "0.0005" 100 25 "$N" 1
  run_one "disprespa_N${N}" "$ISO" "run_style respa 2 20 inner 1 5.0 6.0 outer 2" "0.01" 100 5 "$N" 20
  run_one "c6_N${N}"        "$C6"  "run_style verlet"                          "0.0005" 100 25 "$N" 1
done
echo "===== SWEEP DONE $(date) ====="
touch "$DIR/SWEEP_DONE"
