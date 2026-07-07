#!/bin/bash
# LAMMPS MD throughput driver for the MACE-ICTC long-range C++ path (mode 6: aoti-md).
#   Usage: bench_lammps_md.sh <core.pt2> <natoms> <dispersion_cutoff>
# If <dispersion_cutoff> > 0, adds the `dispersion <cutoff>` keyword (REQUIRED for mbd-slq
# cores; the C++ MBD solver reads the second dispersion edge list). The reciprocal
# (electrostatic) C++ solver auto-activates from the <core>.pt2.json sidecar -- no keyword.
# Builds a random periodic box, runs a short NVE MD, prints LAMMPS "Loop time"/"Performance".
set -e
PT2="${1:?core.pt2 path}"
NATOMS="${2:-256}"
DISP_CUT="${3:-0}"
STEPS="${STEPS:-100}"
MAIN_CUT="${MAIN_CUT:-5.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MACE_ICTC_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
LMP="${LMP:-lmp}"
PY="${PY:-python3}"

export LD_LIBRARY_PATH="$($PY -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

# cueq-exported .pt2 bakes a cuequivariance custom op; LAMMPS libtorch must dlopen it.
# (ICTC bridge-U exports, selected by backend key `ictd-bridge-u`, are pure
# PyTorch and don't need these.) The engine reads
# MFF_CUSTOM_OPS_LIB (cueq ops .so) + MFF_LIBPYTHON (for Py* symbols).
SP="$($PY -c 'import cuequivariance_ops_torch,os;print(os.path.dirname(cuequivariance_ops_torch.__file__))' 2>/dev/null || true)"
if [ -n "$SP" ] && [ -d "$SP/_ext" ]; then
  CUE_SO="$(ls "$SP/_ext" | grep -m1 '\.so$' || true)"
  CUE_OPS="$($PY -c 'import cuequivariance_ops,os;print(os.path.join(os.path.dirname(cuequivariance_ops.__file__),"lib","libcue_ops.so"))' 2>/dev/null || true)"
  if [ -n "$CUE_SO" ] && [ -n "$CUE_OPS" ]; then
    CUE_EXT="$SP/_ext/$CUE_SO"
    export MFF_CUSTOM_OPS_LIB="${CUE_OPS}:${CUE_EXT}"
  fi
fi
PY_PREFIX="$($PY -c 'import sys;print(sys.prefix)')"
PY_MM="$($PY -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export MFF_LIBPYTHON="${MFF_LIBPYTHON:-${PY_PREFIX}/lib/libpython${PY_MM}.so}"
# the embedded CPython (for cueq custom-op registration) needs its stdlib + site-packages:
export PYTHONHOME="${PYTHONHOME:-${PY_PREFIX}}"
export PYTHONPATH="${PY_PREFIX}/lib/python${PY_MM}/site-packages:${REPO}:${PYTHONPATH:-}"

# Uniform fixed density RHO_TARGET=0.04 (matches bench_lr_throughput.py): box edge=(NATOMS/0.04)**(1/3)
# at every NATOMS, identical density to the in-graph eager/aoti-infer paths. The max(2*c+0.1,...) floor
# is a HARD PBC requirement (LAMMPS minimum-image + the engine single-image MBD guard need box>=2*cutoff);
# it only binds at small N (disp_cut=9 -> floor 18.1A > 0.04-density box until N~232), keeping disp
# aoti-md runnable rather than erroring. Density is uniform across N wherever the floor does not bind.
RHO_TARGET=0.04
BOX=$($PY -c "c=max($MAIN_CUT,$DISP_CUT);print(max(2.0*c+0.1, ($NATOMS/$RHO_TARGET)**(1.0/3.0)))")

# split NATOMS across 4 element types (H C N O) so pair_coeff has 4 symbols
N1=$(( NATOMS/4 )); N2=$(( NATOMS/4 )); N3=$(( NATOMS/4 )); N4=$(( NATOMS - N1 - N2 - N3 ))

DISP_KW=""
if $PY -c "import sys;sys.exit(0 if float('$DISP_CUT')>0 else 1)"; then
  DISP_KW="dispersion $DISP_CUT"
fi

IN=$(mktemp /tmp/in.mff_md.XXXX)
cat > "$IN" <<LMPIN
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
pair_style mff/torch ${MAIN_CUT} cuda ${DISP_KW}
pair_coeff * * ${PT2} H C N O

velocity all create 300 42
fix 1 all nve
thermo 20
thermo_style custom step temp pe etotal
run ${STEPS}
LMPIN

echo "=== LAMMPS input (${IN}) ==="
echo "pair_style mff/torch ${MAIN_CUT} cuda ${DISP_KW}  | box=${BOX} natoms=${NATOMS} steps=${STEPS}"
"$LMP" -in "$IN"
rm -f "$IN"
