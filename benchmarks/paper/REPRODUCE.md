# Reproducing the Paper Benchmarks

This file maps each reported benchmark artifact to the tracked data, plotting
script, and fresh-run command. The repository keeps lightweight CSV/JSON/SVG
records in Git. Large checkpoints, AOTInductor `.pt2` packages, full MD
trajectories, and some release-only logs are external artifacts.

The reported timings were measured on an NVIDIA RTX 4090 D with CUDA, PyTorch
2.7, `mace-torch==0.3.16`, `e3nn==0.5.9`, and the `USER-MFFTORCH` LAMMPS
package built against the same LibTorch/Python environment. Absolute timings
will move with GPU, driver, PyTorch, and LAMMPS build details.

## Replot From Archived Data

Run from the repository root:

```bash
python benchmarks/paper/scripts/plot_benchmark_figures.py
python benchmarks/paper/scripts/training/plot_paper_training_figures.py
python benchmarks/paper/scripts/plot_lr_throughput.py \
  benchmarks/paper/results/long_range/lr_throughput_channels64.json \
  benchmarks/paper/figures/lr_throughput_channels64
```

These commands regenerate the tracked benchmark figures from
`benchmarks/paper/results/`.

## Isolated Operator Benchmark

Data:

- `benchmarks/paper/results/operator/operator_compile_fwbw_flat.csv`
- `benchmarks/paper/results/operator/operator_cartnn_vs_ictc.csv`
- `benchmarks/paper/results/operator/cartnn_fairness_audit_20260618/*.csv`
- `benchmarks/paper/results/operator/ictp_official_20260712/*.csv`
- `benchmarks/paper/results/operator/ictp_official_20260712/operator_ictp_validation.json`

Fresh-run commands:

```bash
python benchmarks/paper/scripts/operator/operator_bench_compile_fwbw.py \
  --out /tmp/mace_ictc_operator_fwbw \
  --device cuda --channels 64 --edges 100000 \
  --configs 1:1,1:2,2:2,2:3

python benchmarks/paper/scripts/operator/operator_bench.py \
  --out /tmp/mace_ictc_operator_refs \
  --device cuda --channels 64 --edges 100000 \
  --configs 1:1,1:2,2:2,2:3 \
  --backends e3nn,cartnn,ictc,cart3l
```

The official ICTP reference is run through a separate adapter so its
non-commercial upstream source remains an unmodified external dependency:

```bash
python benchmarks/paper/scripts/operator/operator_bench_ictp.py \
  --ictp-root /path/to/nec-research/ictp \
  --out /tmp/mace_ictc_operator_ictp \
  --device cuda --channels 64 --edges 100000 \
  --configs 1:1,1:2,2:2,2:3,3:3 \
  --dtype float32 --mode forward_backward \
  --warmup 20 --measured 50

python benchmarks/paper/scripts/operator/operator_bench_ictp.py \
  --ictp-root /path/to/nec-research/ictp \
  --out /tmp/mace_ictc_operator_ictp_forward \
  --device cuda --channels 64 --edges 100000 \
  --configs 1:1,1:2,2:2,2:3,3:3 \
  --dtype float32 --mode forward_only \
  --warmup 20 --measured 50
```

The archived ICTP record pins upstream commit
`f40592a5687ec1d03219300ee557b2660f7d0369`. The adapter verifies path-set
and output-width equality, float64 covariance, symmetry, tracelessness, and
finite gradients before timing. Publications using the ICTP software must
acknowledge that it was developed by NEC Laboratories Europe GmbH.

CACE is a separate reference workload, not a matched MACE tensor-product
baseline:

```bash
python benchmarks/paper/scripts/operator/operator_bench_cace_atp.py \
  --cace-root /path/to/cace \
  --out /tmp/mace_ictc_cace_atp
```

Post-processing:

```bash
python benchmarks/paper/scripts/operator/summarize_operator.py \
  /tmp/mace_ictc_operator_refs
```

## Whole-Model Throughput

Data:

- `benchmarks/paper/results/model/selected_fixed_configs_channels64.csv`
- `benchmarks/paper/results/model/selected_best_modes_channels64.csv`
- `benchmarks/paper/results/model/raw/*.csv`

Fresh-run command:

```bash
PYTHONPATH=/path/to/mace-torch-0.3.16:$PWD \
python -m mace_ictc.bench.bench_mace_ictc_vs_mace \
  --device cuda --dtype float32 \
  --channels 64 --avg-degree 50 \
  --atoms-list 512,1024,2048,4096,8192 \
  --configs 1:1,1:2,2:2,2:3 \
  --out-dir /tmp/mace_ictc_whole_model
```

Selection/post-processing:

```bash
python benchmarks/paper/scripts/model/plot_best_modes.py
python benchmarks/paper/scripts/plot_benchmark_figures.py
```

The benchmark uses synthetic fixed-edge graphs to measure backend throughput;
it is not a chemical-accuracy task.

## Long-Range Throughput

Data:

- `benchmarks/paper/results/long_range/lr_throughput_channels64.json`

Main fresh-run command for the table conditions
`none,elec,elec-mp,disp,both`:

```bash
PY=/path/to/python \
LMP=/path/to/lmp \
MACE_ICTC_REPO=$PWD \
python benchmarks/paper/scripts/long_range/bench_lr_throughput.py \
  --modes train,makefx-train,infer,aoti-infer,md,aoti-md \
  --conditions none,elec,elec-mp,disp,both \
  --sizes 128,256,512,1024,2048 \
  --json-out /tmp/lr_throughput_channels64.json
```

The `aoti-md` mode shells out to:

```bash
bash benchmarks/paper/scripts/long_range/bench_lammps_md.sh \
  /path/to/core.pt2 512 9.0
```

The figure also contains deployed-MD auxiliary curves for `disp-respa` and
`disp-c6`. These require exported `.pt2` cores and the rRESPA training artifact
used in the original 4090 run:

```bash
LRX=/path/to/lrx-or-release-artifacts \
LR_ISO_PT2=/path/to/disp_path1_mbdslq_c64_cut9_iso.pt2 \
LR_C6_PT2=/path/to/disp_c6_c64_cut9.pt2 \
PY=/path/to/python \
LMP=/path/to/lmp \
bash benchmarks/paper/scripts/long_range/fig_sweep_deployed_md.sh
```

If the original 4090 log layout is available, assemble the combined JSON with:

```bash
LRX=/path/to/lrx-or-release-artifacts \
python benchmarks/paper/scripts/long_range/assemble_lr_json.py
```

`benchmarks/paper/scripts/long_range/lr_pipeline_rerun.sh` is the archived
as-run pipeline helper for re-exporting the C6/rRESPA cores and rerunning the
deployed-MD sweeps. It intentionally still expects the external
`respa_trained/` artifact tree.

## Training Curves, NTK, and MD Parity

Training data and summaries:

- `benchmarks/paper/results/training/md17_training_curves_combined.csv`
- `benchmarks/paper/results/training/matched_training_curves_combined.csv`
- `benchmarks/paper/results/training/*aggregate*.csv`

Fresh-run entry points:

```bash
python benchmarks/paper/scripts/training/prepare_md17_public.py \
  --root /tmp/pyg_md17_cache \
  --out-root /tmp/mace_ictc_public_md17

DATA_ROOT=/tmp/mace_ictc_public_md17 \
bash benchmarks/paper/scripts/training/run_md17_training_matrix.sh

python benchmarks/paper/scripts/training/summarize_md17_training.py \
  /tmp/mace_ictc_public_md17_train_RUN \
  --out /tmp/mace_ictc_public_md17_train_RUN/summary.csv
```

NTK diagnostics:

```bash
python benchmarks/paper/scripts/training/diagnose_ntk_spectrum.py \
  --data-dir /tmp/mace_ictc_public_md17/revised_benzene \
  --out-dir /tmp/mace_ictc_ntk/revised_benzene_batch0 \
  --atomic-energy-keys H,C,N,O \
  --atomic-energy-values 0,0,0,0

python benchmarks/paper/scripts/training/summarize_ntk_sweeps.py \
  /tmp/mace_ictc_ntk \
  --out-dir /tmp/mace_ictc_ntk/aggregate
```

Long-MD checkpoint correspondence:

```bash
python benchmarks/paper/scripts/model/md_parity_off23_long.py \
  --engine native \
  --mace-model /path/to/MACE-OFF23_small.model \
  --outdir /tmp/mace_ictc_md_parity

python benchmarks/paper/scripts/model/md_parity_off23_long.py \
  --engine ictc \
  --mace-model /path/to/MACE-OFF23_small.model \
  --ictc-checkpoint /path/to/off23_small_ictc_bridge_u_float64.pth \
  --outdir /tmp/mace_ictc_md_parity
```

These runs depend on public datasets plus large checkpoints that are not kept in
Git. Use the release artifact bundle when exact byte-for-byte provenance is
needed.

## Table Generation

The paper source repository generates appendix tables from the archived data via:

```bash
python scripts/gen_throughput_tables.py
```

Run that command in the paper-source checkout, not in this source-code
repository.
