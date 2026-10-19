"""Within-sbatch large-atom / large-model training comparison orchestration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from architecture_cases import chorus_spec, dpa_spec
from run_training_mixed_sweep import VARIANTS


FUSION = ["--training-species-fusion", "--radial-coefficient-fusion",
          "--phase-coefficient-mixed-fusion", "--training-mixed-destination",
          "--training-destination-first", "1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True,
                        choices=("atom_scale", "l3", "c256", "d4", "l3c192d3", "l3c256d4"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--result-set", default="large_training_v1")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--fusion-variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--selection-screen", nargs="+")
    parser.add_argument("--selection-qualification", nargs="+")
    parser.add_argument("--large-oracle-save-on-cpu", action="store_true")
    parser.add_argument("--large-oracle-chunk-edges", type=int, default=0)
    parser.add_argument("--parameter-matched-dpa-root", type=Path,
                        help="Fixed degree/depth DPA initializations selected by parameter count, separately per scope")
    parser.add_argument("--oracle-atoms", type=int, default=1024)
    parser.add_argument("--native-baseline", action='store_true',
                        help='Pair the previous native schedule on the same card, separating kernel gains from the new DPA model matching')
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "GPU-A100":
        raise RuntimeError("Large training tests require the designated GPU-A100 sbatch allocation")
    root, tools = args.root, args.tools
    source, hp_source = Path(os.environ["SOURCE"]), Path(os.environ["HP_SOURCE"])
    assets = Path("/public/home/sps-xia/rxlin/a100_scaling_assets_20260804")
    architectures = Path("/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations")
    overlay = assets / "dpa4_torch211_overlay"
    native_python = "/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python"
    dpa_python = "/public/home/sps-xia/rxlin/venvs/dpa4-a100-scaling/bin/python"
    destination = root / "results" / args.result_set / args.case
    logs = root / "logs" / args.result_set / args.case
    destination.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    common_env = dict(os.environ)
    common_env.update(NVIDIA_TF32_OVERRIDE="0", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MAX_JOBS="4",
                      TORCH_CUDA_ARCH_LIST="8.0", CUDA_HOME="/usr/local/cuda",
                      PATH="/usr/local/cuda/bin:" + os.environ["PATH"],
                      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    native_env = dict(common_env)
    cache = root / "cache" / "native"
    native_env.update(PYTHONPATH=f"{tools}:{source / 'benchmarks/hp'}:{source}:{hp_source}",
        XDG_CACHE_HOME=str(cache / "xdg"), TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
        FSCETP_ICTD_CACHE_DIR=str(cache / "ictd"), TRITON_CACHE_DIR=str(cache / "triton"),
        PYTHONPYCACHEPREFIX=str(cache / "pycache"), TMPDIR=str(root / "cache/tmp"))
    dpa_env = dict(common_env)
    cuda_libs = sorted(str(p) for p in (overlay / "nvidia").glob("*/lib") if p.is_dir())
    dcache = root / "cache" / "dpa"
    for folder in ("xdg/torch/kernels", "inductor", "triton", "pycache", "tmp"):
        (dcache / folder).mkdir(parents=True, exist_ok=True)
    dpa_env.update(PYTHONPATH=f"{tools}:{overlay}:{source}",
        LD_LIBRARY_PATH=":".join([str(assets / "mpi_libs"), str(overlay / "torch/lib"),
                                  *cuda_libs, os.environ.get("LD_LIBRARY_PATH", "")]),
        XDG_CACHE_HOME=str(dcache / "xdg"), TORCHINDUCTOR_CACHE_DIR=str(dcache / "inductor"),
        TRITON_CACHE_DIR=str(dcache / "triton"), PYTHONPYCACHEPREFIX=str(dcache / "pycache"),
        TMPDIR=str(dcache / "tmp"), DP_INTRA_OP_PARALLELISM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="1")
    is_atoms = args.case == "atom_scale"
    sizes = args.sizes or ([6144, 8192, 10000, 16384] if is_atoms else [1024, 4096])
    if min(sizes) < 1:
        raise ValueError("Atom counts must be positive")
    scopes = ["persistent"] if is_atoms else ["persistent", "final"]
    matches = {}
    if args.parameter_matched_dpa_root:
        if is_atoms:
            raise ValueError('Parameter matching applies to the explicit architecture matrix')
        matching = json.loads((args.parameter_matched_dpa_root/'parameter_matches.json').read_text())
        if matching['status'] != 'DONE':
            raise ValueError('Parameter selection has not finished')
        matches = matching['cases'][args.case]['selected']
        for scope in scopes:
            spec = chorus_spec(args.case, scope)
            if (matches[scope]['lmax'] != spec['lmax'] or
                    matches[scope]['layers'] != spec['interactions'] or
                    matches[scope]['chorus_channels'] != spec['channels']):
                raise ValueError('DPA match changed the degree/depth or fixed CHORUS architecture')
    selected = {scope: args.fusion_variant for scope in scopes}
    selection = {}
    if bool(args.selection_screen) != bool(args.selection_qualification):
        raise ValueError("Both screening and qualification sets are required for selection")
    if args.selection_screen:
        if len(args.selection_screen) != len(args.selection_qualification):
            raise ValueError("Each screening set requires its own qualification set")
        for scope in scopes:
            candidates = []
            sources = [(screen_set, qual_set, variant) for screen_set, qual_set in
                zip(args.selection_screen, args.selection_qualification) for variant in VARIANTS]
            for screen_set, qual_set, variant_name in sources:
                screen_file = root / "results" / screen_set / args.case / f"{scope}_n1024_{variant_name}.json"
                qual_file = root / "results" / qual_set / args.case / f"{scope}_n128_{variant_name}.json"
                if not screen_file.is_file() or not qual_file.is_file():
                    continue
                screen, qual = (json.loads(path.read_text()) for path in (screen_file, qual_file))
                if (qual.get("status") not in ("PASS", "PASS_WITH_ELEMENTWISE_EXCEPTIONS")
                        or len(qual.get("qualification", [])) != 3 or "median_ms" not in screen):
                    continue
                identity = ("checkpoint", "source", "trainability", "trainable_parameters", "plan",
                            "arithmetic", "architecture", "backward_targets", "gradient_execution")
                if any(screen.get(key) != qual.get(key) for key in identity):
                    raise ValueError(f"Candidate screen/oracle identity changed: {screen_file}")
                if (screen.get("gradient_execution") != "autograd.grad+accumulation"
                        or screen.get("backward_targets") != "all_trainable_parameters"
                        or any(screen["graph_execution_policy"][key]
                               for key in ("capture", "replay", "inductor_cudagraphs"))):
                    raise ValueError(f"Candidate changed gradient/graph protocol: {screen_file}")
                candidates.append(dict(variant=variant_name, median_ms=screen["median_ms"],
                    screen=str(screen_file), qualification=str(qual_file)))
            if not candidates:
                raise ValueError(f"No qualified candidate is available for {args.case}/{scope}")
            chosen = min(candidates, key=lambda item: item["median_ms"])
            selected[scope] = chosen["variant"]
            selection[scope] = dict(chosen=chosen, candidates=candidates)
    policies = {"--parameter-targets-only" in VARIANTS[value] for value in selected.values()}
    if len(policies) != 1:
        raise ValueError("Both scopes and DPA must share one explicit gradient-target policy")
    parameter_targets = policies.pop()
    baseline_variants = {'l3': 'cache16jvp', 'l3c192d3': 'cache32jvp', 'l3c256d4': 'cache16splitjvp'}
    if args.native_baseline and (args.case not in baseline_variants or not parameter_targets):
        raise ValueError('The previous native schedule requires an established high-l all-parameter target case')
    manifest = dict(case=args.case, sizes=sizes, scopes=scopes, neighbors=50,
        repeats=args.repeats, warmup=args.warmup, changing_batches=3,
        source=str(source), hp_source=str(hp_source), tools=str(tools),
        slurm_job=os.environ["SLURM_JOB_ID"],
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        timed_work="energy+conservative force+E/F loss+mixed double backward+AdamW update",
        graph_capture=False, graph_replay=False, precision="strict FP32; no AMP/TF32",
        model_kind="trained_checkpoint" if is_atoms else "fixed_constructor_initialization",
        architecture=None if is_atoms else {scope: chorus_spec(args.case, scope) for scope in scopes},
        dpa_architecture=None if is_atoms else dpa_spec(args.case),
        note="CHORUS and DPA channel definitions differ; no accuracy-equivalence assertion.")
    manifest.update(fusion_variant=args.fusion_variant, scoped_variants=selected, selection=selection,
        backward_targets="all_trainable_parameters" if parameter_targets else "all_leaves",
        gradient_execution="autograd.grad+accumulation" if parameter_targets else "Tensor.backward")
    manifest.update(parameter_matching=matches or None, oracle_atoms=args.oracle_atoms)
    manifest['native_baseline_variant'] = baseline_variants[args.case] if args.native_baseline else None
    if matches:
        manifest['dpa_architecture'] = {scope: {**dpa_spec(args.case), 'channels': match['dpa_channels']}
            for scope, match in matches.items()}
        manifest['note'] = ('Same lmax and interaction depth; DPA channels selected by parameter count only. '
                            'The exact parameter mismatch is retained separately for each scope; no accuracy-equivalence assertion.')
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    records = []

    def child(label, command, env, output):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite prior measurement {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        log = logs / f"{label}.log"
        started = time.time()
        entry = dict(label=label, command=command, log=str(log), output=str(output),
                     slurm_job=os.environ["SLURM_JOB_ID"],
                     CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
        with log.open("x") as stream:
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            process = subprocess.Popen(command, env=env, cwd=root,
                                       stdout=stream, stderr=subprocess.STDOUT)
            entry["pid"] = process.pid
            print(json.dumps(entry), flush=True)
            (destination / "active_child.json").write_text(json.dumps(entry, indent=2) + "\n")
            try:
                code = process.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -9
        entry.update(exit_code=code, elapsed_seconds=time.time() - started)
        if code:
            content = log.read_text(errors="replace")
            kind = "OOM" if "out of memory" in content.lower() else "TIMEOUT" if code == -9 else "FAIL"
            entry["status"] = kind
            if not output.exists():
                output.write_text(json.dumps({**entry, "atoms": next((command[i + 1] for i, x in
                    enumerate(command) if x == "--atoms"), None)}, indent=2) + "\n")
        else:
            entry["status"] = json.loads(output.read_text())["status"]
        records.append(entry)
        (destination / "child_records.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        return code

    def native_command(scope, n, out, tag, qualification=False, baseline=False):
        if is_atoms:
            command = [native_python, str(tools / "profile_training_nograph.py"),
                       "--checkpoint", str(assets / "persistent_t1x.pth")]
        else:
            initial = architectures / args.case / f"chorus_{scope}_{args.case}_initialization.pt"
            command = [native_python, str(tools / "profile_training_architecture_nograph.py"),
                       "--architecture-case", args.case, "--scope", scope,
                       "--checkpoint", str(initial)]
        command += ["--atoms", str(n), "--neighbors", "50", "--rounds", "1",
                    "--warmup", str(1 if qualification else args.warmup),
                    "--repeats", str(1 if qualification else args.repeats),
                    *FUSION, *VARIANTS[baseline_variants[args.case] if baseline else selected[scope]],
                    "--tag", tag, "--out", str(out)]
        if not qualification:
            command.append("--measurement-only")
        elif n > 128:
            if args.large_oracle_save_on_cpu:
                command += ["--reference-save-on-cpu", "--reference-empty-cache"]
            if args.large_oracle_chunk_edges:
                command += ["--reference-tp-chunk-edges", str(args.large_oracle_chunk_edges)]
        return command

    def dpa_command(n, out, scope=None):
        directory = (args.parameter_matched_dpa_root / args.case / scope if matches else
                     assets / "dpa4/c32" if is_atoms else architectures / args.case / "dpa4")
        checkpoint = directory / ("model.pt" if is_atoms else "initialization.pt")
        command = [dpa_python, str(tools / "profile_training_dpa_nograph.py"),
                "--atoms", str(n), "--neighbors", "50", "--checkpoint", str(checkpoint),
                "--config", str(directory / "input.json"), "--warmup", str(args.warmup),
                "--repeats", str(args.repeats), "--out", str(out)]
        if parameter_targets:
            command.append("--parameter-targets-only")
        return command

    failed_small = set()
    if not is_atoms:
        for scope in scopes:
            output = destination / "qualification128" / f"{scope}.json"
            if child(f"qualification128_{scope}", native_command(scope, 128, output,
                     "qualification128", True), native_env, output):
                failed_small.add(scope)
    for n in sizes:
        for round_index in (0, 1):
            if matches:
                order = [label for scope in scopes for label in (scope, 'dpa_'+scope)]
                if round_index:order.reverse()
            else:
                order = [*scopes, "dpa"] if round_index == 0 else ["dpa", *reversed(scopes)]
            if args.native_baseline:
                for native_scope in scopes:
                    order.insert(order.index(native_scope)+(1 if round_index else 0), 'baseline_'+native_scope)
            for scope in order:
                output = destination / f"n{n}" / f"{scope}_r{round_index}.json"
                label = f"n{n}_{scope}_r{round_index}"
                if scope in failed_small:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(dict(status="SKIPPED_SMALL_QUALIFICATION_FAILURE",
                        scope=scope, atoms=n), indent=2) + "\n")
                    continue
                command, env = ((dpa_command(n, output, scope.removeprefix('dpa_')), dpa_env)
                    if scope == "dpa" or scope.startswith('dpa_') else
                    (native_command(scope.removeprefix('baseline_'), n, output, label,
                        baseline=scope.startswith('baseline_')), native_env))
                child(label, command, env, output)
    # Same-size original-formula qualification is separate from all paired
    # timings. A failed large oracle never invalidates a successful timed step
    # nor turns a small-graph qualification into a large-graph PASS.
    if not is_atoms:
        for scope in scopes:
            if scope in failed_small:
                continue
            output = destination / f"qualification{args.oracle_atoms}" / f"{scope}.json"
            child(f"qualification{args.oracle_atoms}_{scope}", native_command(scope, args.oracle_atoms, output,
                  f"qualification{args.oracle_atoms}", True), native_env, output)
    status = "DONE_WITH_FAILURES" if any(r["status"] in ("FAIL", "TIMEOUT") for r in records) else "DONE"
    (destination / "case_status.json").write_text(json.dumps(dict(status=status, records=records), indent=2) + "\n")
    print(json.dumps(dict(case=args.case, status=status)), flush=True)


if __name__ == "__main__":
    main()
