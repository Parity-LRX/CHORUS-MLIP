"""Within-sbatch screening of live high-l TP adjoints, without CUDA Graph.

Each candidate is a separate process with the same fixed initialization and
three changing batches. These short timings are screens, not final paired
DPA comparisons or substitutes for original-formula gradient qualification.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


VARIANTS = {
    "baseline": [],
    **{f"first_{schedule}": ["--training-first-schedule", schedule,
        "--training-mixed-cache", "16", "--training-mixed-grouped", "--training-mixed-fuse-jvp",
        "--parameter-targets-only", "--training-terminal-fusion"]
       for schedule in ("pruned", "grouped", "compact", "compact_grouped")},
    **{f"radial_joint_{tile//1024}k": ["--training-radial-tp-fusion", "--training-radial-cached",
        "--training-mixed-cache", "16", "--training-mixed-fuse-jvp", "--training-mixed-grouped",
        "--radial-tile-edges", str(tile), "--training-first-schedule", "pruned",
        "--parameter-targets-only", "--training-terminal-fusion"]
       for tile in (8192, 32768, 65536)},
    **{f"radial_bound_{tile//1024}k": ["--training-radial-tp-fusion", "--training-radial-cached",
        "--training-mixed-cache", "16", "--training-mixed-fuse-jvp", "--training-mixed-grouped",
        "--training-radial-first-bounded", "--radial-tile-edges", str(tile),
        "--training-first-schedule", "pruned", "--parameter-targets-only", "--training-terminal-fusion"]
       for tile in (8192, 32768, 65536)},
    "t32": ["--training-mixed-threads", "32"],
    "t64": ["--training-mixed-threads", "64"],
    "group32": ["--training-mixed-threads", "32", "--training-mixed-grouped"],
    "group64": ["--training-mixed-threads", "64", "--training-mixed-grouped"],
    "group128": ["--training-mixed-threads", "128", "--training-mixed-grouped"],
    "compact": ["--training-mixed-compact"],
    "compact32": ["--training-mixed-compact", "--training-mixed-threads", "32"],
    "compact64": ["--training-mixed-compact", "--training-mixed-threads", "64"],
    "compactg32": ["--training-mixed-compact", "--training-mixed-threads", "32", "--training-mixed-grouped"],
    "compactg64": ["--training-mixed-compact", "--training-mixed-threads", "64", "--training-mixed-grouped"],
    "radial": ["--training-radial-tp-fusion", "--training-mixed-compact"],
    "radial32": ["--training-radial-tp-fusion", "--training-mixed-compact", "--training-mixed-threads", "32"],
    "radialg32": ["--training-radial-tp-fusion", "--training-mixed-compact", "--training-mixed-threads", "32", "--training-mixed-grouped"],
    "targets": ["--parameter-targets-only"],
    "compacttargets": ["--training-mixed-compact", "--parameter-targets-only"],
    "compact32targets": ["--training-mixed-compact", "--training-mixed-threads", "32", "--parameter-targets-only"],
    "compactg32targets": ["--training-mixed-compact", "--training-mixed-threads", "32", "--training-mixed-grouped", "--parameter-targets-only"],
    "radial32targets": ["--training-radial-tp-fusion", "--training-mixed-compact", "--training-mixed-threads", "32", "--parameter-targets-only"],
    "compact32term": ["--training-mixed-compact", "--training-mixed-threads", "32", "--parameter-targets-only", "--training-terminal-fusion"],
    "radial32term": ["--training-radial-tp-fusion", "--training-mixed-compact", "--training-mixed-threads", "32", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache8term": ["--training-mixed-cache", "8", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache16term": ["--training-mixed-cache", "16", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache32term": ["--training-mixed-cache", "32", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache64term": ["--training-mixed-cache", "64", "--parameter-targets-only", "--training-terminal-fusion"],
    "radialcache": ["--training-radial-tp-fusion", "--training-radial-cached", "--parameter-targets-only", "--training-terminal-fusion"],
    "jointfirst": ["--training-mixed-cache", "16", "--training-first-cache", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache32first": ["--training-mixed-cache", "32", "--training-first-cache", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache64first": ["--training-mixed-cache", "64", "--training-first-cache", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache8jvp": ["--training-mixed-cache", "8", "--training-mixed-fuse-jvp", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache16jvp": ["--training-mixed-cache", "16", "--training-mixed-fuse-jvp", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache32jvp": ["--training-mixed-cache", "32", "--training-mixed-fuse-jvp", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache16split": ["--training-mixed-cache", "16", "--training-mixed-grouped", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache32split": ["--training-mixed-cache", "32", "--training-mixed-grouped", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache16splitjvp": ["--training-mixed-cache", "16", "--training-mixed-grouped", "--training-mixed-fuse-jvp", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache32splitjvp": ["--training-mixed-cache", "32", "--training-mixed-grouped", "--training-mixed-fuse-jvp", "--parameter-targets-only", "--training-terminal-fusion"],
    "cache32splitall": ["--training-mixed-cache", "32", "--training-mixed-grouped", "--training-mixed-fuse-jvp", "--training-first-cache", "--parameter-targets-only", "--training-terminal-fusion"],
    "radialfirst": ["--training-radial-tp-fusion", "--training-radial-cached", "--training-first-cache", "--parameter-targets-only", "--training-terminal-fusion"],
    "radial8k": ["--training-radial-tp-fusion", "--training-radial-cached", "--radial-tile-edges", "8192", "--parameter-targets-only", "--training-terminal-fusion"],
    "radial32k": ["--training-radial-tp-fusion", "--training-radial-cached", "--radial-tile-edges", "32768", "--parameter-targets-only", "--training-terminal-fusion"],
    "radial64k": ["--training-radial-tp-fusion", "--training-radial-cached", "--radial-tile-edges", "65536", "--parameter-targets-only", "--training-terminal-fusion"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--atoms", type=int, default=1024)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--scopes", nargs="+", choices=("persistent", "final"), default=["persistent"])
    parser.add_argument("--result-set", required=True)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "GPU-A100":
        raise RuntimeError("Only the designated GPU-A100 sbatch allocation is allowed")
    root, tools = args.root, args.tools
    source, hp = Path(os.environ["SOURCE"]), Path(os.environ["HP_SOURCE"])
    cache = root / "cache/native"
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{tools}:{source / 'benchmarks/hp'}:{source}:{hp}",
        NVIDIA_TF32_OVERRIDE="0", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MAX_JOBS="4",
        TORCH_CUDA_ARCH_LIST="8.0", CUDA_HOME="/usr/local/cuda",
        PATH="/usr/local/cuda/bin:" + env["PATH"],
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        XDG_CACHE_HOME=str(cache / "xdg"), TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
        FSCETP_ICTD_CACHE_DIR=str(cache / "ictd"), TRITON_CACHE_DIR=str(cache / "triton"),
        PYTHONPYCACHEPREFIX=str(cache / "pycache"), TMPDIR=str(root / "cache/tmp"))
    destination = root / "results" / args.result_set / args.case
    logs = root / "logs" / args.result_set / args.case
    destination.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    architecture = Path("/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations")
    records = []
    for scope in args.scopes:
        for variant in args.variants:
            label = f"{scope}_n{args.atoms}_{variant}"
            output, log = destination / f"{label}.json", logs / f"{label}.log"
            if output.exists() or log.exists():
                raise FileExistsError(label)
            command = ["/public/home/sps-xia/rxlin/.conda/envs/mff/bin/python",
                str(tools / "profile_training_architecture_nograph.py"),
                "--architecture-case", args.case, "--scope", scope,
                "--checkpoint", str(architecture / args.case /
                    f"chorus_{scope}_{args.case}_initialization.pt"),
                "--atoms", str(args.atoms), "--neighbors", "50", "--rounds", "1",
                "--warmup", "2", "--repeats", str(args.repeats),
                "--training-species-fusion", "--radial-coefficient-fusion",
                "--phase-coefficient-mixed-fusion", "--training-mixed-destination",
                "--training-destination-first", "1", *VARIANTS[variant],
                "--tag", label, "--out", str(output)]
            if not args.qualify:
                command.append("--measurement-only")
            if args.profile:
                command.append("--profile")
            entry = dict(label=label, command=command, source=str(source), hp_source=str(hp),
                slurm_job=os.environ["SLURM_JOB_ID"],
                CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                output=str(output), log=str(log), kind="short_candidate_screen")
            started = time.time()
            with log.open("x") as stream:
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                process = subprocess.Popen(command, env=env, cwd=root,
                    stdout=stream, stderr=subprocess.STDOUT)
                entry["pid"] = process.pid
                (destination / "active_child.json").write_text(json.dumps(entry, indent=2) + "\n")
                print(json.dumps(entry), flush=True)
                try:
                    code = process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    code = -9
            entry.update(exit_code=code, elapsed_seconds=time.time() - started)
            if code:
                content = log.read_text(errors="replace")
                entry["status"] = "OOM" if "out of memory" in content.lower() else "FAIL"
            else:
                result = json.loads(output.read_text())
                entry.update(status=result["status"], measurements=result["measurements"])
            records.append(entry)
            (destination / "screen.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps({k: v for k, v in entry.items() if k != "measurements"}), flush=True)
            if code and args.stop_on_error:
                raise RuntimeError(f"Candidate failed: {label}; preserved log: {log}")


if __name__ == "__main__":
    main()
