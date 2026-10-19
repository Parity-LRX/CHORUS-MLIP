"""Summarize qualified short screens; execute only inside the XHPC sbatch job."""
import argparse
import json
import os
from pathlib import Path


SETS = (
    ("first_cached_screen_v6", "first_cached_qual_v6"),
    ("wide_cached_screen_v7", "wide_cached_qual_v7"),
    ("radial_tiles_screen_v8", "radial_tiles_qual_v8"),
    ("fused_jvp_screen_v9", "fused_jvp_qual_v9"),
    ("group_pilot_screen_v10", "group_pilot_qual_v10"),
    ("first_fusion_screen_v13", "first_fusion_qual_v13"),
    ("radial_first_screen_v14", "radial_first_qual_v14"),
    ("first_scale_screen_v15", "first_scale_qual_v15"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "GPU-A100":
        raise RuntimeError("Candidate reduction belongs to the designated XHPC sbatch allocation")
    rows = []
    lines = ["# 高 L 候选：速度与显存筛选", "",
        "本表为 1024 原子、50 邻居、12 或 16 步短筛选，不能代替 50 步正反配对结论。",
        "每个候选均要求三批 128 原子的输出、全参数梯度和 AdamW 更新核验通过。",
        "FP32、AMP/TF32 off；新建 autograd、完整 E/F 损失，无 CUDA Graph。", "",
        "| 配置 | scope | 候选 | ms | reserved GB | Pareto |",
        "| --- | --- | --- | ---: | ---: | --- |"]
    for case in ("l3", "l3c192d3", "l3c256d4"):
        for scope in ("persistent", "final"):
            group = []
            for screen_set, qual_set in SETS:
                for path in sorted((args.root / "results" / screen_set / case).glob(f"{scope}_n1024_*.json")):
                    if path.name.endswith((".trace.json", ".qualification.json")):
                        continue
                    value = json.loads(path.read_text())
                    qualifier = args.root / "results" / qual_set / case / path.name.replace("_n1024_", "_n128_")
                    if "median_ms" not in value or not qualifier.is_file():
                        continue
                    qual = json.loads(qualifier.read_text())
                    if qual.get("status") not in ("PASS", "PASS_WITH_ELEMENTWISE_EXCEPTIONS") or len(qual.get("qualification", [])) != 3:
                        continue
                    identity = ("source", "checkpoint", "trainability", "trainable_parameters", "architecture", "plan", "arithmetic")
                    if any(value.get(key) != qual.get(key) for key in identity):
                        raise AssertionError(f"Changed candidate/oracle identity: {path}")
                    policy = value["graph_execution_policy"]
                    if any(policy[key] for key in ("capture", "replay", "inductor_cudagraphs")) or value["backward_targets"] != "all_trainable_parameters":
                        raise AssertionError(f"Changed training protocol: {path}")
                    measured = value["measurements"][0]
                    group.append(dict(case=case, scope=scope, variant=path.stem.split("_n1024_", 1)[1],
                        median_ms=value["median_ms"], peak_allocated=measured["peak_allocated"],
                        peak_reserved=measured["peak_reserved"], source=value["source"],
                        screen=str(path), qualification=str(qualifier), plan=value["plan"]))
            for row in group:
                row["pareto"] = not any(other["median_ms"] <= row["median_ms"] and
                    other["peak_reserved"] <= row["peak_reserved"] and
                    (other["median_ms"] < row["median_ms"] or other["peak_reserved"] < row["peak_reserved"])
                    for other in group)
            group.sort(key=lambda row: row["median_ms"])
            rows.extend(group)
            for row in group:
                lines.append(f'| {case} | {scope} | {row["variant"]} | {row["median_ms"]:.3f} | '
                    f'{row["peak_reserved"]/1e9:.3f} | {"是" if row["pareto"] else "—"} |')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidate_summary.json").write_text(json.dumps(dict(
        kind="short_candidate_screen_not_paired_conclusion", atoms=1024, neighbors=50, rows=rows), indent=2)+"\n")
    (args.out / "CANDIDATES.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
