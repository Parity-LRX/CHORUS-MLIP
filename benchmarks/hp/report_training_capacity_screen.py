"""Remote reduction of explicitly unqualified large-N capacity screens."""
import argparse
import json
from pathlib import Path
import re
from report_native_training_release import case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    oom = {}
    stage, atoms = None, None
    for line in args.log.read_text().splitlines():
        if line.startswith("+ ") and "profile_training_dpa_nograph.py" in line:
            stage = "dpa"
        elif line.startswith("+ ") and "profile_training_nograph.py" in line:
            stage = "fused"
        if line.startswith("+ ") and (found := re.search(r"--atoms (\d+)", line)):
            atoms = int(found.group(1))
        if "torch.OutOfMemoryError:" in line and stage is not None and atoms is not None:
            oom.setdefault(f"{stage}:{atoms}", []).append(line)
    result = {"scope": "DIAGNOSTIC_NO_LARGE_REFERENCE", "formal_large_N_qualification": False,
              "full_gpu_training_step": True, "cuda_graph": False,
              "oom_log": str(args.log), "oom_events": oom, "rows": []}
    lines = ["# 大规模训练容量/吞吐筛选", "",
             "本表为完整 GPU E/F/混合反传/AdamW 筛选；这些点尚未完成同尺寸原式梯度校验。",
             "三批输入轮换、无 CUDA Graph、FP32/TF32 off；每步 autograd 构建包含在计时内。",
             "邻居表及输入数据准备在计时外完成。", "",
             "| 原子数 / 50 邻居 | CHORUS ms | DPA compile ms | DPA / CHORUS | CHORUS reserved GB | DPA reserved GB |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in (4096, 8192, 10000):
        row = {"atoms": n, "fused": case(args.root/f"n{n}", "fused"),
               "dpa": case(args.root/f"n{n}", "dpa")}
        fused, dpa = row["fused"], row["dpa"]
        if "median_ms" in fused:
            if "median_ms" in dpa:
                row["screen_speedup_over_dpa"] = dpa["median_ms"]/fused["median_ms"]
                dtime, ratio, dmem = f'{dpa["median_ms"]:.3f}', f'{row["screen_speedup_over_dpa"]:.3f}×', f'{dpa["peak_reserved"]/1e9:.3f}'
            else:
                count = len(oom.get(f"dpa:{n}", []))
                dtime, ratio, dmem = (f"OOM（{count} 次）" if count else "未完成"), "—", "—"
                row["dpa_oom_attempts"] = count
            lines.append(f'| {n} | {fused["median_ms"]:.3f} | {dtime} | {ratio} | {fused["peak_reserved"]/1e9:.3f} | {dmem} |')
        result["rows"].append(row)
    lines += ["", "OOM 点只报告容量差异，不计算或外推速度比。",
              "正式原式及全部参数梯度检查另见配对报告；4096 原子后续已完成正式校验。",
              "大图筛选不替代这些点自身的原式校验。", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"capacity_screen_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.out/"TRAINING_CAPACITY_SCREEN.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
