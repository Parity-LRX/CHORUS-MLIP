"""Remote batch-only aggregation; never run numerical reductions on the Mac."""
import argparse
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--candidate", default="source_v4")
    p.add_argument("--result-set", default="")
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args()
    data_root = args.root / "results" / args.result_set
    rows, records, missing = [], {}, []
    for n in (128, 512):
        groups = {}
        for prefix in ("source_v1", args.candidate, "dpa"):
            values = []
            for repeat in (0, 1):
                path = data_root / f"n{n}" / f"{prefix}_r{repeat}.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                data = json.loads(path.read_text())
                policy = data["graph_execution_policy"]
                if any(policy[k] for k in ("capture", "replay", "inductor_cudagraphs")):
                    raise RuntimeError(f"Graph-enabled record cannot qualify: {path}")
                values.append(data)
            groups[prefix] = values
        records[str(n)] = groups
        if any(len(v) != 2 for v in groups.values()):
            continue
        old, new, dpa = (statistics.median(v["median_ms"] for v in groups[name])
                         for name in ("source_v1", args.candidate, "dpa"))
        memory = {name: statistics.median(
            max(m["peak_reserved"] for m in v["measurements"]) if "measurements" in v
            else v["peak_reserved"] for v in groups[name]) / 1e9 for name in groups}
        rows.append({"atoms": n, "old_ms": old, "new_ms": new, "dpa_ms": dpa,
            "time_reduction_fraction": 1. - new / old, "speedup_vs_old": old / new,
            "speedup_vs_dpa": dpa / new, "reserved_gb": memory,
            "strict_elementwise_gradient_failures_old": [v["strict_elementwise_gradient_failures"] for v in groups["source_v1"]],
            "strict_elementwise_gradient_failures_new": [v["strict_elementwise_gradient_failures"] for v in groups[args.candidate]]})
    tests = []
    for path in sorted((args.root / "results").glob("*.xml")):
        root = ET.parse(path).getroot()
        suites = list(root) if root.tag == "testsuites" else [root]
        tests.append({"path": str(path), **{key: sum(int(s.attrib.get(key, 0)) for s in suites)
                       for key in ("tests", "failures", "errors", "skipped")}})
    variable_path = data_root / "n512/variable_n.json"
    variable = json.loads(variable_path.read_text()) if variable_path.exists() else None
    result = {"rows": rows, "missing": missing, "tests": tests, "variable": variable,
              "records": records, "graph_capture_or_replay": False}
    out = args.root / args.out_dir
    out.mkdir(exist_ok=True)
    (out / "training_nograph_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# 无 CUDA Graph 的完整训练 step", "",
        "全部数值工作、编译、计时和本汇总由 sbatch / GPU-A100 / XHPC-A100 执行。",
        "CHORUS 原 checkpoint、全部 1,029,967 个可训练参数及 FP32/TF32 off 保持；",
        "完整 E/F 损失、混合二阶反传与 AdamW 更新，每步重新建立 autograd 图。",
        "capture/replay 由运行时守卫禁止，DPA 的 Inductor cudagraphs 同样关闭。", "",
        "同尺寸三批次变化坐标、元素及边顺序；旧/新 HP 在同一作业内正反顺序测量，",
        "DPA 使用匹配的三批次和真实可微 compile。所有准备/JIT 排除在稳态计时外。",
        "DPA level-2 value path 仍为上游仅推理实现，不宣称训练调用了该核。", "",
        "| 原子数 / 50 邻居 | 旧 HP ms | 新 HP ms | DPA compile ms | 新/旧吞吐 | 相对 DPA |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['atoms']} | {row['old_ms']:.3f} | {row['new_ms']:.3f} | {row['dpa_ms']:.3f} | {row['speedup_vs_old']:.3f}× | {row['speedup_vs_dpa']:.3f}× |")
    lines += ["", "本轮只替换相位上下文 gather–SiLU 的一阶/混合二阶伴随；",
        "没有改变参数、优化器、精度、模型或数值资格阈值。",
        "未将固定 Graph 结果或微核结果记为整网训练提升。", ""]
    for row in rows:
        lines.append(f"- N={row['atoms']} reserved GB（旧/新/DPA）：" +
            "/".join(f"{row['reserved_gb'][k]:.3f}" for k in ("source_v1", args.candidate, "dpa")) +
            f"；跨三批次逐元素超差参数张量记录数旧={row['strict_elementwise_gradient_failures_old']}，新={row['strict_elementwise_gradient_failures_new']}。")
    if variable:
        lines += ["", f"变 N/E 序列：{variable['status']}，原子数 " +
            "/".join(str(v['atoms']) for v in variable['batches']) +
            f"；完整序列 median step {variable['median_ms']:.3f} ms。",
            "该变尺寸检查没有匹配的 DPA 序列计时，不给出跨模型加速比。"]
    if missing:
        lines += ["", "尚缺记录：", *[f"- {path}" for path in missing]]
    lines += ["", "该伴随候选没有形成明确的完整 step 吞吐收益，作为默认关闭的实验选项保留。",
        "原训练默认路径不变。逐元素梯度异常仍需解决，不将张量范数通过写成逐元素通过。"]
    lines += ["", "回归及失败记录见 summary JSON 的 tests 字段；初次 Triton 语法失败和",
        "未通过的旧快照保留，不计作数值通过。", ""]
    (out / "TRAINING_NOGRAPH_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"rows": rows, "tests": tests, "missing": missing,
                      "variable_status": None if variable is None else variable['status']}, indent=2))


if __name__ == "__main__":
    main()
