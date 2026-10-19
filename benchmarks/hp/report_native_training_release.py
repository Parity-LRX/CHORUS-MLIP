"""Remote-only summary of immutable no-CUDA-Graph native training pairs."""
import argparse
import json
from pathlib import Path
import statistics


def case(root, name):
    paths = [root/f"{name}_r{r}.json" for r in range(2)]
    if not all(path.is_file() for path in paths):
        return {"status": "INCOMPLETE", "missing": [str(path) for path in paths if not path.is_file()]}
    values = [json.loads(path.read_text()) for path in paths]
    measurements = [row for value in values for row in value.get("measurements", [value])]
    gradients = [error["relative_l2"] for value in values
                 for batch in value.get("qualification", [])
                 for error in batch["parameter_gradient_parity"].values()]
    return {"status": [value["status"] for value in values],
            "median_ms": statistics.median(value["median_ms"] for value in values),
            "round_medians_ms": [value["median_ms"] for value in values],
            "peak_allocated": max(row["peak_allocated"] for row in measurements),
            "peak_reserved": max(row["peak_reserved"] for row in measurements),
            "maximum_gradient_relative_l2": max(gradients, default=None),
            "strict_failure_tensors": sum(value.get("strict_elementwise_gradient_failures", 0) for value in values),
            "inputs": [str(path) for path in paths]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(128, 512, 2048))
    args = parser.parse_args()
    result = {"no_cuda_graph": True, "external_tp_libraries": False,
              "precision": "FP32, AMP/TF32 off", "rows": []}
    lines = ["# 原生训练融合同卡配对", "",
             "完整 E/F 损失、混合二阶反传、AdamW；三批变化输入，正反顺序各一轮。",
             "CUDA Graph 全程禁用，DPA 使用实际可微 compile 路径。", "",
             "| 原子数 / 50 邻居 | 对照 ms | 融合 ms | DPA compile ms | DPA/融合 | 融合 reserved GB | DPA reserved GB |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in args.sizes:
        folder = args.root/f"n{n}"
        row = {"atoms": n, "control": case(folder, "control"),
               "fused": case(folder, "fused"), "dpa": case(folder, "dpa")}
        if all("median_ms" in row[name] for name in ("control", "fused", "dpa")):
            c, f, d = (row[name] for name in ("control", "fused", "dpa"))
            row["speedup_over_control"] = c["median_ms"]/f["median_ms"]
            row["speedup_over_dpa"] = d["median_ms"]/f["median_ms"]
            row["reserved_reduction_vs_dpa"] = 1-f["peak_reserved"]/d["peak_reserved"]
            row["target_2x_and_lower_reserved_met"] = (row["speedup_over_dpa"] >= 2
                                                      and f["peak_reserved"] < d["peak_reserved"])
            lines.append(f'| {n} | {c["median_ms"]:.3f} | {f["median_ms"]:.3f} | {d["median_ms"]:.3f} | '
                         f'{row["speedup_over_dpa"]:.3f}× | {f["peak_reserved"]/1e9:.3f} | {d["peak_reserved"]/1e9:.3f} |')
        else:
            lines.append(f"| {n} | 未完成 | 未完成 | 未完成 | — | — | — |")
        result["rows"].append(row)
    lines += ["", "小的逐元素偏差依用户确认保留但不阻塞计时；原输出、逐张量梯度范数、",
              "非有限值及首步优化器更新检查保持强制。逐项异常与每次计时保留在原始 JSON。",
              "本表不把显存下降等同于吞吐目标已完成。", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"native_training_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.out/"NATIVE_TRAINING_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
