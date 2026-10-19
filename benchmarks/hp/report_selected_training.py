"""Remote-only reduction of qualified native training / compiled DPA pairs."""
import argparse
import json
from pathlib import Path
from report_native_training_release import case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(2048, 4096))
    parser.add_argument("--qualification-name", help="Independent original-formula qualification JSON, before any performance ratio is accepted")
    parser.add_argument("--qualification-root", type=Path, help="Read immutable original qualification from a previous same-source run")
    parser.add_argument("--timed-steps", type=int, default=20)
    args = parser.parse_args()
    rows = []
    lines = ["# 训练主线融合复测", "",
             "完整 E/F、混合反传与 AdamW；每步重新构建 autograd，CUDA Graph 禁用。",
             f"同卡正反顺序各一轮，每轮 {args.timed_steps} 次，三批变化输入；FP32，AMP/TF32 off。", "",
             "| 原子数 / 50 邻居 | CHORUS ms | DPA compile ms | DPA / CHORUS | CHORUS reserved GB | DPA reserved GB |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in args.sizes:
        fused, dpa = case(args.root/f"n{n}", "fused"), case(args.root/f"n{n}", "dpa")
        row = {"atoms": n, "fused": fused, "dpa": dpa}
        qualified = all(status in ("PASS", "PASS_WITH_ELEMENTWISE_EXCEPTIONS")
                        for status in fused.get("status", [])) and len(fused.get("status", [])) == 2
        if not qualified and args.qualification_name:
            path = (args.qualification_root or args.root)/f"n{n}"/f"{args.qualification_name}.json"
            if path.is_file():
                qualification = json.loads(path.read_text())
                if (qualification.get("performance_qualified") and len(qualification.get("qualification", [])) == 3
                        and qualification.get("status") in ("PASS", "PASS_WITH_ELEMENTWISE_EXCEPTIONS")):
                    identity = ("checkpoint", "source", "batches", "trainability", "trainable_parameters",
                                "parameters", "tp_coefficient_trainability", "arithmetic", "plan")
                    paired = [json.loads(Path(value).read_text()) for value in fused["inputs"]]
                    qualified = all(all(value.get(key) == qualification.get(key) for key in identity)
                                    for value in paired)
                    row["independent_qualification"] = str(path)
                    row["independent_qualification_identity_matches"] = qualified
                    row["qualification_gradient_max_relative_l2"] = max(
                        error["relative_l2"] for batch in qualification["qualification"]
                        for error in batch["parameter_gradient_parity"].values())
        row["original_formula_qualified"] = qualified
        if all("median_ms" in v for v in (fused, dpa)):
            row["speedup_over_dpa"] = dpa["median_ms"]/fused["median_ms"]
            row["paired_round_speedups"] = [d/f for d, f in zip(dpa["round_medians_ms"], fused["round_medians_ms"])]
            row["reserved_reduction_vs_dpa"] = 1-fused["peak_reserved"]/dpa["peak_reserved"]
            row["throughput_and_memory_target_met"] = (row["speedup_over_dpa"] >= 2 and
                                                       fused["peak_reserved"] < dpa["peak_reserved"])
            row["target_met"] = qualified and row["throughput_and_memory_target_met"]
            row["every_paired_round_at_least_2x"] = all(ratio >= 2 for ratio in row["paired_round_speedups"])
            lines.append(f'| {n} | {fused["median_ms"]:.3f} | {dpa["median_ms"]:.3f} | '
                         f'{row["speedup_over_dpa"]:.3f}× | {fused["peak_reserved"]/1e9:.3f} | {dpa["peak_reserved"]/1e9:.3f} |')
        else:
            lines.append(f"| {n} | 未完成 | 未完成 | — | — | — |")
        rows.append(row)
    lines += ["", "原始输出、逐张量梯度范数、有限值及优化器更新检查保留。",
              "原式 oracle 的 checkpoint/CPU 卸载仅用于未计时校验，不进入 GPU 训练 step。",
              "小的逐元素舍入异常单独记录，不改变强制验收阈值。", ""]
    if args.qualification_name:
        lines += ["原式资格在独立进程中完成，计时使用同卡正反配对；报告核对 checkpoint、",
                  "源码路径、全部输入批次元数据、参数 trainability、执行计划及算术设置一致。",
                  "数值资格进程的单步计时不用于本表，容量筛选的原始标记不被改写。", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"selected_training_summary.json").write_text(json.dumps({"rows": rows}, indent=2)+"\n")
    (args.out/"SELECTED_TRAINING_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
