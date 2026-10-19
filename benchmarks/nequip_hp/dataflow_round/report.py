"""Aggregate native-dataflow regressions only inside the designated batch job."""
import argparse
import json
import os
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


LABELS = ("no_chorus", "previous", "optimized", "dpa")
CASES = ("l2d2", "l3d3")
SCOPES = ("final", "persistent")


def read(path):
    return json.loads(path.read_text())


def combine(records, n):
    values = [next(item for item in record["results"] if item["atoms"] == n)
              for record in records]
    if not all(value["status"] == "PASS" for value in values):
        return dict(status="CAPACITY_LIMIT", rounds=values)
    samples = [sample for value in values for sample in value["samples_ms"]]
    return dict(status="PASS", median_ms=statistics.median(samples), samples_ms=samples,
                round_medians_ms=[value["median_ms"] for value in values],
                peak_allocated=max(value["peak_allocated"] for value in values),
                peak_reserved=max(value["peak_reserved"] for value in values))


def timing_table(rows):
    lines = ["| L / 层数 | scope | N | task | 无 CHORUS ms | 优化前 ms | 优化后 ms | 本轮加速 | DPA4 ms | 相对 DPA4 |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        def latency(label):
            value = row[label]
            return f'{value["median_ms"]:.3f}' if value["status"] == "PASS" else "OOM"
        gain = f'{row["gain_over_previous"]:.3f}×' if "gain_over_previous" in row else "—"
        dpa = f'{row["gain_over_dpa"]:.3f}×' if "gain_over_dpa" in row else "—"
        lines.append(f'| {row["lmax"]} / {row["layers"]} | {row["scope"]} | {row["atoms"]} | '
                     f'{row["task"]} | {latency("no_chorus")} | {latency("previous")} | '
                     f'{latency("optimized")} | {gain} | {latency("dpa")} | {dpa} |')
    return lines


def memory_table(rows):
    lines = ["| L / 层数 | scope | N | task | allocated GB 无 / 前 / 后 / DPA | reserved GB 无 / 前 / 后 / DPA | 本轮 allocated / reserved 降幅 |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for row in rows:
        def memory(key):
            return " / ".join(f'{row[label][key]/1e9:.3f}' if row[label]["status"] == "PASS" else "OOM"
                              for label in LABELS)
        reductions = " / ".join(f'{row[key]:.1%}' if key in row else "—"
                                for key in ("allocated_reduction", "reserved_reduction"))
        lines.append(f'| {row["lmax"]} / {row["layers"]} | {row["scope"]} | {row["atoms"]} | '
                     f'{row["task"]} | {memory("peak_allocated")} | {memory("peak_reserved")} | {reductions} |')
    return lines


def main(args):
    if (not os.environ.get("SLURM_JOB_ID") or
            os.environ.get("SLURM_JOB_PARTITION") != "GPU-A100" or
            "XHPC-A100" not in os.environ.get("SLURM_JOB_NODELIST", "")):
        raise RuntimeError("Report aggregation also requires the designated sbatch allocation")
    source = args.root / "results" / args.run
    unit_files = sorted((args.root / "results" / args.unit_run).glob("unit_*.xml"))
    if not unit_files:
        raise AssertionError("Missing complete unit-test record")
    units = []
    for path in unit_files:
        tree = ET.parse(path)
        suites = list(tree.getroot().iter("testsuite"))
        if not suites or any(int(suite.get("failures", 0)) or int(suite.get("errors", 0))
                             for suite in suites):
            raise AssertionError("Unit qualification failed")
        units.append(dict(path=str(path), tests=sum(int(suite.get("tests", 0)) for suite in suites)))
    qualification_path = args.root / "results" / args.qualification_run / "optimization_qualification.json"
    qualification = read(qualification_path)
    if qualification["status"] != "PASS":
        raise AssertionError("Full-output / mixed-gradient qualification did not pass")
    report = dict(status="PASS", run=args.run, unit_tests=units,
                  formula_qualification=str(qualification_path), pairs={}, results=[])
    lines = ["# NequIP–CHORUS 原生数据流优化", "",
             "同一份模型配置与初始化；不改 rank、通道、CG 路径、训练损失或优化器。",
             "本轮将独立 irrep 切片改为一次 split，按原路径执行通道 GEMM，",
             "把线性路径归一化放到权重侧，把线性邻居归一化放到聚合后的节点侧。",
             "二次 Hermitian density 的归一化保持原式。", "",
             "## 比较协议", "",
             "每组在一张物理 A100 上依次测无 CHORUS → 优化前 → 优化后 → DPA4，再反序测第二轮。",
             "所有方案采用独立进程，使用相同三组变化周期构型、4 Å cutoff 与 50 邻居。",
             "推理包括完整 E/F/virial；训练包括同一 E/F/virial 损失、混合反传和 AdamW 更新。",
             "预构建邻居拓扑；计入输入张量复制、几何重算和完整输出。",
             "网络与力为 FP32，TF32/AMP 关闭；原生归约 dtype 保留。无 CUDA Graph、无 AOTI。",
             "DPA4 推理实际执行 compile + Triton level-2 value 内核；训练使用 compile，",
             "上游推理专用 level-2 value 内核不参与训练。", "",
             "无 CHORUS 基线保留同通道、同层数及所有公共骨干 checkpoint 张量，使用相同通用数据流优化与 OEQ 中性层融合。",
             "优化前/后使用完全相同的完整 checkpoint；DPA4 保持上一轮按参数量选定的固定配对。", "",
             "## 参数与同卡记录", "",
             "| L / 层数 | scope | NequIP C | 参数量 无 / CHORUS | DPA C / 参数量 | DPA 参数差 | Job |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for case in CASES:
        for scope in SCOPES:
            directory = source / case / scope
            manifest = read(directory / "commands.json")
            if (manifest["status"] != "DONE" or manifest["mode"] != "formal" or
                    len(manifest["commands"]) != 16 or
                    any(item["status"] != "DONE" or item["exit_code"] != 0
                        for item in manifest["commands"])):
                raise AssertionError(f"Incomplete formal regression: {case}/{scope}")
            if manifest["engine_order"] != [list(LABELS), list(reversed(LABELS))]:
                raise AssertionError("Formal forward/reverse order changed")
            records = {}
            uuids = set()
            t2_proof = None
            for label in LABELS:
                for task in ("inference", "train"):
                    records[label, task] = []
                    for round_id in (0, 1):
                        value = read(directory / f"{label}_{task}_r{round_id}.json")
                        if value["status"] not in ("PASS", "PARTIAL_CAPACITY"):
                            raise AssertionError("Numerical/runtime qualification failed")
                        if value["graph_execution_policy"]["capture"] or value["graph_execution_policy"]["replay"]:
                            raise AssertionError("CUDA Graph execution detected")
                        uuids.add(value["gpu_uuid"])
                        if label == "dpa":
                            if (not value["compiled_core_slots"] or not value["triton_modules"] or
                                    not all(module["configured_level"] == 2 and
                                            module["value_path"] == "_TritonSO2ValuePath"
                                            for module in value["triton_modules"])):
                                raise AssertionError("DPA compile/T2 configuration is incomplete")
                            if task == "inference" and round_id == 0:
                                proof = next(item for item in value["results"] if item["atoms"] == 1024)
                                profile = directory / "dpa_inference_r0.profile.txt"
                                content = profile.read_text()
                                markers = [name for name in ("_rotate_mix_fwd_kernel", "_stack_gemm_m0_kernel",
                                                            "_stack_gemm_m1_kernel") if name in content]
                                if len(markers) != 3 and not proof.get("level2_profiler_events"):
                                    raise AssertionError("No runtime evidence of the level-2 value path")
                                t2_proof = dict(profile=str(profile), concrete_kernels=markers,
                                                custom_op_events=proof.get("level2_profiler_events", []))
                        records[label, task].append(value)
            if len(uuids) != 1 or "unavailable" in uuids:
                raise AssertionError("The regression must use one identified physical GPU")
            metadata = records["optimized", "inference"][0]["metadata"]
            expected = metadata["nequip_parameters"]
            for label in ("previous", "optimized"):
                if any(record["parameters"] != expected for task in ("inference", "train")
                       for record in records[label, task]):
                    raise AssertionError("CHORUS parameter count changed")
            bare = records["no_chorus", "inference"][0]["parameters"]
            report["pairs"][f"{case}/{scope}"] = dict(metadata=metadata, gpu_uuid=next(iter(uuids)),
                job=manifest["job"], no_chorus_parameters=bare, inference_t2_runtime_proof=t2_proof,
                qualifications={f"{label}/{task}": [record["qualification"] for record in records[label, task]]
                                for label in LABELS for task in ("inference", "train")})
            lines.append(f'| {metadata["lmax"]} / {metadata["layers"]} | {scope} | '
                         f'{metadata["nequip_channels"]} | {bare:,} / {expected:,} | '
                         f'{metadata["dpa_channels"]} / {metadata["dpa_parameters"]:,} | '
                         f'{metadata["signed_parameter_mismatch"]:+.2%} | {manifest["job"]} |')
            for task in ("inference", "train"):
                for n in ([128, 1024, 4096] if task == "inference" else [128, 1024]):
                    row = dict(case=case, scope=scope, lmax=metadata["lmax"], layers=metadata["layers"],
                               task=task, atoms=n)
                    for label in LABELS:
                        row[label] = combine(records[label, task], n)
                    opt = row["optimized"]
                    if opt["status"] == "PASS":
                        for label, key in (("previous", "gain_over_previous"), ("dpa", "gain_over_dpa"),
                                           ("no_chorus", "gain_over_no_chorus")):
                            if row[label]["status"] == "PASS":
                                row[key] = row[label]["median_ms"] / opt["median_ms"]
                        if row["previous"]["status"] == "PASS":
                            for memory in ("allocated", "reserved"):
                                row[f"{memory}_reduction"] = 1 - opt[f"peak_{memory}"] / row["previous"][f"peak_{memory}"]
                    report["results"].append(row)
    selected = [row for row in report["results"] if row["atoms"] == 1024]
    lines += ["", "## 1024 原子：时间", ""] + timing_table(selected)
    lines += ["", "## 1024 原子：显存", ""] + memory_table(selected)
    lines += ["", "## 全部尺寸：时间", ""] + timing_table(report["results"])
    lines += ["", "## 全部尺寸：显存", ""] + memory_table(report["results"])
    lines += ["", "本轮加速 = 优化前耗时 / 优化后耗时；相对 DPA4 = DPA4 耗时 / 优化后耗时。",
              "比值大于 1 表示优化后更快；显存降幅负值表示该项增加。GB 为十进制。",
              "每轮预热 9 次，推理测 30 次、训练测 18 次；两轮样本合并取中位数。",
              "显存记录两轮最大峰值；逐次计时、逐轮中位数、allocated/reserved 与数值误差均保留在 JSON。",
              f'完整单元测试：{sum(item["tests"] for item in units)} 项通过；原式与混合梯度资格通过。', ""]
    output = args.root / "reports" / args.run
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "NEQUIP_NATIVE_DATAFLOW.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--unit-run", default="unit_v3")
    parser.add_argument("--qualification-run", default="qualify_v2")
    main(parser.parse_args())
