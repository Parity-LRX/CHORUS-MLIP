"""Reduce large-training raw records inside sbatch, retaining every OOM/failure."""
import argparse
import json
from pathlib import Path
import statistics


GOOD = {"PASS", "PASS_WITH_ELEMENTWISE_EXCEPTIONS", "DIAGNOSTIC_NO_LARGE_REFERENCE"}
IDENTITY = ("checkpoint", "source", "trainability", "parameters", "trainable_parameters",
            "tp_coefficient_trainability", "arithmetic", "plan", "architecture", "backward_targets", "gradient_execution")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {"status": "MISSING"}


def aggregate(paths, native, expected_samples):
    records = [read(path) for path in paths]
    result = {"inputs": list(map(str, paths)), "statuses": [r["status"] for r in records]}
    if not all(r["status"] in GOOD and "median_ms" in r for r in records):
        return result, records
    for record in records:
        policy = record["graph_execution_policy"]
        if any(policy[key] for key in ("capture", "replay", "inductor_cudagraphs")):
            raise AssertionError("CUDA Graph enabled in a purported no-graph result")
        measured = record["measurements"][0] if native else record
        if len(measured["samples"]) != expected_samples:
            raise AssertionError("Wrong number of complete timed training updates")
        if native and (not measured["fresh_autograd_graph_per_step"] or
                       not measured["post_timing_parameters_and_gradients_finite"]):
            raise AssertionError("Native training-step integrity check failed")
    medians = [record["median_ms"] for record in records]
    memory = [record["measurements"][0] if native else record for record in records]
    result.update(median_ms=statistics.median(medians), round_medians_ms=medians,
        peak_allocated=max(r["peak_allocated"] for r in memory),
        peak_reserved=max(r["peak_reserved"] for r in memory),
        trainable_parameters=records[0]["trainable_parameters"])
    if native:
        if any(any(record.get(key) != records[0].get(key) for key in IDENTITY) for record in records):
            raise AssertionError("Native paired rounds have different model/execution identities")
        result["atoms_per_second"] = statistics.median(
            r["measurements"][0]["atoms_per_second_sequence"] for r in records)
    return result, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--result-set")
    parser.add_argument("--case-result-set", nargs="+", default=[], metavar='CASE=SET',
                        help="Use a separately paired result set for selected cases without copying raw records")
    parser.add_argument("--cases", nargs="+", default=["atom_scale", "l3", "c256", "d4", "l3c192d3", "l3c256d4"])
    parser.add_argument("--target-speedup", type=float, default=2.)
    parser.add_argument("--qualification-root", type=Path, action="append", default=[],
                        help="Separate untimed original-formula oracle records, preserving earlier OOMs")
    args = parser.parse_args()
    overrides = dict(value.split('=', 1) for value in args.case_result_set)
    if set(overrides)-set(args.cases):
        raise ValueError('Result-set override names an unrequested case')
    rows, manifests, case_statuses = [], [], []
    lines = ["# 大原子数与大模型：无 CUDA Graph 训练", "",
        "A100 40 GB，50 cutoff 邻居，严格 FP32；AMP/TF32 关闭。每步包含能量、保守力、",
        "E/F 损失、混合二阶反传与 AdamW 更新。三个变化批次，正反顺序各 50 步；",
        "邻居表准备不计时。CUDA Graph API 及 Inductor cudagraphs 均禁用。", "",
        "原子数扩展使用原 Persistent 训练 checkpoint；模型扩展复用固定构造器初始化，",
        "卷积 TP 系数保持可训练。两家通道定义不同，以下是实现吞吐而非等精度配对。", "",
        "| 配置 | scope | 原子数 | 参数 M：CHORUS / DPA | CHORUS ms | DPA compile ms | DPA/CHORUS | reserved GB：CHORUS / DPA | 两轮及显存达标 | 原式梯度核验 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for name in args.cases:
        result_set = overrides.get(name, args.result_set or ("large_training_v1" if name == "atom_scale" else "large_training_v2"))
        folder = args.root / "results" / result_set / name
        manifest = read(folder / "manifest.json")
        if manifest.get("status") == "MISSING":
            case_statuses.append({"case": name, "status": "MISSING"})
            continue
        manifests.append(manifest)
        case_status = read(folder / "case_status.json")
        case_statuses.append({"case": name, "status": case_status["status"]})
        for n in manifest["sizes"]:
            for scope in manifest["scopes"]:
                match = (manifest.get('parameter_matching') or {}).get(scope)
                dpa_label = 'dpa_'+scope if match else 'dpa'
                dpa, dpa_values = aggregate([folder / f"n{n}" / f"{dpa_label}_r{r}.json" for r in (0, 1)],
                                   False, manifest["repeats"])
                native, values = aggregate([folder / f"n{n}" / f"{scope}_r{r}.json" for r in (0, 1)],
                                            True, manifest["repeats"])
                expected_targets = manifest.get("backward_targets", "all_leaves")
                for value in [*dpa_values, *values]:
                    if "median_ms" in value and value.get("backward_targets", "all_leaves") != expected_targets:
                        raise AssertionError("DPA and CHORUS must use identical backward target policies")
                small = read(folder / "qualification128" / f"{scope}.json")
                full_path = folder / f"qualification{n}" / f"{scope}.json"
                full = read(full_path)
                original_status = full["status"]
                attempts = [dict(path=str(full_path), status=full['status'])]
                selected_alternative = False
                for qualification_root in args.qualification_root:
                    alternative_path = qualification_root / name / f"n{n}" / f"{scope}.json"
                    alternative = read(alternative_path)
                    identity_matches = ('median_ms' in native and
                        all(all(value.get(key) == alternative.get(key) for key in IDENTITY) for value in values))
                    attempts.append(dict(path=str(alternative_path), status=alternative['status'],
                        timed_identity_matches=identity_matches))
                    if (alternative["status"] in ("PASS", "PASS_WITH_ELEMENTWISE_EXCEPTIONS")
                            and len(alternative.get("qualification", [])) == 3 and identity_matches
                            and not selected_alternative):
                        full_path, full = alternative_path, alternative
                        selected_alternative = True
                qualified = (full["status"] in ("PASS", "PASS_WITH_ELEMENTWISE_EXCEPTIONS") and
                    len(full.get("qualification", [])) == 3 and "median_ms" in native and
                    all(all(value.get(key) == full.get(key) for key in IDENTITY) for value in values))
                row = dict(case=name, scope=scope, atoms=n, native=native, dpa=dpa,
                    parameter_matching=match,
                    small128_qualification_status=small["status"],
                    same_size_qualification_status=full["status"],
                    initial_same_size_qualification_status=original_status,
                    qualification_attempts=attempts,
                    same_size_original_formula_qualified=qualified,
                    original_qualification_file=str(full_path) if full_path.exists() else None,
                    slurm_job=manifest["slurm_job"], CUDA_VISIBLE_DEVICES=manifest["CUDA_VISIBLE_DEVICES"])
                if manifest.get('native_baseline_variant'):
                    baseline, baseline_values = aggregate([
                        folder/f'n{n}'/f'baseline_{scope}_r{r}.json' for r in (0, 1)], True, manifest['repeats'])
                    row['native_baseline'] = baseline
                    if 'median_ms' in native and 'median_ms' in baseline:
                        for value in baseline_values:
                            for key in ('checkpoint', 'architecture', 'trainable_parameters', 'arithmetic', 'backward_targets'):
                                if value.get(key) != values[0].get(key):
                                    raise AssertionError('Native baseline changed the model or training arithmetic')
                        row['native_speedup_over_previous_schedule'] = baseline['median_ms']/native['median_ms']
                if match:
                    if ('trainable_parameters' in native and native['trainable_parameters'] != match['chorus_parameters'] or
                            'trainable_parameters' in dpa and dpa['trainable_parameters'] != match['dpa_parameters']):
                        raise AssertionError('Timed parameters differ from preselected matching counts')
                    for value in dpa_values:
                        if 'median_ms' in value:
                            descriptor = value['descriptor']
                            if (descriptor['channels'] != match['dpa_channels'] or descriptor['lmax'] != match['lmax'] or
                                    descriptor['n_blocks'] != match['layers']):
                                raise AssertionError('Timed DPA architecture changed from the parameter match')
                if qualified:
                    row["gradient_max_relative_l2"] = max(error["relative_l2"] for batch in full["qualification"]
                        for error in batch["parameter_gradient_parity"].values())
                if "median_ms" in native and "median_ms" in dpa:
                    row["speedup_over_dpa"] = dpa["median_ms"] / native["median_ms"]
                    row["paired_round_speedups"] = [d / c for d, c in
                        zip(dpa["round_medians_ms"], native["round_medians_ms"])]
                    row["reserved_reduction_vs_dpa"] = 1 - native["peak_reserved"] / dpa["peak_reserved"]
                    row["throughput_and_memory_target_met"] = (min(row["paired_round_speedups"]) >= args.target_speedup and
                        native["peak_reserved"] < dpa["peak_reserved"] and
                        native["peak_allocated"] < dpa["peak_allocated"])
                rows.append(row)
                def timing(value):
                    return f'{value["median_ms"]:.3f}' if "median_ms" in value else "/".join(value["statuses"])
                native_memory = f'{native["peak_reserved"]/1e9:.3f}' if "peak_reserved" in native else "—"
                dpa_memory = f'{dpa["peak_reserved"]/1e9:.3f}' if "peak_reserved" in dpa else "—"
                params = f'{native["trainable_parameters"]/1e6:.3f}' if "trainable_parameters" in native else "—"
                params += ' / '+(f'{dpa["trainable_parameters"]/1e6:.3f}' if 'trainable_parameters' in dpa else '—')
                ratio = f'{row["speedup_over_dpa"]:.3f}×' if "speedup_over_dpa" in row else "—"
                if qualified:
                    verification = "本尺寸通过"
                elif small["status"] in ("PASS", "PASS_WITH_ELEMENTWISE_EXCEPTIONS"):
                    suffix = "本尺寸参考 OOM" if full["status"] == "OOM" else \
                        f'本尺寸 {full["status"]}' if full["status"] in ("FAIL", "TIMEOUT") else "本尺寸未核验"
                    verification = "128 通过；"+suffix
                else:
                    verification = "见既有 2048/4096 核验" if name == "atom_scale" else small["status"]
                target = "是" if row.get("throughput_and_memory_target_met") else "否" if "median_ms" in dpa and "median_ms" in native else "—"
                lines.append(f'| {name} | {scope} | {n} | {params} | {timing(native)} | {timing(dpa)} | '
                             f'{ratio} | {native_memory} / {dpa_memory} | {target} | {verification} |')
    lines += ["", "模型矩阵：l3=C128/lmax3/2层；c256=C256/lmax2/2层；d4=C128/lmax2/4层；",
        "l3c192d3=C192/lmax3/3层；l3c256d4=C256/lmax3/4层。", ""]
    if any(manifest.get('parameter_matching') for manifest in manifests):
        lines += ['参数量匹配：保持双方同 lmax、同 interaction/块数，DPA 仅改变通道数。',
            '通道在预先指定的 32..160、步长 4 网格上按实际参数量最近选择，不用速度挑模型。',
            'Persistent 与 Final 分别匹配；参数相近不是等精度声明。', '',
            '| 配置 | scope | CHORUS C | DPA C | DPA 参数相对差 |',
            '| --- | --- | ---: | ---: | ---: |']
        for manifest in manifests:
            for scope, match in (manifest.get('parameter_matching') or {}).items():
                lines.append(f"| {manifest['case']} | {scope} | {match['chorus_channels']} | {match['dpa_channels']} | {match['signed_relative_mismatch']:+.2%} |")
        lines.append('')
    else:
        lines += ['旧固定配置 DPA 使用 C32/C64/C32/C48/C64；该配对不匹配参数量。', '']
    if any('native_baseline' in row for row in rows):
        lines += ['同卡旧原生调度与本轮选择单列，避免把对照模型改变产生的速度比当作算子优化收益。', '',
            '| 配置 | scope | 原子数 | 旧原生 ms | 本轮 ms | 原生调度加速 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
        for row in rows:
            if 'native_speedup_over_previous_schedule' in row:
                lines.append(f"| {row['case']} | {row['scope']} | {row['atoms']} | {row['native_baseline']['median_ms']:.3f} | {row['native']['median_ms']:.3f} | {row['native_speedup_over_previous_schedule']:.3f}× |")
        lines.append('')
    lines += [
        "DPA 使用真实可微 compile 训练路径。Triton inference level 2 已配置，但上游训练分支",
        "不执行该 value kernel，不能表述为训练中启用了 level-2 推理核。", "",
        "OOM 与未完成值不进入速度比。大尺寸计时、128 原子原式核验和本尺寸原式核验分栏；",
        "原式核验进程不参与配对计时。小的逐元素舍入差异记录但不阻断范数/更新准则。", ""]
    lines += [f"本表目标为两轮均至少 {args.target_speedup:g}× 且 allocated/reserved 峰值显存均更少；各次配对使用相同 backward target。", ""]
    complete = len(case_statuses) == len(args.cases) and all(s["status"].startswith("DONE") for s in case_statuses)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = dict(status="COMPLETE" if complete else "PARTIAL", rows=rows,
                   case_statuses=case_statuses, manifests=manifests, target_speedup=args.target_speedup)
    (args.out / "large_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "LARGE_TRAINING_RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    if not complete and not args.allow_partial:
        raise SystemExit("Not all requested cases have reached a terminal status")


if __name__ == "__main__":
    main()
