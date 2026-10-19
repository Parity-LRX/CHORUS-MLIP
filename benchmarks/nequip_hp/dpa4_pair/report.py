"""Aggregate paired records remotely, without mixing hardware or scopes."""
import argparse
import json
import os
from pathlib import Path
import statistics


def main(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Aggregation is also a remote batch task")
    source=args.root/"results"/args.run
    report=dict(status="PASS",run=args.run,pairs={},results=[])
    lines=["# NequIP–CHORUS OEQ 对 DPA4 compile / Triton level 2", "",
        "同卡 A100，禁用 CUDA Graph/AOTI；网络/力 FP32、TF32/AMP off。",
        "每组在一张卡上以 NequIP→DPA、DPA→NequIP 两轮顺序执行，进程隔离。",
        "推理为完整 E/F/virial；训练含相同 E/F/virial 损失、混合反传与 AdamW。",
        "输入为相同三组周期构型、50 邻居与 4 Å cutoff；排除邻居搜索，计入输入复制与几何重算。", "",
        "DPA4 推理核验 compile 和 level-2 value 路径；训练开启 compile，level-2 推理专用 value 核不参与训练。",
        "这是同角动量/层数、近参数量的执行效率比较，不是等精度或重新训练后的物理预测比较。", "",
        "## 参数量配对", "",
        "| lmax / 层数 | scope | NequIP C / 参数量 | DPA C / 参数量 | DPA 参数差 |",
        "| --- | --- | ---: | ---: | ---: |"]
    for case in ("l2d2","l3d3"):
        for scope in ("final","persistent"):
            directory=source/case/scope
            commands=json.loads((directory/"commands.json").read_text())
            if commands["status"]!="DONE" or commands.get("pilot") or commands["rounds"]!=2:
                raise AssertionError("Missing complete formal pair")
            records={}
            uuids=set()
            t2_proof=None
            for engine in ("nequip","dpa"):
                for task in ("inference","train"):
                    records[engine,task]=[]
                    for round_id in (0,1):
                        value=json.loads((directory/f"{engine}_{task}_r{round_id}.json").read_text())
                        if value["status"] not in ("PASS","PARTIAL_CAPACITY"):
                            raise AssertionError("Failed numerical or runtime qualification")
                        if value["graph_execution_policy"]["capture"] or value["graph_execution_policy"]["replay"]:
                            raise AssertionError("CUDA Graph is not permitted")
                        uuids.add(value["gpu_uuid"])
                        if engine=="dpa" and (not value["compiled_core_slots"] or
                            not all(m["configured_level"]==2 and m["value_path"]=="_TritonSO2ValuePath"
                                    for m in value["triton_modules"])):
                            raise AssertionError("DPA compile / T2 configuration not supported")
                        if engine=="dpa" and task=="inference" and round_id==0:
                            proof=next(r for r in value["results"] if r["atoms"]==1024)
                            # Inductor can inline torch.library.triton_op, so
                            # runtime profiler keys become concrete kernel names.
                            profile=directory/"dpa_inference_r0.profile.txt"
                            profile_text=profile.read_text()
                            markers=[name for name in ("_rotate_mix_fwd_kernel",
                                "_stack_gemm_m0_kernel","_stack_gemm_m1_kernel") if name in profile_text]
                            if len(markers)!=3 and not proof.get("level2_profiler_events"):
                                raise AssertionError("No actual level-2 custom-op execution in the inference profile")
                            t2_proof=dict(profile=str(profile),concrete_kernels=markers,
                                custom_op_events=proof.get("level2_profiler_events",[]))
                        records[engine,task].append(value)
            if len(uuids)!=1 or "unavailable" in uuids:
                raise AssertionError("Paired records must identify one physical GPU")
            metadata=records["nequip","inference"][0]["metadata"]
            report["pairs"][f"{case}/{scope}"]=dict(metadata=metadata,gpu_uuid=next(iter(uuids)),
                job=commands["job"],inference_t2_runtime_proof=t2_proof)
            lines.append(f'| {metadata["lmax"]} / {metadata["layers"]} | {scope} | '
                f'{metadata["nequip_channels"]} / {metadata["nequip_parameters"]:,} | '
                f'{metadata["dpa_channels"]} / {metadata["dpa_parameters"]:,} | '
                f'{metadata["signed_parameter_mismatch"]:+.2%} |')
            for task in ("inference","train"):
                for n in ([128,1024,4096] if task=="inference" else [128,1024]):
                    row=dict(case=case,scope=scope,lmax=metadata["lmax"],layers=metadata["layers"],task=task,atoms=n)
                    for engine in ("nequip","dpa"):
                        values=[next(r for r in record["results"] if r["atoms"]==n) for record in records[engine,task]]
                        if not all(v["status"]=="PASS" for v in values):
                            row[engine]=dict(status="OOM",rounds=values)
                            continue
                        samples=[sample for v in values for sample in v["samples_ms"]]
                        row[engine]=dict(status="PASS",median_ms=statistics.median(samples),
                            round_medians_ms=[v["median_ms"] for v in values],samples_ms=samples,
                            peak_allocated=max(v["peak_allocated"] for v in values),
                            peak_reserved=max(v["peak_reserved"] for v in values))
                    if all(row[e]["status"]=="PASS" for e in ("nequip","dpa")):
                        row["speedup_dpa_over_nequip"]=row["dpa"]["median_ms"]/row["nequip"]["median_ms"]
                        row["reserved_dpa_over_nequip"]=row["dpa"]["peak_reserved"]/row["nequip"]["peak_reserved"]
                    report["results"].append(row)
    def table(rows):
        result=["| lmax / 层数 | scope | N | task | NequIP ms | DPA4 ms | NequIP 加速 | reserved GB NequIP / DPA |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
        for r in rows:
            if "speedup_dpa_over_nequip" not in r:
                result.append(f'| {r["lmax"]} / {r["layers"]} | {r["scope"]} | {r["atoms"]} | {r["task"]} | OOM/见 JSON | — | — | — |')
            else:
                a,b=r["nequip"],r["dpa"]
                result.append(f'| {r["lmax"]} / {r["layers"]} | {r["scope"]} | {r["atoms"]} | {r["task"]} | '
                    f'{a["median_ms"]:.3f} | {b["median_ms"]:.3f} | {r["speedup_dpa_over_nequip"]:.3f}× | '
                    f'{a["peak_reserved"]/1e9:.3f} / {b["peak_reserved"]/1e9:.3f} |')
        return result
    lines += ["", "## 1024 原子", ""]+table([r for r in report["results"] if r["atoms"]==1024])
    lines += ["", "## 全部尺寸", ""]+table(report["results"])
    lines += ["", "加速比 = DPA4 耗时 / NequIP–CHORUS 耗时；低于 1 表示 NequIP 较慢。",
        "两轮逐次 walltime 合并取中位数；显存为两轮峰值中的最大值，GB 为十进制。",
        "allocated、逐轮中位数、逐次计时、原式输出及参数梯度核验均保留在 JSON。", ""]
    output=args.root/"reports"/args.run
    output.mkdir(parents=True,exist_ok=False)
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"NEQUIP_DPA4_MATCHED.md").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--run",required=True)
    main(parser.parse_args())
