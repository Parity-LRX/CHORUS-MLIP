"""Create the numerical iteration report inside an XHPC sbatch allocation."""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('sbatch allocation required')
    def read(path):return json.loads(path.read_text()) if path.exists() else {}
    record = dict(report_job=os.environ['SLURM_JOB_ID'], persistent=[], final=[], diagnostics=[], tests={})
    lines = ['# 径向 gate/伴随融合迭代：中显存档与 Final 数值路径', '',
        '全部数值计算、计时、汇总通过 `sbatch / GPU-A100 / XHPC-A100`。'
        '固定原 checkpoint、全部参数、50 个截断邻居、FP32/TF32 off；virial 为 FP64。'
        '显存为十进制 GB，包含 CUDA Graph 私有池的 reserved。', '',
        '## Persistent：保留本次前向的深层 gate', '',
        '原生 TP/VJP 直接输出每边距离梯度，不物化完整 gate 梯度；'
        '深层保留 gate 值以避免重算，首层保留原径向网络并选用无多余输入伴随的调度。'
        '所有实时输入每次重新计算，不跨调用缓存 gate。默认最快档和全有界容量档均保留。', '',
        '| 原子数 | 配置 | ms | reserved GB | 相对同卡 DPA lower |',
        '| --- | --- | ---: | ---: | ---: |']
    public = args.root/'results/cached_gates_public_v16/persistent'
    for n in (10000, 32768):
        directory = public/f'n{n}'
        dpa = read(directory/'dpa4_lower.json')
        default = read(directory/'default.json')
        for label, filename in [('原最快默认', 'default'), ('保留 gate，tile=65536', 'public_cached_tiles_65536'),
                                ('保留 gate，tile=262144', 'public_cached_tiles_262144'),
                                ('DPA4-mini compile + 实际 Triton 2', 'dpa4_lower')]:
            path = directory/f'{filename}.json'
            data = read(path)
            if 'median_ms' not in data:
                lines.append(f'| {n} | {label} | {data.get("status", "MISSING")} | — | — |')
                continue
            ratio = dpa['median_ms']/data['median_ms']
            row = dict(atoms=n, label=label, file=str(path), median_ms=data['median_ms'],
                reserved_gb=data['peak_reserved']/1e9, speedup_vs_dpa_lower=ratio,
                parity=data.get('parity'), capture_validation=data.get('capture_validation'),
                plan=data.get('plan'))
            if filename.startswith('public_cached'):
                row.update(time_change_vs_default=data['median_ms']/default['median_ms']-1,
                    reserved_reduction_vs_default=1-data['peak_reserved']/default['peak_reserved'])
            record['persistent'].append(row)
            lines.append(f'| {n} | {label} | {data["median_ms"]:.3f} | {row["reserved_gb"]:.3f} | {ratio:.3f}× |')
    lines += ['', 'DPA 的完整 full/lower 入口输出一致性在计时尺寸检查，未加速原式另在 1024 原子核对；'
        'CHORUS 对照为登记的全尺寸原式 E/F/virial，32,768 原子使用已核验的完整依赖 halo。'
        '通过点同时检查改变坐标后的实时重放。Persistent 的两倍 DPA 目标尚未完成。', '',
        '65,536 原子保留 gate 档通过 eager 原式数值检查，但在 CUDA Graph 捕获时 OOM。'
        '该点不计作通过。全有界档继续用于更大图，原始失败和日志保留。', '',
        '| 65,536 原子配置 | 状态 / ms | reserved GB |', '| --- | ---: | ---: |']
    for name in ('default', 'public_cached_tiles_65536', 'public_cached_tiles_262144',
                 'public_destination_tiles_65536', 'public_destination_tiles_262144'):
        path = args.root/'results/cached_gates_public_v14/persistent/n65536'/f'{name}.json'
        data = read(path)
        lines.append(f'| {name} | '+(f'{data["median_ms"]:.3f} | {data["peak_reserved"]/1e9:.3f} |'
            if 'median_ms' in data else f'{data.get("status", "MISSING")} | — |'))
    lines += ['', '## Final：映射布局与力归约', '',
        '先逐项关闭相位、密度、通道、激活、元素映射、谐波、几何和大图调度，再做组合对照。'
        '恢复原通道/激活/元素映射，并使用稳定 FP32 力归约的候选接受同一个登记参照及阈值。', '',
        '| 对照 | 重复次数 | 力分量全部通过次数 | 单次超差分量范围 |', '| --- | ---: | ---: | ---: |']
    for directory in sorted((args.root/'results').glob('precision_ablation_*')):
        for path in sorted(directory.glob('*.json')):
            data = read(path)
            if not isinstance(data, dict) or 'case' not in data:continue
            failed = [r['parity']['forces']['failed_components'] for r in data.get('repeats', [])]
            if not failed:continue
            row = dict(case=data['case'], file=str(path), repeats=len(failed),
                       all_force_components_passed=sum(v == 0 for v in failed),
                       failures_by_repeat=failed,
                       max_bound_ratio_by_repeat=[r['parity']['forces']['max_bound_ratio'] for r in data['repeats']])
            record['diagnostics'].append(row)
            lines.append(f'| {data["case"]} | {len(failed)} | {row["all_force_components_passed"]} | {min(failed)}–{max(failed)} |')
    lines += ['', '该组合同时改变映射计算布局与最终力归约，尚不能把误差归于唯一一个算子。'
        '原默认路径的超差记录继续有效。以下为该显式候选的独立完整计时与原式/重放核验；'
        '不替换默认 Final，也不改写先前 10,000 原子的结果。', '',
        '| 32,768 原子路径 | 同卡点 | 状态 / ms | reserved GB | 相对同卡 DPA lower |',
        '| --- | --- | ---: | ---: | ---: |']
    for route, tag in (('原通道/激活/元素映射 + CSR 力', 'final_stable_maps_v17'),
                       ('原通道/激活 + HP 元素映射 + CSR 力', 'final_stable_selected_v20')):
        directory = args.root/'results'/tag/'final/n32768'
        dpa = read(directory/'dpa4_lower.json')
        for name in ('dpa4_full', 'dpa4_lower', 'default', 'none_forcecsr_tiles_65536'):
            path = directory/f'{name}.json'
            data = read(path)
            if 'median_ms' in data:
                ratio = dpa['median_ms']/data['median_ms']
                row = dict(route=route, label=name, file=str(path), median_ms=data['median_ms'],
                    reserved_gb=data['peak_reserved']/1e9, speedup_vs_dpa_lower=ratio,
                    plan=data.get('plan'), parity=data.get('parity'), capture_validation=data.get('capture_validation'))
                record['final'].append(row)
                lines.append(f'| {route} | {name} | {data["median_ms"]:.3f} | {row["reserved_gb"]:.3f} | {ratio:.3f}× |')
            else:lines.append(f'| {route} | {name} | {data.get("status", "MISSING")} | — | — |')
    xml = args.root/'results/tests_cached_release_v16.xml'
    if xml.exists():
        suites = ET.parse(xml).getroot().iter('testsuite')
        counts = [int(s.attrib.get(key, 0)) for s in suites for key in ('tests', 'failures', 'errors')]
        record['tests'] = dict(tests=sum(counts[0::3]), failures=sum(counts[1::3]), errors=sum(counts[2::3]), file=str(xml))
    heldout = read(args.root/'results/public_cached_v16/persistent_heldout.json')
    record['heldout'] = heldout
    lines += ['', '## 回归与接口', '',
        f'整套远端回归：{record["tests"].get("tests", "MISSING")} 项，'
        f'failure={record["tests"].get("failures", "MISSING")}，error={record["tests"].get("errors", "MISSING")}。'
        f'16 帧 T1x 留出 E/F：{heldout.get("status", "MISSING")}。'
        '覆盖固定/可微相位系数、第一及混合高阶导数、空边、通道尾部、实时参数更新、'
        '坐标/元素/拓扑/晶胞变化以及 CUDA Graph 状态失效。', '',
        '新增公开开关为 `radial_cache_gates=True`，要求同时开启 `radial_jvp`、'
        '`radial_gate_pipeline` 和 `radial_destination`。该轮针对冻结推理，训练吞吐计时未更新。', '',
        '原生启动块/路径分组筛选仅为算子级诊断，未形成明显收益，默认启动参数不变。'
        '模型和算子仍来自 ICTC-O3-HP，未引入外部 TP 库。', '',
        '基础设施记录：source_v10 首次同步漏掉谐波基资产，子进程在模型加载时退出；'
        '后续完整快照修正。作业 92217 在 DPA 阶段取消，用独立输出目录修正 Slurm '
        '逗号分隔环境参数的传递。两者均不计入数值通过或性能结果。', '']
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'CACHED_GATE_ITERATION.md').write_text('\n'.join(lines))
    (args.out/'iteration_summary.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out), tests=record['tests'], heldout=heldout.get('status'),
        persistent=[{k:v for k,v in row.items() if k in ('atoms','label','median_ms','reserved_gb','speedup_vs_dpa_lower',
            'time_change_vs_default','reserved_reduction_vs_default')} for row in record['persistent']],
        final=[{k:v for k,v in row.items() if k in ('label','median_ms','reserved_gb','speedup_vs_dpa_lower')} for row in record['final']]), indent=2), flush=True)


if __name__ == '__main__':main()
