"""Within-sbatch arithmetic and evidence table for the architecture matrix."""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from architecture_cases import CASES


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def dpa_valid(value, blocks):
    return bool(value and 'median_ms' in value and value.get('DP_COMPILE_INFER') == '1'
        and value.get('DP_TRITON_INFER') == '2' and value.get('should_use_compile')
        and value.get('compiled_core_slots', 0) > 0 and len(value.get('triton_modules', [])) == blocks
        and all(v['level'] == 2 and v['fused_value_path'] == '_TritonSO2ValuePath'
                for v in value['triton_modules']))


def dpa_status(value):
    if not value:return 'MISSING'
    if 'median_ms' in value:return 'PASS'
    status = value.get('status', 'INCOMPLETE')
    if status == 'OOM' and value.get('log'):
        log = Path(value['log']).read_text(errors='replace')
        if 'ref_value = reference(' in log:return 'REFERENCE_OOM'
        return 'EXECUTION_OOM'
    return status


def collect(root, result_set, scheduler_states):
    rows = []
    for name, spec in CASES.items():
        directory = root/result_set/name
        manifest = read(directory/'manifest.json')
        if not manifest:continue
        dpa_metadata = read(root/'initializations'/name/'dpa4/metadata.json') or {}
        for n in manifest['sizes']:
            base = directory/f'n{n}'
            full, lower = [read(base/f'dpa4_{b}.json') for b in ('full', 'lower')]
            valid_dpa = dpa_valid(full, spec['layers']) and dpa_valid(lower, spec['layers'])
            for value in (full, lower):
                if value and 'median_ms' in value and value.get('CUDA_VISIBLE_DEVICES') != manifest['CUDA_VISIBLE_DEVICES']:
                    raise ValueError('DPA allocation differs from its paired manifest')
            for scope in ('final', 'persistent'):
                chorus = read(base/f'chorus_{scope}.json')
                row = {'case': name, 'atoms': n, 'scope': scope, 'result_set': result_set,
                    'chorus_channels': spec['channels'], 'lmax': spec['lmax'], 'layers': spec['layers'],
                    'dpa_channels': spec['dpa_channels'], 'dpa_mmax': 1,
                    'chorus_status': chorus.get('status', 'MISSING') if chorus else 'MISSING',
                    'dpa_full_status': dpa_status(full), 'dpa_lower_status': dpa_status(lower),
                    'dpa_actual_compile_triton2_verified': valid_dpa,
                    'dpa_level2_value_path_eligible_width': spec['dpa_channels'] in (32, 64, 96, 128),
                    'slurm_job': manifest['slurm_job'], 'GPU_allocation': manifest['CUDA_VISIBLE_DEVICES'],
                    'dpa_parameters': dpa_metadata.get('parameters')}
                row['dpa_acceleration'] = ('compile+level2_value_path' if valid_dpa else
                    'compile+level2_requested_value_path_unavailable' if spec['dpa_channels'] == 48 else
                    'not_verified_at_this_size')
                if full and 'median_ms' in full:row.update(dpa_full_ms=full['median_ms'])
                if lower and 'median_ms' in lower:
                    row.update(dpa_lower_ms=lower['median_ms'], dpa_parameters=lower['parameters'],
                               dpa_reserved_gib=lower['peak_reserved']/2**30,
                               dpa_allocated_gib=lower['peak_allocated']/2**30)
                if chorus:
                    qualified = chorus.get('measured_original_qualified', False)
                    row.update(chorus_parameters=chorus.get('parameters'),
                        source=chorus.get('source_snapshot'), stage=chorus.get('stage'),
                        original_record_status=chorus.get('status'),
                        measured_original_qualified=qualified,
                        reference_execution=chorus.get('reference_execution'),
                        parity=chorus.get('parity'), capture_validation=chorus.get('capture_validation'))
                    attempts = [(path, read(path)) for path in
                        sorted(root.glob(f'qualifications*/{name}/n{n}/{scope}.json'))]
                    row['supplemental_attempts'] = [dict(path=str(path), status=value.get('status'),
                        stage=value.get('stage'), slurm_job=value.get('slurm_job'),
                        scheduler_state=scheduler_states.get(value.get('slurm_job')))
                        for path, value in attempts if value]
                    supplemental_path, supplemental = next(((path, value) for path, value in attempts
                        if value and value.get('status') == 'PASS'), (None, None))
                    if supplemental and supplemental.get('status') == 'PASS':
                        if (supplemental['spec'] != chorus['spec']
                            or supplemental['initialization_file'] != chorus['initialization_file']
                            or supplemental['atoms'] != n):
                            raise ValueError('Supplemental qualification identity differs from measured model')
                        qualified = True
                        row.update(measured_original_qualified=True,
                            supplemental_qualification=str(supplemental_path),
                            reference_execution=supplemental['reference'], parity=supplemental['parity'])
                        if chorus.get('status') == 'TIMED_WITH_SMALL_ORACLE_ONLY':
                            row['chorus_status'] = 'PASS_WITH_SUPPLEMENTAL_ORIGINAL'
                    is_oom = chorus.get('status') == 'OOM' or 'out of memory' in chorus.get('error', '').lower()
                    if is_oom:
                        row['oom_context'] = ('original_reference' if 'original' in chorus.get('stage', '')
                                              else 'graph_capture_workflow' if chorus.get('stage') == 'capture_and_live_coordinate_verification'
                                              else 'HP_execution_or_capture')
                        row['chorus_status'] = ('REFERENCE_OOM' if row['oom_context'] == 'original_reference'
                                                else 'CAPTURE_WORKFLOW_OOM' if row['oom_context'] == 'graph_capture_workflow'
                                                else 'EXECUTION_OOM')
                    if chorus.get('error'):row['error'] = chorus['error']
                    if chorus.get('status') in ('PASS', 'TIMED_WITH_SMALL_ORACLE_ONLY'):
                        if chorus['CUDA_VISIBLE_DEVICES'] != manifest['CUDA_VISIBLE_DEVICES']:
                            raise ValueError('CHORUS allocation differs from paired manifest')
                        row.update(chorus_ms=chorus['median_ms'],
                            chorus_reserved_gib=chorus['peak_reserved']/2**30,
                            chorus_capture_allocated_gib=chorus['capture_peak_allocated']/2**30)
                        if valid_dpa and qualified:
                            row.update(speedup_vs_full=full['median_ms']/chorus['median_ms'],
                                speedup_vs_lower=lower['median_ms']/chorus['median_ms'],
                                reserved_reduction_fraction=1-chorus['peak_reserved']/lower['peak_reserved'])
                rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--result-set', nargs='+', default=['results'])
    p.add_argument('--slurm-jobs', nargs='*', default=[])
    p.add_argument('--slurm-accounting-file', type=Path)
    args = p.parse_args()
    accounting = ''
    scheduler_states = {}
    if args.slurm_accounting_file:
        accounting = args.slurm_accounting_file.read_text()
    elif args.slurm_jobs:
        accounting = subprocess.check_output(['sacct', '-j', ','.join(args.slurm_jobs),
            '--format=JobID,State,ExitCode,Elapsed,MaxRSS', '--noheader', '--parsable2'], text=True)
    if accounting:
        for line in accounting.splitlines():
            fields = line.split('|')
            scheduler_states[fields[0]] = fields[1]
    rows = []
    for result_set in args.result_set:rows.extend(collect(args.root, result_set, scheduler_states))
    for row in rows:
        baseline = next((r for r in rows if r['case'] == 'base' and r['atoms'] == row['atoms']
                         and r['scope'] == row['scope']), None)
        if baseline and row.get('chorus_ms') and baseline.get('chorus_ms'):
            row['chorus_time_growth_vs_base'] = row['chorus_ms']/baseline['chorus_ms']
        if baseline and row.get('dpa_lower_ms') and baseline.get('dpa_lower_ms'):
            row['dpa_time_growth_vs_base'] = row['dpa_lower_ms']/baseline['dpa_lower_ms']
    report = {'protocol': 'Fixed untrained constructor initializations; not trained/accuracy-matched models.',
        'boundary': 'N1024 primary, selected N10000 stress; exact 50 neighbors; full E/F/virial; FP32/TF32 off, FP64 virial.',
        'capacity_note': 'CHORUS Cartesian channels and DPA SO(2) channels are different definitions. Parameters reported explicitly.',
        'execution': 'CHORUS captured fixed-graph E/F/virial; DPA official compiled full and prebuilt-edge lower.',
        'result_sets': args.result_set,
        'counts': dict(Counter(row['chorus_status'] for row in rows)),
        'slurm_accounting': accounting, 'rows': rows}
    report_name = ('report_complete' if len(args.result_set) > 1 else
                   'report' if args.result_set == ['results'] else 'report_'+args.result_set[0])
    destination = args.root/report_name
    destination.mkdir(parents=True, exist_ok=True)
    if accounting:(destination/'slurm_accounting.txt').write_text(accounting)
    (destination/'architecture_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# CHORUS / DPA4 架构扩展同卡比较', '',
        '固定初始化的执行扩展测试；不是训练精度、等容量或长轨迹验证。Final 与 Persistent 使用相同结构矩阵，correlation=3、rank16。', '',
        '神经网络/力 FP32、TF32 off，virial FP64。DPA compile=1、Triton level2，保留原 mmax=1。每行在同一作业、同一物理 A100 顺序测量。', '',
        'CHORUS 使用已验证的固定图 CUDA Graph；DPA 使用官方 compiled full/lower 入口。每点取 15 次同步 walltime 中位数；编译、参照生成和捕获准备均不计时。', '',
        'DPA C48 不在官方融合 value path 的支持宽度 {32,64,96,128} 内；其 compile 生效，但 level2 value path 回退，单独标为 T2-unavailable，不据此给融合基线加速比。', '',
        '原公式在测量尺寸验证通过后才给加速比；参照显存受限但可计时的点仅列原始耗时。OOM 与数值失败分别记录。', '']
    def number(value, digits=3):return '—' if value is None else f'{value:.{digits}f}'
    for n in sorted({r['atoms'] for r in rows}):
        lines += [f'## {n} 原子 / 50 邻居', '',
            '| 配置 | Scope | CHORUS / DPA C | 参数量 CHORUS / DPA | CHORUS ms | DPA lower ms | 加速比 | reserved GiB CHORUS / DPA | CHORUS 状态 | DPA 状态 |',
            '| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |']
        for r in rows:
            if r['atoms'] != n:continue
            params = f"{r.get('chorus_parameters', '—')} / {r.get('dpa_parameters', '—')}"
            memory = f"{number(r.get('chorus_reserved_gib'), 2)} / {number(r.get('dpa_reserved_gib'), 2)}"
            dpa_state = r['dpa_lower_status']
            if dpa_state == 'PASS':
                dpa_state += ' / T2' if r['dpa_actual_compile_triton2_verified'] else ' / T2-unavailable'
            lines.append(f"| L{r['lmax']} / {r['layers']} 层 | {r['scope']} | {r['chorus_channels']} / {r['dpa_channels']} | {params} | "
                f"{number(r.get('chorus_ms'))} | {number(r.get('dpa_lower_ms'))} | {number(r.get('speedup_vs_lower'), 2)} | {memory} | {r['chorus_status']} | {dpa_state} |")
        lines.append('')
    lines += ['## 时间增长（相对同批 L2 / C128 / 2 层基线）', '',
              '| 原子数 | 配置 | Scope | CHORUS 耗时倍数 | DPA 耗时倍数 |',
              '| ---: | --- | --- | ---: | ---: |']
    for r in rows:
        if r['case'] == 'base' or 'chorus_time_growth_vs_base' not in r:continue
        lines.append(f"| {r['atoms']} | {r['case']} | {r['scope']} | "
            f"{number(r.get('chorus_time_growth_vs_base'), 2)} | {number(r.get('dpa_time_growth_vs_base'), 2)} |")
    lines += ['', '所有参数量均为对应实际构造模型计数。两种通道定义不同，不按通道数字主张等容量；本轮不包含新增架构的训练或精度结论。', '']
    lines += ['## 验证与资源边界', '',
        'PASS_WITH_SUPPLEMENTAL_ORIGINAL 表示计时文件之外的完整尺寸原公式核对通过；原计时及此前失败记录不修改。', '',
        'CAPTURE_WORKFLOW_OOM 发生在 HP 普通全尺寸预热之后的 CUDA Graph 捕获/重放核验流程，不等同于普通前向无法运行。REFERENCE_OOM 表示未加速参照受限，不能据此断言加速入口无法运行。', '',
        '最大 Persistent 的独立参照采用逐原子 product 分块及原模块重算，并先与 128 原子未分块原式核对；分块仅用于参照，不进入计时。CPU 暂存的早期尝试及内存保护取消记录保留。', '',
        '汇总状态：'+', '.join(f'{key}: {value}' for key, value in report['counts'].items())+'。', '']
    if accounting:lines += ['Slurm 状态及退出码见 [slurm_accounting.txt](slurm_accounting.txt)。', '']
    (destination/'ARCHITECTURE_SCALING.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':main()
