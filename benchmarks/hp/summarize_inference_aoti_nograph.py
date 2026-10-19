"""Remote-only reduction of same-card graph-free inference records."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def no_graph(record):
    policy=record['graph_execution_policy']
    if any(policy[key] for key in ('capture','replay','inductor_cudagraphs')):
        raise ValueError('CUDA Graph result is not admissible')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--case-run',nargs='*',default=[])
    p.add_argument('--scope-run',nargs='*',default=[])
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    case_runs=dict(item.split('=',1) for item in args.case_run)
    scope_runs=dict(item.split('=',1) for item in args.scope_run)
    rows=[];attempts=[]
    for case in ('l3','l3c192d3','l3c256d4'):
        for scope in ('persistent','final'):
            selected=scope_runs.get(f'{case}/{scope}',case_runs.get(case,args.run))
            package=args.root/'packages'/selected/case/f'{scope}.pt2'
            export=read(package.with_suffix('.export.json'))
            loaded=read(Path(str(package)+'.standalone.json'))
            native=read(Path(str(package)+'.native.json'))
            dpa=read(Path(str(package)+'.dpa.json'))
            if export is None:continue
            attempt=dict(case=case,scope=scope,export=export['status'],
                package=str(package),stage=export.get('stage'),error=export.get('error'),
                standalone=None if loaded is None else loaded['status'],
                native=None if native is None else native['status'],dpa=None if dpa is None else dpa['status'])
            attempts.append(attempt)
            if any(item is None or item['status']!='PASS' for item in (export,loaded,native,dpa)):
                continue
            for item in (export,loaded,native,dpa):no_graph(item)
            if not loaded['standalone_without_model_imports']:
                raise ValueError('Package loader depends on model imports')
            if len({item['slurm_job'] for item in (export,loaded,native,dpa)})!=1 or len({
                    item['CUDA_VISIBLE_DEVICES'] for item in (export,loaded,native,dpa)})!=1:
                raise ValueError('Performance records are not from the same GPU allocation')
            for fast in loaded['results']:
                n,e=fast['atoms'],fast['edges']
                if e!=n*50:continue
                eager=next(item for item in native['results'] if item['atoms']==n and item['edges']==e)
                low=next(item for item in dpa['results'] if item['atoms']==n and item['boundary']=='lower')
                full=next(item for item in dpa['results'] if item['atoms']==n and item['boundary']=='full')
                best=fast if fast['median_ms']<=eager['median_ms'] else eager
                best_name='AOTI' if best is fast else 'native_eager'
                rows.append(dict(case=case,scope=scope,atoms=n,neighbors=50,run=selected,
                    chorus_parameters=export['parameters'],dpa_parameters=dpa['parameters'],
                    parameter_mismatch=(dpa['parameters']-export['parameters'])/export['parameters'],
                    dpa_channels=dpa['descriptor']['channels'],
                    aoti_ms=fast['median_ms'],native_eager_ms=eager['median_ms'],
                    dpa_lower_ms=low['median_ms'],dpa_full_ms=full['median_ms'],
                    aoti_over_native=eager['median_ms']/fast['median_ms'],
                    aoti_over_dpa_lower=low['median_ms']/fast['median_ms'],
                    aoti_over_dpa_full=full['median_ms']/fast['median_ms'],
                    best_execution=best_name,best_ms=best['median_ms'],
                    best_over_dpa=low['median_ms']/best['median_ms'],
                    best_reserved=best['peak_reserved'],best_allocated=best['peak_allocated'],
                    aoti_reserved=fast['peak_reserved'],native_reserved=eager['peak_reserved'],
                    dpa_reserved=low['peak_reserved'],aoti_allocated=fast['peak_allocated'],
                    dpa_allocated=low['peak_allocated'],
                    dpa_level2_active=dpa['level2_value_path_active'],
                    torch_chorus=loaded['torch'],torch_dpa=dpa['torch'],
                    job=export['slurm_job'],package=str(package)))
    args.out.mkdir(parents=True,exist_ok=True)
    required={(case,scope) for case in ('l3','l3c256d4') for scope in ('persistent','final')}
    complete=required.issubset({(row['case'],row['scope']) for row in rows if row['atoms']==1024})
    report=dict(status='COMPLETE' if complete else 'PARTIAL',run=args.run,case_runs=case_runs,
        scope_runs=scope_runs,rows=rows,attempts=attempts,
        policy='No CUDA Graph on either side; true standalone AOTI; strict FP32 network/forces and FP64 virial',
        matching='same lmax and interaction depth, DPA widths selected by parameter count only',
        model_kind='fixed_constructor_initialization_not_trained_accuracy_pair')
    (args.out/'inference_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 无 CUDA Graph 的 AOTI 推理对照','',
        '同卡顺序比较完整 E/F/virial。CHORUS 为实际编译、独立进程加载的动态 N/E `.pt2`；',
        'DPA4 开启 compile 与 Triton 2，实际 value path 单独记录。两侧禁止 capture/replay。',
        '网络与力保持 FP32/TF32 off，virial 为 FP64；邻居构建与编译不进入稳态计时。','',
        '本表使用固定结构初始化，评估执行开销，不是已训练的等精度模型配对。',
        '原子数/边数变化均与对应原始公式核验；不是固定输入缓存。','',
        '| 配置 | scope | N | DPA C | AOTI ms | 原生 eager ms | DPA lower ms | 最快路线/DPA 加速 | reserved GB 最快/DPA | 实际 T2 value |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for row in rows:
        lines.append(f'| {row["case"]} | {row["scope"]} | {row["atoms"]} | {row["dpa_channels"]} | '
            f'{row["aoti_ms"]:.3f} | {row["native_eager_ms"]:.3f} | {row["dpa_lower_ms"]:.3f} | '
            f'{row["best_execution"]}: {row["best_over_dpa"]:.3f}× | {row["best_reserved"]/1e9:.3f}/{row["dpa_reserved"]/1e9:.3f} | '
            f'{"active" if row["dpa_level2_active"] else "unsupported width"} |')
    lines+=['','各行的原始计时、参数差、完整输出资格、运行命令和作业身份保存在相邻 JSON 中。',
        '显存 GB=10^9 bytes；显存与吞吐分别判断，不用一个指标代替另一个。','',
        '## 本轮尝试','']
    lines.extend(f'- {item["case"]}/{item["scope"]}: export={item["export"]}, '
        f'load={item["standalone"]}, native={item["native"]}, DPA={item["dpa"]}; stage={item["stage"]}'
        for item in attempts)
    supported=args.root/'followup/supported_v11'
    if (supported/'summary.json').is_file():
        report['supported_width_repeat']=read(supported/'summary.json')
        (args.out/'inference_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        lines+=['','## 独立同卡复测与实际 T2 参考','',
                (supported/'SUPPORTED_T2_REPEAT.md').read_text()]
    persistent_t2=args.root/'followup/persistent_actual_t2_v14'
    native=read(persistent_t2/'persistent_native.json')
    dpa=read(persistent_t2/'dpa_c96.json')
    if native is not None and dpa is not None and native['status']==dpa['status']=='PASS':
        no_graph(native);no_graph(dpa)
        if native['slurm_job']!=dpa['slurm_job'] or native['CUDA_VISIBLE_DEVICES']!=dpa['CUDA_VISIBLE_DEVICES']:
            raise ValueError('Persistent actual-T2 auxiliary crossed GPU allocations')
        eager=next(r for r in native['results'] if r['atoms']==1024 and r['edges']==51200)
        low=next(r for r in dpa['results'] if r['atoms']==1024 and r['boundary']=='lower')
        auxiliary=dict(chorus_parameters=native['parameters'],dpa_parameters=dpa['parameters'],
            parameter_mismatch=(dpa['parameters']-native['parameters'])/native['parameters'],
            native_ms=eager['median_ms'],dpa_ms=low['median_ms'],
            speedup=low['median_ms']/eager['median_ms'],
            native_reserved=eager['peak_reserved'],dpa_reserved=low['peak_reserved'],
            dpa_channels=dpa['descriptor']['channels'],
            dpa_level2_active=dpa['level2_value_path_active'],job=native['slurm_job'],
            role='actual-T2 auxiliary, smaller DPA parameter count; not a matched-parameter replacement')
        report['persistent_actual_t2_auxiliary']=auxiliary
        lines+=['','## 最大 Persistent 的实际 T2 补测','',
            '同卡、1024 原子、50 邻居、完整 E/F/virial、无 CUDA Graph。',
            f'CHORUS C256 四层 Persistent：{native["parameters"]:,} 参数，'
            f'原生 {eager["median_ms"]:.3f} ms，reserved {eager["peak_reserved"]/1e9:.3f} GB。',
            f'DPA C{dpa["descriptor"]["channels"]}：{dpa["parameters"]:,} 参数，'
            f'实际 T2 value={dpa["level2_value_path_active"]}，{low["median_ms"]:.3f} ms，'
            f'reserved {low["peak_reserved"]/1e9:.3f} GB。',
            f'加速 {auxiliary["speedup"]:.3f}×；DPA 参数量差 {auxiliary["parameter_mismatch"]:+.2%}。',
            '此项保留为较小 DPA 模型的实际 T2 参考，主配对仍是近参数量的 C108。']
    (args.out/'inference_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'INFERENCE_AOTI_NO_GRAPH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
