"""Same-card independent repeat plus explicit actual-Triton-2 width bracket."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run',required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Same-card followup requires sbatch')
    out=args.root/'followup'/args.run
    out.mkdir(parents=True,exist_ok=False)
    tools=args.source/'benchmarks/hp'
    records=[]
    def child(label,script,flags):
        command=[sys.executable,str(tools/script),*map(str,flags)]
        log=out/f'{label}.log'
        with log.open('x') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=args.root)
            entry=dict(label=label,command=command,pid=process.pid,log=str(log))
            records.append(entry)
            (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps(entry),flush=True)
            code=process.wait()
        entry['exit_code']=code
        (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
        if code:raise RuntimeError(f'{label} failed with exit code {code}: {log}')
    matched=Path('/public/home/sps-xia/rxlin/chorus_hp_training_nograph_20261018/parameter_matched_v1/l3')
    for scope in ('persistent','final'):
        package=args.root/'packages/prepared_csr_t211_v9/l3'/f'{scope}.pt2'
        child(f'{scope}_aoti','load_inference_aoti_nograph.py',
            ['--package',package,'--out',out/f'{scope}_aoti.json'])
        child(f'{scope}_native','profile_inference_native_nograph.py',
            ['--package',package,'--out',out/f'{scope}_native.json'])
        child(f'dpa_{scope}','profile_inference_dpa_nograph.py',
            ['--config',matched/scope/'input.json','--checkpoint',matched/scope/'initialization.pt',
             '--atoms',1024,'--out',out/f'dpa_{scope}.json'])
    template=Path('/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations/l3/dpa4/input.json')
    for channels in (32,64):
        directory=out/f'dpa_c{channels}'
        child(f'prepare_c{channels}','prepare_dpa_inference_reference.py',
            ['--template',template,'--channels',channels,'--out',directory])
        child(f'dpa_c{channels}','profile_inference_dpa_nograph.py',
            ['--config',directory/'input.json','--checkpoint',directory/'initialization.pt',
             '--atoms',1024,'--out',out/f'dpa_c{channels}.json'])
    rows=[]
    for scope in ('persistent','final'):
        aoti=json.loads((out/f'{scope}_aoti.json').read_text())
        native=json.loads((out/f'{scope}_native.json').read_text())
        fast=next(r for r in aoti['results'] if r['atoms']==1024)
        eager=next(r for r in native['results'] if r['atoms']==1024)
        for label in (scope,'c32','c64'):
            dpa=json.loads((out/f'dpa_{label}.json').read_text())
            low=next(r for r in dpa['results'] if r['boundary']=='lower')
            if len({aoti['slurm_job'],native['slurm_job'],dpa['slurm_job']})!=1:
                raise ValueError('Same-card comparison crossed batch allocations')
            rows.append(dict(scope=scope,dpa_label=label,atoms=1024,neighbors=50,
                chorus_parameters=native['parameters'],dpa_parameters=dpa['parameters'],
                dpa_channels=dpa['descriptor']['channels'],triton2_value_active=dpa['level2_value_path_active'],
                aoti_ms=fast['median_ms'],native_ms=eager['median_ms'],dpa_ms=low['median_ms'],
                dpa_over_aoti=low['median_ms']/fast['median_ms'],dpa_over_native=low['median_ms']/eager['median_ms'],
                aoti_reserved=fast['peak_reserved'],native_reserved=eager['peak_reserved'],dpa_reserved=low['peak_reserved']))
    report=dict(status='PASS',slurm_job=os.environ['SLURM_JOB_ID'],rows=rows,
        primary='near-parameter-count widths selected before inference timing',
        auxiliary='supported-width 32/64 bracket with different parameter counts; not substituted for the primary pairing',
        precision='strict FP32/TF32 off; FP64 virial',graph_capture=False,graph_replay=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 同卡独立复测与实际 Triton 2 参考','',
        '1024 原子、50 邻居，完整 E/F/virial；双方禁止 CUDA Graph。','',
        '| CHORUS | DPA C | DPA params | 实际 T2 | AOTI ms | 原生 ms | DPA ms | DPA/AOTI |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["scope"]} | {row["dpa_channels"]} | {row["dpa_parameters"]} | '
            f'{row["triton2_value_active"]} | {row["aoti_ms"]:.3f} | {row["native_ms"]:.3f} | '
            f'{row["dpa_ms"]:.3f} | {row["dpa_over_aoti"]:.3f}× |')
    lines+=['','主配对为近参数量；C32/C64 是实际支持 T2 value 的上下宽度参考，其参数量不同。',
        '完整原式核验、逐次计时、参数量、显存和命令保存在相邻 JSON/日志中。']
    (out/'SUPPORTED_T2_REPEAT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
