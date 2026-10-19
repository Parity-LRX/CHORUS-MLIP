"""Remote-only native-fusion summary; never selects an AOTI route."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    source=json.loads(args.source.read_text())
    if source['status']!='COMPLETE':raise ValueError('Inference matrix is incomplete')
    rows=[]
    for row in source['rows']:
        if row['torch_chorus']!=row['torch_dpa']:
            raise ValueError('Native/DPA runtime mismatch')
        rows.append(dict(case=row['case'],scope=row['scope'],atoms=row['atoms'],neighbors=row['neighbors'],
            chorus_parameters=row['chorus_parameters'],dpa_parameters=row['dpa_parameters'],
            parameter_mismatch=row['parameter_mismatch'],dpa_channels=row['dpa_channels'],
            native_ms=row['native_eager_ms'],dpa_lower_ms=row['dpa_lower_ms'],
            dpa_full_ms=row['dpa_full_ms'],speedup=row['dpa_lower_ms']/row['native_eager_ms'],
            native_reserved=row['native_reserved'],dpa_reserved=row['dpa_reserved'],
            dpa_level2_active=row['dpa_level2_active'],job=row['job']))
    result=dict(status='COMPLETE',execution='native fused only; no CUDA Graph; no AOTI selection',
        source_record=str(args.source),rows=rows,
        model_kind=source['model_kind'],matching=source['matching'],
        persistent_actual_t2_auxiliary=source.get('persistent_actual_t2_auxiliary'),
        supported_width_repeat=[{key:row[key] for key in ('scope','dpa_channels','chorus_parameters',
            'dpa_parameters','triton2_value_active','native_ms','dpa_ms','dpa_over_native',
            'native_reserved','dpa_reserved')} for row in source.get('supported_width_repeat',{}).get('rows',[])])
    lines=['# 原生融合推理：无 CUDA Graph','',
        'Persistent 与 Final 统一采用 ICTC-O3-HP 原生算子融合。所有数字均来自',
        '同卡顺序计时的完整 E/F/virial；不采用 AOTI，也不选择不同执行路线的最小值。',
        '网络与力 FP32，TF32/AMP off，virial FP64；两侧禁用 CUDA Graph。','',
        '模型使用固定结构初始化，比较执行开销；同 lmax=3、同层数，DPA 通道按参数量选择。','',
        '## 1024 原子、50 邻居','',
        '| CHORUS | scope | DPA C | 原生 ms | DPA lower ms | 加速 | reserved GB 原生/DPA | 实际 T2 value |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- | --- |']
    def table_row(row):
        label='C128 / 2层' if row['case']=='l3' else 'C256 / 4层'
        return (f'| {label} | {row["scope"]} | {row["dpa_channels"]} | {row["native_ms"]:.3f} | '
                f'{row["dpa_lower_ms"]:.3f} | {row["speedup"]:.3f}× | '
                f'{row["native_reserved"]/1e9:.3f}/{row["dpa_reserved"]/1e9:.3f} | '
                f'{"active" if row["dpa_level2_active"] else "unsupported width"} |')
    lines.extend(table_row(row) for row in rows if row['atoms']==1024)
    lines+=['','DPA4 均设置 compile 与 Triton level 2；实际内核支持情况单列，不能用开关代替。',
        'DPA lower 不含邻居整理；CHORUS 包含周期几何、力组装与 virial。DPA full 原始计时也保留。',
        'GB 为十进制峰值 reserved；原始记录另存 allocated 和逐次计时。','',
        '## 全部已测尺寸','',
        '| 配置 | scope | N | 原生 ms | DPA lower ms | 加速 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    lines.extend(f'| {row["case"]} | {row["scope"]} | {row["atoms"]} | {row["native_ms"]:.3f} | '
        f'{row["dpa_lower_ms"]:.3f} | {row["speedup"]:.3f}× |' for row in rows)
    aux=result['persistent_actual_t2_auxiliary']
    if aux:
        lines+=['','## 最大 Persistent：实际启用 T2 的补测','',
            f'同卡原生 {aux["native_ms"]:.3f} ms，DPA C{aux["dpa_channels"]} '
            f'{aux["dpa_ms"]:.3f} ms，加速 {aux["speedup"]:.3f}×。',
            f'CHORUS {aux["chorus_parameters"]:,} 参数，DPA {aux["dpa_parameters"]:,} 参数'
            f'（差 {aux["parameter_mismatch"]:+.2%}）；DPA 实际 T2 value={aux["dpa_level2_active"]}。',
            f'reserved 为 {aux["native_reserved"]/1e9:.3f}/{aux["dpa_reserved"]/1e9:.3f} GB。',
            '此项仍未达到 1.5×；近参数量主配对为 C108，不用较大或较小 DPA 替换其结论。']
    lines+=['','## C128 双 scope 独立复测与实际 T2 参考','',
        '| scope | DPA C | 实际 T2 | 原生 ms | DPA ms | 加速 |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    lines.extend(f'| {row["scope"]} | {row["dpa_channels"]} | {row["triton2_value_active"]} | '
        f'{row["native_ms"]:.3f} | {row["dpa_ms"]:.3f} | {row["dpa_over_native"]:.3f}× |'
        for row in result['supported_width_repeat'])
    lines+=['','C32/C64 为不同参数量的实际 T2 参考，不是近参数量主配对。',
        '原生方案在小体系尚有 Python/核调度开销，并非全部已测尺寸都达到吞吐目标。']
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'native_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'INFERENCE_NATIVE_NO_GRAPH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE',rows=len(rows),output=str(args.out)),indent=2),flush=True)


if __name__=='__main__':main()
