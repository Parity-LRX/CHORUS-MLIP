"""Remote-only reduction of the same-card NequIP OEQ execution records."""
import json
import os
from pathlib import Path


root=Path(os.environ['SOURCE']).parent
version=os.environ.get('BENCH_VERSION','v6')
sources=sorted((root/'results').glob(f'*_{version}/benchmark.json'))
if len(sources)!=int(os.environ.get('EXPECTED_FAMILIES','4')):
    raise RuntimeError('Required benchmark families are incomplete')
rows=[];attempts=[]
for path in sources:
    record=json.loads(path.read_text())
    policy=record['graph_execution_policy']
    if any(policy[k] for k in ('capture','replay','inductor_cudagraphs')):
        raise RuntimeError('Graph execution is not admissible')
    spec=record['config']
    attempts.append(dict(path=str(path),status=record['status'],job=record['slurm_job']))
    for optimized in record['results']:
        if optimized['mode']!='optimized' or optimized['status']!='PASS':continue
        n,task=optimized['atoms'],optimized['task']
        legacy=next(r for r in record['results'] if r['mode']=='legacy' and r['atoms']==n and r['task']==task)
        old=next(r for r in record['results'] if r['mode']=='tp_only' and r['atoms']==n and r['task']==task)
        if legacy['status']!='PASS' or old['status']!='PASS':continue
        rows.append(dict(scope=spec['chorus_scope'],channels=spec['chemical_embedding_irreps_out'],
            lmax=len(spec['irreps_edge_sh'].split('+'))-1,layers=spec['num_layers'],rank=spec['chorus_rank'],
            parameters=record['parameters'],atoms=n,neighbors=50,task=task,
            legacy_ms=legacy['median_ms'],optimized_ms=optimized['median_ms'],tp_only_ms=old['median_ms'],
            speedup_over_legacy=legacy['median_ms']/optimized['median_ms'],
            speedup_over_tp_only=old['median_ms']/optimized['median_ms'],
            legacy_reserved=legacy['peak_reserved'],optimized_reserved=optimized['peak_reserved'],
            legacy_allocated=legacy['peak_allocated'],optimized_allocated=optimized['peak_allocated'],
            original_outputs=optimized['original_outputs'],
            parameter_gradient_max_relative_l2=optimized.get('parameter_gradient_max_relative_l2'),
            slurm_job=record['slurm_job'],cuda_visible_devices=record['cuda_visible_devices']))
out=root/f'report_{version}';out.mkdir(exist_ok=True)
summary=dict(status='COMPLETE' if all(a['status']=='COMPLETE' for a in attempts) else 'PARTIAL',
    rows=rows,attempts=attempts,
    baseline='existing remote optimized OEQ density, not the slower TP-only public overlay',
    protocol='same card and same fixed state; full E/F/virial; training includes mixed backward and AdamW; no CUDA Graph/AOTI')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# NequIP–CHORUS：原生 OpenEquivariance 优化','',
    '同卡顺序测量，网络 FP32、TF32/AMP off，不使用 CUDA Graph 或 AOTI。',
    '推理包含完整 E/F/virial；训练包含 E/F/virial 损失、力混合反传和 AdamW。',
    '主基线是远端已有的 OpenEquivariance Hermitian 优化版；TP-only 结果另列。',
    '各执行方案加载同一份固定初始化，参数、rank、CG 路径和相位方程不变。','',
    '通道配置：'+', '.join(sorted({row['channels'] for row in rows}))+'；rank=16。','',
    '## 1024 原子、50 邻居','',
    '| scope | L / 层数 | task | 原优化 ms | 新优化 ms | 加速 | reserved GB 原/新 |',
    '| --- | --- | --- | ---: | ---: | ---: | --- |']
def line(r):
    return (f'| {r["scope"]} | {r["lmax"]} / {r["layers"]} | {r["task"]} | '
        f'{r["legacy_ms"]:.3f} | {r["optimized_ms"]:.3f} | {r["speedup_over_legacy"]:.3f}× | '
        f'{r["legacy_reserved"]/1e9:.3f}/{r["optimized_reserved"]/1e9:.3f} |')
lines.extend(line(r) for r in rows if r['atoms']==1024)
lines+=['','## 全部尺寸','',
    '| scope | L / 层数 | N | task | TP-only ms | 原优化 ms | 新优化 ms | 加速 | reserved GB 原/新 |',
    '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
for r in rows:
    lines.append(f'| {r["scope"]} | {r["lmax"]} / {r["layers"]} | {r["atoms"]} | {r["task"]} | '
        f'{r["tp_only_ms"]:.3f} | {r["legacy_ms"]:.3f} | {r["optimized_ms"]:.3f} | '
        f'{r["speedup_over_legacy"]:.3f}× | {r["legacy_reserved"]/1e9:.3f}/{r["optimized_reserved"]/1e9:.3f} |')
lines+=['','GB 使用十进制峰值 reserved；allocated 与逐次计时保存在原始 JSON 中。',
    '邻居构建、原式核验和 OEQ 核冷启动不计入稳态。输入图复制、完整输出和优化器更新计入相应 step。',
    '此表比较固定初始化结构的执行开销，不是重新训练后的等精度或物理预测比较。','',
    '## 实现','',
    '- Hermitian 收缩将独立 rank 放到批次轴，使用单重度 OEQ TP 和标量 +/-1 常量权重。',
    '- 相位网络首层按边/源节点/目标节点分块，节点投影在 gather 前每个节点只计算一次。',
    '- 元素输出映射先做共享 bank GEMM，再按节点属性混合，避免逐节点复制整个权重 bank；保留软属性梯度。',
    '- Final 的角动量非零中性层使用 OEQ 融合 TP convolution；标量首层保留普通 TP，避免额外 workspace。',
    '- 相位诊断或干预 hook 存在时自动保留原相位网络调用，以维护原来的观测和干预语义。',
    '- 保留原 Parameter 对象、checkpoint 键、非零相位初始化与所有 Hermitian 实/虚路径。']
(out/'NEQUIP_OEQ_NATIVE.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:10]+[line(r) for r in rows if r['atoms']==1024]),flush=True)
