"""Remote-only environment and existing full-force OEQ qualification."""
import importlib.util
import json
import os
from pathlib import Path

import torch
from no_cuda_graph import forbid_cuda_graph


def main(policy):
    import openequivariance as oeq
    import openequivariance._torch.extlib as ext
    import nequip
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('A100 numerical qualification requires sbatch')
    if not ext.USE_PRECOMPILED_EXTENSION or not ext.BUILT_EXTENSION:
        raise RuntimeError('The existing precompiled OEQ extension must load')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    source=Path(os.environ['SOURCE'])
    path=source/'nequip_root/scripts/smoke_openequivariance_parity.py'
    spec=importlib.util.spec_from_file_location('original_oeq_smoke',path)
    smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)
    rows=[smoke.compare_mode(False,'final'),smoke.compare_mode(True,'final'),
          smoke.compare_mode(True,'persistent')]
    result=dict(status='PASS',torch=torch.__version__,oeq=oeq.__version__,
        nequip_source=nequip.__file__,oeq_source=oeq.__file__,precompiled_extension=True,
        graph_execution_policy=policy,slurm_job=os.environ['SLURM_JOB_ID'],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),results=rows)
    out=Path(os.environ['OUT']);out.mkdir(parents=True,exist_ok=True)
    (out/'environment_qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with forbid_cuda_graph(purpose='NequIP inference/training') as policy:main(policy)
