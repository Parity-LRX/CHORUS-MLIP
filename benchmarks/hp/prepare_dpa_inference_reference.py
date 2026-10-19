"""Construct a declared, level-2-supported width in the allocated batch job."""
import argparse
import copy
import json
import os
from pathlib import Path

from profile_inference_dpa_nograph import torch,get_model,normalize


p=argparse.ArgumentParser()
p.add_argument('--template',type=Path,required=True)
p.add_argument('--channels',type=int,choices=(32,64),required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Reference model construction requires sbatch')
config=json.loads(args.template.read_text())
config['model']['descriptor']['channels']=args.channels
config['model']['descriptor']['use_amp']=False
config['model'].update(use_compile=True,enable_tf32=False)
torch.manual_seed(20261017)
model=get_model(copy.deepcopy(normalize(copy.deepcopy(config))['model'])).cuda().eval().requires_grad_(False)
args.out.mkdir(parents=True,exist_ok=False)
(args.out/'input.json').write_text(json.dumps(config,indent=2)+'\n')
torch.save({'model':{'model.Default.'+key:value.cpu() if isinstance(value,torch.Tensor) else value
                    for key,value in model.state_dict().items()}},args.out/'initialization.pt')
(args.out/'metadata.json').write_text(json.dumps(dict(channels=args.channels,
    parameters=sum(p.numel() for p in model.parameters()),lmax=config['model']['descriptor']['lmax'],
    layers=config['model']['descriptor']['n_blocks'],selection='declared supported-width bracket 32 and 64',
    model_seed=20261017,initialization='constructor_only_not_trained'),indent=2)+'\n')
