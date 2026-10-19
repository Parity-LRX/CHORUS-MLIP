"""Construct/save explicit DPA4 initialization in an allocated remote job."""
import argparse
import copy
import gc
import json
from pathlib import Path
from architecture_cases import CASES, MODEL_SEED, dpa_spec
from profile_dpa_boundary import get_model, normalize, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(MODEL_SEED)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = dpa_spec(args.case)
    if (args.out_dir/'metadata.json').exists() and (args.out_dir/'initialization.pt').exists():
        metadata = json.loads((args.out_dir/'metadata.json').read_text())
        if metadata['spec'] != spec:raise ValueError('Existing DPA initialization has a different architecture specification')
        print(json.dumps({'reused_initialization': True, **metadata}, indent=2), flush=True)
        return
    config = json.loads(args.template.read_text())
    config['_comment'] = 'Fixed constructor initialization for architecture throughput only; not a trained checkpoint.'
    descriptor = config['model']['descriptor']
    descriptor.update(channels=spec['channels'], lmax=spec['lmax'], mmax=spec['mmax'],
                      n_blocks=spec['n_blocks'], seed=MODEL_SEED, use_amp=False, precision='float32')
    descriptor.pop('l_schedule', None)
    descriptor.pop('m_schedule', None)
    config['model']['fitting_net']['seed'] = MODEL_SEED
    config['model'].update(use_compile=True, enable_tf32=False)
    normalized = normalize(copy.deepcopy(config))
    model = get_model(copy.deepcopy(normalized['model'])).cuda().eval().requires_grad_(False)
    if {v.dtype for v in model.parameters()} != {torch.float32}:
        raise ValueError('DPA model parameters are not strictly FP32')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'input.json').write_text(json.dumps(config, indent=2)+'\n')
    state = {'model.Default.'+k: v.cpu() if isinstance(v, torch.Tensor) else v
             for k, v in model.state_dict().items()}
    torch.save({'model': state}, args.out_dir/'initialization.pt')
    metadata = {'case': args.case, 'spec': spec, 'parameters': sum(v.numel() for v in model.parameters()),
                'descriptor': descriptor, 'source_snapshot': str(Path(__file__).resolve().parents[2])}
    (args.out_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)
    del model, state
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':main()
