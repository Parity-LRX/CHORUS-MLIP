"""Same-degree/depth DPA4 channel selection by measured parameter count only.

No throughput or memory measurements influence this selection. The original
CHORUS initializations stay fixed. Only DPA channels differ from its template;
rank, focus, grid branches, mmax and all remaining architecture options stay put.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path

from architecture_cases import CASES, MODEL_SEED
from profile_training_dpa_nograph import get_model, normalize, torch
from no_cuda_graph import forbid_cuda_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['l3', 'l3c192d3', 'l3c256d4'])
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'GPU-A100'
            or 'XHPC-A100' not in os.environ.get('SLURM_JOB_NODELIST', '')):
        raise RuntimeError('Parameter construction/counting requires the designated A100 sbatch job')
    args.out.mkdir(parents=True, exist_ok=False)
    architecture = Path('/public/home/sps-xia/rxlin/chorus_hp_architecture_20261017/initializations')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    report = dict(rule='minimize absolute relative parameter-count mismatch on channels 32..160 step 4',
        selection_metric='parameter_count_only_not_speed', maximum_desired_relative_mismatch=.05,
        slurm_job=os.environ['SLURM_JOB_ID'], CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        channels=list(range(32, 161, 4)), model_seed=MODEL_SEED, cases={})

    def construct(config):
        torch.manual_seed(MODEL_SEED)
        model = get_model(copy.deepcopy(normalize(copy.deepcopy(config))['model'])).cuda().eval()
        if {value.dtype for value in model.parameters()} != {torch.float32}:
            raise ValueError('Parameter scan changed FP32 architecture precision')
        return model

    for case in args.cases:
        fixed = CASES[case]
        template = json.loads((architecture/case/'dpa4/input.json').read_text())
        if (template['model']['descriptor']['lmax'] != fixed['lmax'] or
                template['model']['descriptor']['n_blocks'] != fixed['layers']):
            raise ValueError('Degree/depth template mismatch')
        targets = {}
        for scope in ('persistent', 'final'):
            record = json.loads((args.root/'results/highl_latest_pair_v4'/case/'n1024'/f'{scope}_r0.json').read_text())
            if record['architecture']['channels'] != fixed['channels']:
                raise ValueError('Fixed CHORUS channel identity changed')
            targets[scope] = record['trainable_parameters']
        table = []
        for channels in report['channels']:
            config = copy.deepcopy(template)
            config['model']['descriptor']['channels'] = channels
            model = construct(config)
            parameters = sum(value.numel() for value in model.parameters())
            trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
            if trainable != parameters:
                raise ValueError('DPA matching must count the same all-trainable parameter set as timing')
            row = dict(channels=channels, parameters=parameters)
            table.append(row)
            print(json.dumps(dict(case=case, lmax=fixed['lmax'], layers=fixed['layers'], **row)), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        selected = {}
        for scope, target in targets.items():
            match = min(table, key=lambda item: (abs(item['parameters']-target), item['channels']))
            directory = args.out/case/scope
            directory.mkdir(parents=True, exist_ok=False)
            config = copy.deepcopy(template)
            config['_comment'] = 'Parameter-matched fixed constructor initialization; not an accuracy-matched trained model.'
            config['model']['descriptor']['channels'] = match['channels']
            model = construct(config)
            if sum(value.numel() for value in model.parameters()) != match['parameters']:
                raise ValueError('Selected architecture count changed on reconstruction')
            state = {'model.Default.'+key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in model.state_dict().items()}
            torch.save({'model': state}, directory/'initialization.pt')
            (directory/'input.json').write_text(json.dumps(config, indent=2)+'\n')
            delta = (match['parameters']-target)/target
            metadata = dict(case=case, scope=scope, chorus_channels=fixed['channels'],
                chorus_parameters=target, dpa_channels=match['channels'], dpa_parameters=match['parameters'],
                signed_relative_mismatch=delta, within_five_percent=abs(delta)<=.05,
                lmax=fixed['lmax'], layers=fixed['layers'],
                unchanged_dpa_options={key: value for key, value in template['model']['descriptor'].items() if key != 'channels'},
                selection='closest_parameter_count_on_predeclared_multiple_of_four_channel_grid',
                model_seed=MODEL_SEED, initialization='constructor_only_not_trained', directory=str(directory))
            (directory/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
            selected[scope] = metadata
            print(json.dumps(dict(selected=metadata)), flush=True)
            del model, state
            gc.collect()
            torch.cuda.empty_cache()
        report['cases'][case] = dict(parameter_scan=table, selected=selected)
        (args.out/'parameter_matches.json').write_text(json.dumps(report, indent=2)+'\n')
    report['status'] = 'DONE'
    (args.out/'parameter_matches.json').write_text(json.dumps(report, indent=2)+'\n')
    print('DONE', flush=True)


if __name__ == '__main__':
    with forbid_cuda_graph():
        main()
