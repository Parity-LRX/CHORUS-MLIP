import os
from pathlib import Path
import pytest
from no_cuda_graph import forbid_cuda_graph


if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('A100 tests require sbatch')
with forbid_cuda_graph(purpose='NequIP unit qualification'):
    source=Path(os.environ['SOURCE'])
    out=Path(os.environ['OUT'])
    raise SystemExit(pytest.main([str(source/'nequip_root/tests/unit/nn/test_chorus.py'),'-q','-x',
        '--junitxml='+str(out/f'unit_{os.environ["SLURM_JOB_ID"]}.xml')]))
