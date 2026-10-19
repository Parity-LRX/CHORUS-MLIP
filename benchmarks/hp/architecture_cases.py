"""Explicit untrained architecture scaling matrix, separate from checkpoints."""
MODEL_SEED = 20261017
GEOMETRY_SEED = 20260728
CASES = {
    'base': dict(channels=128, lmax=2, layers=2, dpa_channels=32, large=True),
    'l3': dict(channels=128, lmax=3, layers=2, dpa_channels=32, large=True),
    'c192': dict(channels=192, lmax=2, layers=2, dpa_channels=48, large=False),
    'c256': dict(channels=256, lmax=2, layers=2, dpa_channels=64, large=False),
    'd3': dict(channels=128, lmax=2, layers=3, dpa_channels=32, large=False),
    'd4': dict(channels=128, lmax=2, layers=4, dpa_channels=32, large=False),
    'l3c192d3': dict(channels=192, lmax=3, layers=3, dpa_channels=48, large=True),
    'l3c256d4': dict(channels=256, lmax=3, layers=4, dpa_channels=64, large=True),
}


def chorus_spec(name, scope):
    case = CASES[name]
    return dict(family='chorus', scope=scope, channels=case['channels'], lmax=case['lmax'],
        interactions=case['layers'], correlation=3, density_rank=16,
        model_seed=MODEL_SEED, avg_num_neighbors=50., initialization='constructor_only_not_trained')


def dpa_spec(name):
    case = CASES[name]
    return dict(family='dpa4', channels=case['dpa_channels'], lmax=case['lmax'],
        mmax=1, n_blocks=case['layers'], model_seed=MODEL_SEED,
        initialization='constructor_only_not_trained')
