"""CPU-only report protocol tests; run in the designated remote sbatch job."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _records(root, dpa_status='PASS', bad_channels=False):
    folder = root/'results/matched/l3'
    match = dict(lmax=3, layers=2, chorus_channels=128, chorus_parameters=1000,
        dpa_channels=48, dpa_parameters=1008, signed_relative_mismatch=.008)
    manifest = dict(case='l3', sizes=[512], scopes=['persistent'], repeats=50,
        parameter_matching={'persistent': match}, native_baseline_variant='cache16jvp',
        backward_targets='all_trainable_parameters', slurm_job='fixture', CUDA_VISIBLE_DEVICES='fixture')
    _write(folder/'manifest.json', manifest)
    _write(folder/'case_status.json', dict(status='DONE'))
    _write(folder/'qualification128/persistent.json', dict(status='PASS'))
    native = dict(status='DIAGNOSTIC_NO_LARGE_REFERENCE', checkpoint='fixed-initialization',
        architecture=dict(channels=128, lmax=3, interactions=2), trainable_parameters=1000,
        arithmetic='strict FP32', backward_targets='all_trainable_parameters',
        graph_execution_policy=dict(capture=False, replay=False, inductor_cudagraphs=False),
        median_ms=10., measurements=[dict(samples=[{} for _ in range(50)],
            fresh_autograd_graph_per_step=True, post_timing_parameters_and_gradients_finite=True,
            peak_allocated=80, peak_reserved=100, atoms_per_second_sequence=51200.)])
    baseline = copy.deepcopy(native)
    baseline['median_ms'] = 12.
    dpa = dict(status=dpa_status)
    if dpa_status == 'PASS':
        dpa.update(median_ms=20., samples=[{} for _ in range(50)],
            graph_execution_policy=native['graph_execution_policy'], peak_allocated=180, peak_reserved=200,
            trainable_parameters=1008, backward_targets='all_trainable_parameters',
            descriptor=dict(channels=52 if bad_channels else 48, lmax=3, n_blocks=2))
    for index in (0, 1):
        for name, value in (('persistent', native), ('baseline_persistent', baseline), ('dpa_persistent', dpa)):
            _write(folder/f'n512/{name}_r{index}.json', value)


def _run(root, extra=()):
    script = Path(__file__).with_name('report_training_large_matrix.py')
    return subprocess.run([sys.executable, str(script), '--root', str(root), '--result-set', 'matched',
        '--cases', 'l3', '--target-speedup', '1.5', '--out', str(root/'report'), *extra], capture_output=True, text=True)


def test_report_separates_native_schedule_gain_from_dpa_match(tmp_path):
    _records(tmp_path)
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path/'report/large_training_summary.json').read_text())['rows'][0]
    assert row['speedup_over_dpa'] == pytest.approx(2.)
    assert row['native_speedup_over_previous_schedule'] == pytest.approx(1.2)
    assert row['throughput_and_memory_target_met']


def test_report_never_converts_dpa_oom_to_speedup(tmp_path):
    _records(tmp_path, dpa_status='OOM')
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path/'report/large_training_summary.json').read_text())['rows'][0]
    assert 'speedup_over_dpa' not in row
    assert 'throughput_and_memory_target_met' not in row


def test_report_rejects_changed_dpa_channel_identity(tmp_path):
    _records(tmp_path, bad_channels=True)
    result = _run(tmp_path)
    assert result.returncode != 0
    assert 'Timed DPA architecture changed' in result.stderr


def test_report_selects_matching_oracle_and_retains_prior_oom(tmp_path):
    _records(tmp_path)
    _write(tmp_path/'oracle_oom/l3/n512/persistent.json', dict(status='OOM'))
    qualified = json.loads((tmp_path/'results/matched/l3/n512/persistent_r0.json').read_text())
    qualified.update(status='PASS', qualification=[dict(
        parameter_gradient_parity={'weight': dict(relative_l2=1e-6)}) for _ in range(3)])
    _write(tmp_path/'oracle_pass/l3/n512/persistent.json', qualified)
    result = _run(tmp_path, ('--qualification-root', str(tmp_path/'oracle_oom'),
        '--qualification-root', str(tmp_path/'oracle_pass')))
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path/'report/large_training_summary.json').read_text())['rows'][0]
    assert row['same_size_original_formula_qualified']
    assert [attempt['status'] for attempt in row['qualification_attempts']] == ['MISSING', 'OOM', 'PASS']

    result = _run(tmp_path, ('--qualification-root', str(tmp_path/'oracle_pass'),
        '--qualification-root', str(tmp_path/'oracle_oom')))
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path/'report/large_training_summary.json').read_text())['rows'][0]
    assert row['same_size_original_formula_qualified']
    assert [attempt['status'] for attempt in row['qualification_attempts']] == ['MISSING', 'PASS', 'OOM']


def test_report_case_override_keeps_raw_result_provenance(tmp_path):
    _records(tmp_path)
    (tmp_path/'results/matched').rename(tmp_path/'results/followup')
    result = _run(tmp_path, ('--case-result-set', 'l3=followup'))
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path/'report/large_training_summary.json').read_text())['rows'][0]
    assert '/results/followup/l3/' in row['native']['inputs'][0]
