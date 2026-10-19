"""Explicit Final execution qualifier; original maps plus stable FP32 forces.

Run the unchanged full E/F/virial profiler and registered oracle. The wrapper
does not change a checkpoint, reference, tolerance, geometry or capture protocol.
"""
import os
import runpy
import sys
from pathlib import Path

source = Path(os.environ['SOURCE'])
selected = tuple(os.environ.get('NATIVE_MAPS', 'channel,activation,species').split(','))
attributes = dict(channel='_hp_channel_linear', activation='_hp_activation', species='_hp_species_linear')
count_names = dict(channel='channel_maps', activation='activations', species='species_maps')
if not selected or any(name not in attributes for name in selected):
    raise ValueError('NATIVE_MAPS must select channel, activation and/or species')
sys.path.insert(0, str(source/'benchmarks/hp'))
import chorus.hp
import experimental_execution

configure = chorus.hp.configure_model
def configure_reference_maps(model, **kwargs):
    plan = configure(model, **kwargs)
    if plan['selected'] != 'reference':
        if model.ictd_fix_phase_scope != 'final':
            raise ValueError('This explicit numerical route is qualified for Final only')
        for module in model.modules():
            for attribute in (attributes[name] for name in selected):
                if hasattr(module, attribute):object.__setattr__(module, attribute, None)
        plan['counts'].update({count_names[name]: 0 for name in selected})
        plan.update(execution_qualifier='original_maps_stable_fp32_force_reduction',
                    original_map_selection=selected,
                    execution_wrapper=str(Path(__file__).resolve()))
    return plan

apply = experimental_execution.apply_experiment
def stable_forces(model, name):
    return {**apply(model, name), 'force_reduction': 'csr'}

chorus.hp.configure_model = configure_reference_maps
experimental_execution.apply_experiment = stable_forces
runpy.run_path(str(source/'benchmarks/hp/profile_scaling.py'), run_name='__main__')
