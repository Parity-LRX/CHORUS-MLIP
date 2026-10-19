"""Remote-only staged export-operator diagnostic; not a performance result."""
import torch
from chorus.test.test_hp_export import _build
from chorus.synthetic import make_fixed_graph
from chorus.hp.export import _PreparedMessage,_prepare_phase,_prepare_density
from chorus.hp.runtime import _borrow_contraction
from chorus.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3,PhaseHermitianFullLResidual
from mace_ictc.interfaces.deployment_triton import prepare_contraction,prepare_harmonics
from types import SimpleNamespace

torch.manual_seed(20261017)
model=_build('persistent',8,1)
model.skip_input_validation=True;model.preserve_edge_order=True
graph=make_fixed_graph(num_nodes=17,avg_degree=5,dtype=torch.float32,device=torch.device('cuda'),seed=91)
ref=model(*graph).detach()
def show(name):
    got=model(*graph).detach()
    print(name,'reference',ref.sum().item(),'got',got.sum().item(),'maxdelta',(got-ref).abs().max().item(),flush=True)
for layer in model.interactions:
    object.__setattr__(layer,'_hp_message',_PreparedMessage(layer,torch.device('cuda')))
show('message')
for layer in model.interactions:
    if layer.phase_enabled:object.__setattr__(layer,'_hp_phase_coefficients',_prepare_phase(layer))
show('phase')
for module in model.modules():
    if isinstance(module,ICTDBridgeUSymmetricContractionSO3):
        borrowed=_borrow_contraction(module,training=False,checkpoint_contraction=True)
        prepare_contraction(borrowed,torch.device('cuda'))
        object.__setattr__(module,'_hp_contraction',borrowed)
show('contraction')
for module in model.modules():
    if isinstance(module,PhaseHermitianFullLResidual):object.__setattr__(module,'_hp_density',_prepare_density(module))
show('density')
holder=SimpleNamespace(ictd_fix_edge_lmax=model.ictd_fix_edge_lmax)
prepare_harmonics(holder,torch.device('cuda'))
object.__setattr__(model,'_hp_harmonics',lambda n,lmax:holder._deployment_harmonics(n))
show('harmonics')
