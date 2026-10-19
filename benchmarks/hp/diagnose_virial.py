"""Remote diagnostic separating model adjoints and extensive reduction error."""
import json
import torch
from profile_scaling import geometry
from chorus.hp import configure_model, energy_forces_virial
from chorus.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
loaded = LAMMPS_MLIAP_MFF.from_checkpoint(
    "/public/home/sps-xia/rxlin/mace_ictc_nequip_matched_checkpoints/chorus_final_r16.pth",
    element_types=["H", "C", "N", "O"], device="cuda", ictd_fix_product_backend="ictd-bridge-u",
    ictd_tp_backend="pytorch", reuse_checkpoint_u=True)
model = loaded.wrapper.model.eval().requires_grad_(False)
graph = geometry(4096, model.max_radius, 20260728)
pos, z, batch, src, dst, shift, cell = graph
vec = pos[dst]-pos[src]+torch.einsum("ei,eij->ej", shift, cell[batch[src]])


def strain_reference(edgewise, dtype):
    strain = torch.zeros_like(cell, dtype=dtype, requires_grad=True)
    deformation = torch.eye(3, device="cuda", dtype=dtype)[None]+strain
    if edgewise:
        strained_vec = torch.einsum("ei,eij->ej", vec.to(dtype), deformation[batch[src]]).float()
        atom = model(*graph, precomputed_edge_vec=strained_vec)
    else:
        atom = model(torch.einsum("ni,nij->nj", pos.to(dtype), deformation[batch]),
                     z, batch, src, dst, shift, cell.to(dtype)@deformation)
    return -torch.autograd.grad(atom.sum(), strain)[0].detach()


def edge_assembly(dtype):
    leaf = vec.detach().requires_grad_(True)
    atom = model(*graph, precomputed_edge_vec=leaf)
    gradient = torch.autograd.grad(atom.sum(), leaf)[0]
    return -(leaf.to(dtype).T@gradient.to(dtype)).detach()[None]


values = {}
for policy in ("reference", "generated"):
    configure_model(model, policy=policy, checkpoint_contraction=True, topology_backend="device")
    values[policy] = {
        "coordinate_strain_f32": strain_reference(False, torch.float32).cpu().tolist(),
        "coordinate_strain_f64": strain_reference(False, torch.float64).cpu().tolist(),
        "edge_strain_f64": strain_reference(True, torch.float64).cpu().tolist(),
        "edge_sum_f32": edge_assembly(torch.float32).cpu().tolist(),
        "edge_sum_f64": edge_assembly(torch.float64).cpu().tolist(),
        "public_virial_f32": energy_forces_virial(model, *graph)["virial"].cpu().tolist()}
path = "/public/home/sps-xia/rxlin/chorus_hp_throughput_20261017/results/virial_diagnostic.json"
with open(path, "w") as handle:
    json.dump(values, handle, indent=2)
print(json.dumps(values, indent=2), flush=True)
