"""Full E/F/virial, with a shared prebuilt topology and live geometry work.

Each invocation is an isolated process on the pair's allocated GPU. Training
includes the same E/F/virial loss, mixed backward, and AdamW on both engines.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from no_cuda_graph import forbid_cuda_graph


def load_graph(root, n, batch):
    record = torch.load(root/f"inputs/n{n}_b{batch}.pt", map_location="cpu", weights_only=False)
    return {k:v.cuda() for k,v in record["data"].items()}


def clone_graph(data):
    return {k:v.detach().clone() for k,v in data.items()}


def build(engine, directory, task, accelerated):
    torch.manual_seed(20261018)
    if engine == "nequip":
        from nequip.model import model_from_config
        spec = json.loads((directory/"nequip_config.json").read_text())
        spec["openequivariance_enabled"] = accelerated
        label=os.environ.get("NEQUIP_CHORUS_VARIANT","optimized")
        if label in ("no_chorus","no_chorus_materialized"):
            spec["chorus_enabled"]=False
        model = model_from_config(spec, initialize=True)
    else:
        from deepmd.pt.model.model import get_model
        from deepmd.utils.argcheck import normalize
        spec = json.loads((directory/"dpa_config.json").read_text())
        spec["model"].update(use_compile=accelerated, enable_tf32=False)
        os.environ.update(DP_COMPILE_INFER=str(int(accelerated)),
                          DP_TRITON_INFER="2" if accelerated else "0")
        model = get_model(copy.deepcopy(normalize(copy.deepcopy(spec))["model"]))
    state = torch.load(directory/f"{engine}_initialization.pt", map_location="cpu", weights_only=False)
    if engine=="nequip" and label in ("no_chorus","no_chorus_materialized"):
        keys=model.state_dict()
        if not all(k in state and state[k].shape==v.shape for k,v in keys.items()):
            raise AssertionError("No-CHORUS backbone does not match the unchanged shared state")
        state={k:state[k] for k in keys}
    model.load_state_dict(state, strict=True)
    if engine=="nequip" and label=="no_chorus_materialized":
        from nequip.nn._openequivariance import OpenEquivarianceTensorProduct
        for module in model.modules():
            if isinstance(module,OpenEquivarianceTensorProduct):
                module.use_fused_convolution=False
                module.convolution=None
    model = model.cuda().train(task == "train").requires_grad_(task == "train")
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise AssertionError("Network parameters must stay FP32")
    return model, state


def forward(engine, model, data):
    graph = clone_graph(data)
    if engine == "nequip":
        from nequip.data import AtomicDataDict as D
        value = model(graph)
        return value[D.TOTAL_ENERGY_KEY], value[D.FORCE_KEY], value[D.VIRIAL_KEY]
    # Compact DPA input uses neighbour -> centre, with the same periodic edge
    # vectors as NequIP. No neighbor search in either timed boundary. Both
    # engines clone the input and recompute displacements from live positions.
    centre, neighbour = graph["edge_index"]
    edge_index = torch.stack((neighbour, centre))
    edge_vec = graph["pos"][neighbour] - graph["pos"][centre]
    edge_vec = edge_vec + graph["edge_cell_shift"] @ graph["cell"].reshape(3,3)
    value = model.forward_lower(graph["pos"].unsqueeze(0), graph["atom_types"].reshape(1,-1),
        edge_index, edge_vec, edge_index, torch.ones(edge_vec.shape[0], device="cuda", dtype=torch.bool))
    return value["energy"], value["extended_force"].reshape(-1,3), value["virial"]


def loss(values, n):
    energy, force, virial = values
    return (energy/n).square().mean() + force.square().mean() + .01*(virial/n).square().mean()


def compare(actual, expected, n):
    records = {}
    for name, a, b in zip(("energy", "force", "virial"), actual, expected):
        a, b = a.reshape(-1), b.to(a.device).reshape(-1)
        torch.testing.assert_close(a, b, atol=5e-4 if name!="virial" else n*5e-5, rtol=2e-3)
        records[name] = dict(max_abs=float((a.double()-b.double()).abs().max()),
            relative_l2=float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-20)),
            accelerated_dtype=str(a.dtype), original_dtype=str(b.dtype))
    return records


def gradients(model):
    return {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}


def qualification(engine, fast, directory, task, data):
    n = data["pos"].shape[0]
    original, _ = build(engine, directory, task, False)
    expected = forward(engine, original, data)
    if task == "train":
        loss(expected,n).backward()
        expected_grad = gradients(original)
    expected = tuple(v.detach().cpu() for v in expected)
    del original
    gc.collect()
    torch.cuda.empty_cache()
    os.environ.update(DP_COMPILE_INFER="1", DP_TRITON_INFER="2")
    actual = forward(engine, fast, data)
    result = dict(original_outputs=compare(actual, expected, n), atoms=n)
    if task == "train":
        loss(actual,n).backward()
        current = gradients(fast)
        if current.keys() != expected_grad.keys():
            raise AssertionError("Parameter-gradient set differs from original formula")
        errors = []
        for name, grad in current.items():
            ref = expected_grad[name]
            torch.testing.assert_close(grad, ref, atol=2e-3, rtol=3e-3)
            errors.append(dict(name=name, max_abs=float((grad-ref).abs().max()),
                relative_l2=float((grad.double()-ref.double()).norm()/ref.double().norm().clamp_min(1e-20))))
        result.update(gradient_tensors=len(current), worst_parameter_gradients=sorted(
            errors, key=lambda r:r["relative_l2"], reverse=True)[:8])
    del actual, expected
    fast.zero_grad(set_to_none=True)
    if engine == "dpa":
        # This independent public API builds its own periodic neighbor list.
        # It must agree with the compact, shared-topology input above.
        graph = clone_graph(data)
        full = fast(graph["pos"].unsqueeze(0), graph["atom_types"].reshape(1,-1),
                    graph["cell"].reshape(1,9))
        compact = forward(engine, fast, data)
        result["compact_vs_native_neighbor_builder"] = compare(compact,
            (full["energy"], full["force"].reshape(-1,3), full["virial"]), n)
        del full, compact, graph
        if not fast.should_use_compile() or not fast.compiled_core_compute_cache:
            raise AssertionError("DPA compiled core was not actually created")
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args, policy):
    if (not os.environ.get("SLURM_JOB_ID") or
            os.environ.get("SLURM_JOB_PARTITION") != "GPU-A100" or
            "XHPC-A100" not in os.environ.get("SLURM_JOB_NODELIST", "")):
        raise RuntimeError("Only the designated A100 sbatch allocation may run this benchmark")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    directory = args.root/"pairs"/args.case/args.scope
    metadata = json.loads((directory/"metadata.json").read_text())
    model, initial = build(args.engine, directory, args.task, True)
    count = sum(p.numel() for p in model.parameters())
    label=os.environ.get("NEQUIP_CHORUS_VARIANT","optimized")
    if count != metadata[f"{args.engine}_parameters"] and not (
            args.engine=="nequip" and label in ("no_chorus","no_chorus_materialized")):
        raise AssertionError("Timed model parameter count changed from the declared pair")
    prop = torch.cuda.get_device_properties(0)
    record = dict(status="RUNNING", engine=args.engine, task=args.task, round=args.round,
        case=args.case, scope=args.scope, parameters=count, metadata=metadata,
        nequip_variant=label if args.engine=="nequip" else None,
        backbone_state="same-width shared checkpoint tensors; CHORUS-only tensors removed" if
            args.engine=="nequip" and label in ("no_chorus","no_chorus_materialized") else "unchanged full checkpoint",
        slurm_job=os.environ["SLURM_JOB_ID"], CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=prop.name, gpu_uuid=str(getattr(prop,"uuid","unavailable")), torch=torch.__version__,
        pid=os.getpid(), command=Path("/proc/self/cmdline").read_bytes().replace(b"\0",b" ").decode(),
        graph_execution_policy=policy, arithmetic="FP32 network/forces; no AMP/TF32; native reduction dtype retained",
        timed_boundary="prebuilt periodic topology; cloned live inputs + geometry + complete E/F/virial",
        training="same E/n squared + F squared + 0.01 (virial/n) squared, backward, AdamW(lr=1e-5,weight_decay=0)",
        changing_batches=3, results=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.out.write_text(json.dumps(record, indent=2)+"\n")
    save()
    try:
        qual_data = load_graph(args.root,128,0)
        record["qualification"] = qualification(args.engine,model,directory,args.task,qual_data)
        del qual_data
        save()
        for n in args.atoms:
            item = dict(atoms=n, edges=n*50, neighbors=50, status="RUNNING")
            record["results"].append(item)
            save()
            optimizer = None
            batches = None
            try:
                model.load_state_dict(initial,strict=True)
                model.zero_grad(set_to_none=True)
                batches = [load_graph(args.root,n,b) for b in range(3)]
                optimizer = (torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.)
                             if args.task=="train" else None)
                def step(index):
                    if optimizer is not None: optimizer.zero_grad(set_to_none=True)
                    values = forward(args.engine,model,batches[index%3])
                    if optimizer is not None:
                        loss(values,n).backward()
                        optimizer.step()
                    return values
                started = time.perf_counter()
                for index in range(args.warmup):
                    value = step(index)
                    if not all(bool(v.isfinite().all()) for v in value):
                        raise AssertionError("Nonfinite warmup output")
                    del value
                torch.cuda.synchronize()
                item["warmup_seconds"] = time.perf_counter()-started
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                samples=[]
                for index in range(args.repeats):
                    torch.cuda.synchronize()
                    started=time.perf_counter()
                    value=step(index)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter()-started)*1000)
                    if not all(bool(v.isfinite().all()) for v in value):
                        raise AssertionError("Nonfinite timed output")
                    del value
                item.update(status="PASS", median_ms=statistics.median(samples), samples_ms=samples,
                    peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
                if optimizer is not None:
                    if not all(bool(p.isfinite().all()) and
                               (p.grad is None or bool(p.grad.isfinite().all())) for p in model.parameters()):
                        raise AssertionError("Nonfinite updated parameter or gradient")
                    item["post_update_parameters_and_gradients_finite"] = True
                if n==1024 and args.round==0:
                    detail=os.environ.get("PROFILE_DETAIL","0")=="1"
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=detail,profile_memory=detail) as profiler:
                        value=step(0)
                        torch.cuda.synchronize()
                        del value
                    events=profiler.key_averages()
                    profile=args.out.with_suffix(".profile.txt")
                    profile.write_text(events.table(sort_by="self_cuda_time_total",row_limit=120))
                    if detail:
                        args.out.with_suffix(".shapes.txt").write_text(profiler.key_averages(
                            group_by_input_shape=True).table(sort_by="self_cuda_time_total",row_limit=90))
                        layers=[]
                        for name,m in model.named_modules():
                            if hasattr(m,"linear_1") and hasattr(m,"linear_2"):
                                layers.append(dict(name=name,in_irreps=str(m.linear_1.irreps_in),
                                    message_irreps=str(m.linear_2.irreps_in),out_irreps=str(m.linear_2.irreps_out),
                                    charged=getattr(m,"chorus",None) is not None))
                        item["layer_layouts"]=layers
                    item["level2_profiler_events"]=[dict(name=e.key,count=e.count) for e in events
                        if any(tag in e.key.lower() for tag in ("so2_rotate_mix","so2_mixing_stack",
                            "_rotate_mix_", "_stack_gemm_", "_stack_gate_"))]
                print(json.dumps(dict(engine=args.engine,task=args.task,round=args.round,
                    **{k:v for k,v in item.items() if k!="samples_ms"})), flush=True)
            except torch.cuda.OutOfMemoryError as error:
                item.update(status="OOM",error=str(error))
                model.zero_grad(set_to_none=True)
            finally:
                del batches,optimizer
                gc.collect()
                torch.cuda.empty_cache()
                save()
        if args.engine=="dpa":
            modules=[dict(name=name,configured_level=m.triton_infer_level,training=m.training,
                value_path=type(getattr(m,"_triton_value_path",None)).__name__)
                for name,m in model.named_modules() if hasattr(m,"triton_infer_level")]
            record["triton_modules"]=modules
            record["compiled_core_slots"]=len(model.compiled_core_compute_cache)
            if not model.should_use_compile() or not record["compiled_core_slots"]:
                raise AssertionError("No active compiled DPA core")
            if not modules or not all(m["configured_level"]==2 and m["value_path"]=="_TritonSO2ValuePath" for m in modules):
                raise AssertionError("Selected DPA architecture does not support the level-2 value path")
            record["level2_value_active_in_training"]=False
            record["level2_value_active_in_inference"]=args.task=="inference"
        record["status"]="PASS" if all(r["status"]=="PASS" for r in record["results"]) else "PARTIAL_CAPACITY"
        save()
    except Exception as error:
        record.update(status="FAIL",error=str(error),traceback=traceback.format_exc())
        save()
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--engine",choices=("nequip","dpa"),required=True)
    parser.add_argument("--case",choices=("l2d2","l3d3"),required=True)
    parser.add_argument("--scope",choices=("final","persistent"),required=True)
    parser.add_argument("--task",choices=("inference","train"),required=True)
    parser.add_argument("--atoms",type=int,nargs="+",required=True)
    parser.add_argument("--round",type=int,required=True)
    parser.add_argument("--warmup",type=int,default=9)
    parser.add_argument("--repeats",type=int,default=30)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    with forbid_cuda_graph(purpose="NequIP / DPA4 matched full EFV") as policy:
        run(args,policy)
