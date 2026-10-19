"""A bounded same-state native-dataflow round, with same-GPU baselines."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main(args):
    out=args.root/"results"/args.run/args.case/args.scope
    out.mkdir(parents=True,exist_ok=False)
    labels=(["no_chorus","no_chorus_materialized","previous"] if args.mode=="baseline" else
            ["previous","optimized"] if args.mode=="screen" else ["no_chorus","previous","optimized","dpa"])
    orders=[labels] if args.mode!="formal" else [labels,list(reversed(labels))]
    manifest=dict(status="RUNNING",mode=args.mode,job=os.environ["SLURM_JOB_ID"],
        case=args.case,scope=args.scope,engine_order=orders,commands=[])
    path=out/"commands.json"
    worker=Path(os.environ["SOURCE"])/"tools/pair/worker.py"
    for round_id,order in enumerate(orders):
        for label in order:
            engine="dpa" if label=="dpa" else "nequip"
            env=os.environ.copy()
            env["NEQUIP_CHORUS_VARIANT"]=label
            env["PROFILE_DETAIL"]="1" if args.mode=="baseline" or (
                args.mode=="formal" and label=="optimized") else "0"
            # The final bare-backbone baseline receives the same generic
            # dataflow optimization as CHORUS; the initial baseline diagnoses
            # the unchanged previous implementation separately.
            candidate=label=="optimized" or (args.mode=="formal" and label=="no_chorus")
            env["PYTHONPATH"]=(env["DPA_PATH"] if engine=="dpa" else
                               env["NEQUIP_PATH"] if candidate else env["NEQUIP_BEFORE_PATH"])
            for task in ("inference","train"):
                if args.mode=="baseline":atoms=[1024]
                elif args.mode=="screen":atoms=[128,1024]
                else:atoms=[128,1024,4096] if task=="inference" else [128,1024]
                result=out/f"{label}_{task}_r{round_id}.json"
                log=result.with_suffix(".log")
                command=[env["DPA_PY" if engine=="dpa" else "NEQUIP_PY"],str(worker),
                    "--root",str(args.root),"--engine",engine,"--case",args.case,"--scope",args.scope,
                    "--task",task,"--round",str(round_id),"--atoms",*map(str,atoms),
                    "--warmup","9","--repeats",str(30 if args.mode=="formal" and task=="inference" else 18),
                    "--out",str(result)]
                item=dict(label=label,task=task,round=round_id,command=command,log=str(log),status="RUNNING")
                manifest["commands"].append(item)
                path.write_text(json.dumps(manifest,indent=2)+"\n")
                print(json.dumps(item),flush=True)
                started=time.monotonic()
                with log.open("w") as handle:
                    process=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
                item.update(exit_code=process.returncode,elapsed_seconds=time.monotonic()-started,
                            status="DONE" if process.returncode==0 else "FAIL")
                path.write_text(json.dumps(manifest,indent=2)+"\n")
                if process.returncode:
                    manifest["status"]="FAIL"
                    path.write_text(json.dumps(manifest,indent=2)+"\n")
                    print(log.read_text()[-10000:],flush=True)
                    raise RuntimeError("Dataflow round worker failed")
    manifest["status"]="DONE"
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print("ROUND_DONE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--run",required=True)
    parser.add_argument("--case",required=True)
    parser.add_argument("--scope",required=True)
    parser.add_argument("--mode",choices=("baseline","screen","formal"),required=True)
    main(parser.parse_args())
