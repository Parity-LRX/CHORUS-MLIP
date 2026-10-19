"""Same-allocation paired subprocess driver, with alternating engine order."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main(args):
    out=args.root/"results"/args.run/args.case/args.scope
    out.mkdir(parents=True,exist_ok=False)
    orders=([["dpa","nequip"]] if args.pilot else [["nequip","dpa"],["dpa","nequip"]])
    manifest=dict(status="RUNNING",job=os.environ["SLURM_JOB_ID"],
        case=args.case,scope=args.scope,rounds=len(orders),engine_order=orders,pilot=args.pilot,commands=[])
    path=out/"commands.json"
    worker=Path(__file__).parent/"worker.py"
    for round_id, engines in enumerate(manifest["engine_order"]):
        for engine in engines:
            for task in ("inference","train"):
                atoms=([128] if args.pilot else [128,1024,4096] if task=="inference" else [128,1024])
                result=out/f"{engine}_{task}_r{round_id}.json"
                log=result.with_suffix(".log")
                env=os.environ.copy()
                env["PYTHONPATH"]=env["NEQUIP_PATH" if engine=="nequip" else "DPA_PATH"]
                command=[env["NEQUIP_PY" if engine=="nequip" else "DPA_PY"],str(worker),
                    "--root",str(args.root),"--engine",engine,"--case",args.case,"--scope",args.scope,
                    "--task",task,"--round",str(round_id),"--atoms",*map(str,atoms),
                    "--warmup","3" if args.pilot else "9",
                    "--repeats","3" if args.pilot else "30" if task=="inference" else "18","--out",str(result)]
                item=dict(engine=engine,task=task,round=round_id,command=command,log=str(log),status="RUNNING")
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
                    print(log.read_text()[-12000:],flush=True)
                    raise RuntimeError("Paired worker failed; see recorded exact command and log")
    manifest["status"]="DONE"
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print("PAIR_DONE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--run",required=True)
    parser.add_argument("--case",required=True)
    parser.add_argument("--scope",required=True)
    parser.add_argument("--pilot",action="store_true")
    main(parser.parse_args())
