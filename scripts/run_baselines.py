"""Run independent benign-setting comparisons; budgets are explicit."""

import argparse
import json
import os
import sys
from pathlib import Path

# Limit independent numerical libraries before imports; actual CPU work uses
# the configurable OpenMP reference kernel and CUDA handles neural work.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

from uav_mfg.baselines import run_monolithic, run_reference
from uav_mfg.cli import load_settings
from uav_mfg.experiment import case_config, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/verified_demo.yaml")
    parser.add_argument("--profile",choices=("diagnostic","demo","paper"),default="demo")
    parser.add_argument("--output",type=Path,default=Path("results/eswa_rebuild/baselines"))
    parser.add_argument("--method",choices=("fsm","monolithic","both"),default="both")
    parser.add_argument("--epochs",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    config=case_config(load_settings(args.config),"p2p_none",args.profile)
    config["seed"]=args.seed
    args.output.mkdir(parents=True,exist_ok=True)
    save_json(args.output/"command.json",{"argv":sys.argv,"seed":args.seed,"epochs":args.epochs,"profile":args.profile})
    summary=[]
    for name,run in (("fsm",run_reference),("monolithic",run_monolithic)):
        if args.method not in (name,"both"):
            continue
        directory=args.output/name
        directory.mkdir(exist_ok=True)
        save_json(directory/"metadata.json",{"status":"running","method":name,"config":config,"seed":args.seed})
        try:
            diagnostics=run(config,directory,seed=args.seed,epochs=args.epochs) if name=="monolithic" else run(config,directory)
        except (Exception,KeyboardInterrupt) as error:
            save_json(directory/"metadata.json",{"status":"failed","method":name,"config":config,"seed":args.seed,
                                               "error":f"{type(error).__name__}: {error}"})
            raise
        summary.append({"method":name,**diagnostics})
        save_json(args.output/"summary.json",summary)
        print(json.dumps(summary[-1]),flush=True)


if __name__=="__main__":
    main()
