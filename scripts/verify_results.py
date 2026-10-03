"""Continue saved independent cases to audited final fields; retain all earlier runs."""

import argparse
import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

from uav_mfg.cli import load_settings, run_case_worker
from uav_mfg.experiment import case_config, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("results/eswa_rebuild"))
    parser.add_argument("--continuation",type=Path,default=Path("results/eswa_continuation"))
    parser.add_argument("--output",type=Path,default=Path("results/eswa_verified"))
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    settings=load_settings("configs/continuation.yaml")
    args.output.mkdir(parents=True,exist_ok=True)
    selection={"argv":sys.argv,"accepted_continuity_tolerance":settings["coupling"]["continuity_tolerance"],
               "disclosure":"Additional continuation with unchanged physical parameters; original runs are retained", "cases":{}}
    jobs=[]
    for name in settings["cases"]:
        source=args.continuation/name if (args.continuation/name/"checkpoint.pt").exists() else args.source/name
        if not (source/"checkpoint.pt").exists():
            raise FileNotFoundError(source/"checkpoint.pt")
        config=case_config(settings,name,"demo")
        selection["cases"][name]={"resume_from":str(source),"earlier_diagnostics":json.loads((source/"diagnostics.json").read_text())}
        jobs.append((config,args.output/name,source/"checkpoint.pt"))
    save_json(args.output/"selection.json",selection)
    save_json(args.output/"command.json",{"argv":sys.argv,"workers":args.workers,"seed":42})
    if (args.output/"baselines").exists():
        raise FileExistsError("Verification destination already has baselines; choose a fresh output directory")
    shutil.copytree(args.source/"baselines",args.output/"baselines")
    summary=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as executor:
        pending=[executor.submit(run_case_worker,config,output,"cuda",42,resume) for config,output,resume in jobs]
        for future in as_completed(pending):
            summary.append(future.result())
            save_json(args.output/"summary.json",summary)
            print(json.dumps(summary[-1]),flush=True)


if __name__=="__main__":
    main()
