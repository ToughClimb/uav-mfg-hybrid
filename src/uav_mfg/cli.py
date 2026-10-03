"""Run the independent solver and render its saved outputs."""

import argparse
import json
import sys
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import torch
import yaml

from .experiment import case_config, run_hybrid, save_json
from .plots import make_figures


def load_settings(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if "extends" not in settings:
        return settings
    base = load_settings(path.parent / settings.pop("extends"))
    def merge(destination, updates):
        for key, value in updates.items():
            if isinstance(value,dict) and isinstance(destination.get(key),dict):
                merge(destination[key],value)
            else:
                destination[key] = value
    merge(base,settings)
    return base


def run_case_worker(config,output,device,seed,resume):
    return {"case":config["name"],**run_hybrid(config,output,device,seed,resume)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", default="configs/reconstruction.yaml")
    run.add_argument("--profile", choices=("demo", "diagnostic", "paper"), default="demo")
    run.add_argument("--cases", nargs="+")
    run.add_argument("--output", type=Path, default=Path("results/eswa_reconstruction"))
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--resume", type=Path)
    run.add_argument("--workers",type=int,default=1,help="Independent case processes (spawn); 2 recommended on one GPU")
    plot = commands.add_parser("plot")
    plot.add_argument("--output", type=Path, default=Path("results/eswa_reconstruction"))
    args = parser.parse_args()
    if args.command == "plot":
        print(make_figures(args.output))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu explicitly")
    base = load_settings(args.config)
    names = args.cases or base["cases"]
    if args.resume and len(names) != 1:
        raise ValueError("--resume requires exactly one case")
    if args.workers<1 or len(set(names))!=len(names):
        raise ValueError("workers must be positive and case names must be unique")
    for name in names:
        if (args.output / name / "metadata.json").exists():
            raise FileExistsError(f"Refusing to overwrite an experiment: {args.output / name}; choose a fresh output root")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "command.json", {"argv": sys.argv, "seed": args.seed, "profile": args.profile})
    summary = []
    if args.workers>1:
        threads=str(base.get("cpu_threads",8))
        os.environ["OMP_NUM_THREADS"]=threads;os.environ["OPENBLAS_NUM_THREADS"]=threads
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as executor:
            jobs=[executor.submit(run_case_worker,case_config(base,name,args.profile),args.output/name,args.device,args.seed,args.resume) for name in names]
            for job in as_completed(jobs):
                summary.append(job.result());save_json(args.output/"summary.json",summary)
                print(json.dumps(summary[-1]),flush=True)
    else:
        for name in names:
            summary.append(run_case_worker(case_config(base,name,args.profile),args.output/name,args.device,args.seed,args.resume))
            save_json(args.output/"summary.json",summary);print(json.dumps(summary[-1]),flush=True)
    print(f"Figures: {make_figures(args.output)}")


if __name__ == "__main__":
    main()
