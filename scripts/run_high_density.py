"""Recompute increased demand with two GPU workers and full mass diagnostics."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import sys

from uav_mfg.cli import load_settings, run_case_worker
from uav_mfg.experiment import case_config, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/density_increase"))
    parser.add_argument("--planar-output", type=Path, default=Path("results/high_density_2d"))
    parser.add_argument("--three-d-output", type=Path, default=Path("results/high_density_3d"))
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    jobs = []
    for config_path, profile, root in (
        ("configs/high_density_2d.yaml", "paper", args.planar_output),
        ("configs/high_density_3d.yaml", "demo", args.three_d_output),
    ):
        base = load_settings(config_path)
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "command.json", {"argv": sys.argv, "configuration": config_path,
                                         "profile": profile, "seed": args.seed, "from_scratch": True})
        for name in base["cases"]:
            directory = root / name
            if directory.exists():
                raise FileExistsError(f"Refusing to overwrite an experiment: {directory}")
            jobs.append((case_config(base, name, profile), directory))
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "command.json", {"argv": sys.argv, "workers": args.workers,
                                             "seed": args.seed, "experiments": [str(path) for _, path in jobs]})
    summary = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        tasks = {executor.submit(run_case_worker, config, directory, "cuda", args.seed, None): directory
                 for config, directory in jobs}
        for task in as_completed(tasks):
            result = {"directory": str(tasks[task]), **task.result()}
            summary.append(result)
            save_json(args.output / "summary.json", summary)
            print(f"FINISHED {tasks[task]}: converged={result['picard_converged']}; rho_max={result['rho_max']:.6g}", flush=True)


if __name__ == "__main__":
    main()
