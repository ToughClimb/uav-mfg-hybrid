"""Measure CPU versus audited CUDA transport after warmup, with parity checks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from uav_mfg.cli import load_settings
from uav_mfg.problem import Problem
from uav_mfg.transport import Grid, Transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/acceleration_benchmark.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.cuda.init()
    config = load_settings("configs/verified_demo.yaml")
    records = []
    for shape in ((40, 40, 16), (100, 100, 50)):
        grid = Grid(Problem(config), shape)
        velocity = config["physics"]["v_free"] * (np.asarray(config["target"]["center"]) - grid.points)
        velocity /= np.linalg.norm(np.asarray(config["target"]["center"]) - grid.points, axis=-1, keepdims=True).clip(min=1e-12)
        velocity[grid.target] = 0
        transport = Transport(grid, velocity)
        initial = np.zeros(shape)
        # Warmup includes CUDA graph capture and allocator initialization.
        transport.solve(initial, max_steps=100, change_tolerance=0, backend="cuda")
        timings = {"cpu": [], "cuda": []}
        solutions, audits = {}, {}
        for _ in range(args.repeats):
            for backend in timings:
                torch.cuda.synchronize()
                started = time.perf_counter()
                solutions[backend], audits[backend] = transport.solve(
                    initial, max_steps=args.steps, change_tolerance=0, backend=backend)
                torch.cuda.synchronize()
                timings[backend].append(time.perf_counter() - started)
        difference = float(np.max(np.abs(solutions["cpu"] - solutions["cuda"])))
        if difference > 1e-12:
            raise AssertionError(f"CPU / GPU density mismatch: {difference}")
        record = {"grid": list(shape), "steps": args.steps, "repeats": args.repeats,
                  "timings_seconds": timings, "median_cpu_seconds": float(np.median(timings["cpu"])),
                  "median_cuda_seconds": float(np.median(timings["cuda"])),
                  "speedup": float(np.median(timings["cpu"]) / np.median(timings["cuda"])),
                  "maximum_density_difference": difference, "audit_every_step": True,
                  "maximum_step_budget_error": {key: value["max_step_mass_budget_error"] for key, value in audits.items()},
                  "scope": "500 explicit transport steps; includes graph capture, transfers and audits; excludes operator assembly",
                  "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads()}
        records.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
