"""Audited Picard coupling; outputs retain settings, provenance, and stopping state."""

import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .problem import Problem
from .transport import Grid, Transport
from .value import Potential, evaluate, sample_points, train_value
from .initialization import initialize_potential


def case_config(base, name, profile):
    config = copy.deepcopy(base)
    config["name"] = name
    config["profile"] = profile
    config["grid"] = config["profiles"][profile]["grid"]
    if name.startswith("homing"):
        config["target"] = {"type": "sphere", "center": [50, 50, 5], "radius": 5}
        config["source"] = {"type": "homing", "rate": config.get("homing_rate",0.0001)}
    if "uniform" in name:
        config["wind"] = {"type": "uniform", "velocity": [5, 0, 0]}
    if "vortex" in name:
        config["wind"] = {"type": "vortex", "center": [50, 50, 15], "radius": 60, "strength": 7}
    if "height" in name:
        config["wind"] = {"type": "height_dependent", "velocity": [6, -2, 0], "decay": 1.8}
    if "obstacle" in name:
        config["obstacles"] = [{"lower": [60, 40, 0], "upper": [70, 50, 30]}]
    # Explicit 2D cases are separate experiments, never mislabeled 3D slices.
    if config.get("spatial_dimension", 3) == 2:
        if config["target"]["type"] == "sphere":
            config["target"] = {**config["target"], "type": "disk"}
        if config["source"]["type"] == "sphere":
            config["source"] = {**config["source"], "type": "disk"}
        if config["wind"]["type"] == "vortex":
            config["wind"]["center"][2] = sum(config["domain"][2]) / 2
    return config


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def resume_projection_mass(checkpoint, path, seen=None):
    """Carry projection accounting across continuations, including older local runs."""
    if "cumulative_projection_mass_removed" in checkpoint:
        value = float(checkpoint["cumulative_projection_mass_removed"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid cumulative projection mass in checkpoint")
        return value
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Cyclic checkpoint continuation history")
    seen.add(path)
    history, metadata = path.parent / "picard.csv", path.parent / "metadata.json"
    if not history.exists() or not metadata.exists():
        raise ValueError("Older checkpoint requires adjacent picard.csv and metadata.json for projection accounting")
    with history.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or int(rows[-1]["outer"]) != checkpoint["outer"]:
        raise ValueError("Checkpoint and projection history have different outer iterations")
    value = sum(float(row["accepted_projection_mass_removed"]) for row in rows)
    previous = json.loads(metadata.read_text(encoding="utf-8")).get("resume")
    if previous:
        earlier = torch.load(previous, map_location="cpu", weights_only=True)
        value += resume_projection_mass(earlier, previous, seen)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Invalid cumulative projection history")
    return value


def environment_metadata():
    import matplotlib
    import scipy

    metadata = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform(),
                "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for key, command in (("git_status", ["git", "status", "--short"]), ("nvidia_smi", ["nvidia-smi"])):
        if shutil.which(command[0]) is None:
            metadata[key] = {"available": False, "reason": f"Optional diagnostic command {command[0]} is not installed"}
            continue
        result = subprocess.run(command, capture_output=True, text=True)
        metadata[key] = result.stdout if result.returncode == 0 else {"error": result.stderr, "returncode": result.returncode}
    paper = Path("reference/paper/main.tex")
    if paper.exists():
        metadata["manuscript_sha256"] = hashlib.sha256(paper.read_bytes()).hexdigest()
    metadata["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path("src").rglob("*"))
                                 if path.suffix in (".py", ".cpp")}
    return metadata


def run_hybrid(config, output, device="cuda", seed=42, resume=None):
    """Record setup failures as well as failures after the coupling loop starts."""
    output=Path(output)
    if (output / "metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite an experiment: {output}; use a fresh output directory")
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    save_json(output/"metadata.json",{"status":"initializing","config":config,"seed":seed,
                                     "device":str(device),"resume":str(resume) if resume else None})
    try:
        return _run_hybrid(config,output,device,seed,resume)
    except (Exception,KeyboardInterrupt) as error:
        metadata=json.loads((output/"metadata.json").read_text())
        metadata.update(status="failed",error=f"{type(error).__name__}: {error}",
                        duration_seconds=time.perf_counter()-started)
        save_json(output/"metadata.json",metadata)
        raise


def _run_hybrid(config, output, device="cuda", seed=42, resume=None):
    config = copy.deepcopy(config)
    # An explicit CPU request applies to both the PINN and the transport solve.
    # CUDA failures remain errors when CUDA is requested.
    if torch.device(device).type == "cpu":
        config["transport"]["backend"] = "cpu"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started=time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(config.get("cpu_threads",min(8,os.cpu_count() or 1)))
    problem = Problem(config)
    grid = Grid(problem, tuple(config["grid"]))
    checks = sample_points(problem, 12000, device=device)
    if problem.dimension == 2 and torch.any(problem.wind(checks)[:, 2].abs() > 1e-12):
        raise ValueError("A two-dimensional problem requires zero vertical wind")
    wind_max = float(torch.linalg.vector_norm(problem.wind(checks), dim=-1).max())
    # Analytic upper bounds for all built-in analytical fields supplement sampling.
    wind_cfg = problem.wind_config
    if wind_cfg["type"] in ("uniform", "height_dependent"):
        wind_max = max(wind_max, float(np.linalg.norm(wind_cfg["velocity"])))
    elif wind_cfg["type"] == "vortex":
        wind_max = max(wind_max, abs(wind_cfg["strength"]) * np.exp(-0.5))
    elif wind_cfg["type"] == "gridded":
        wind_max = max(wind_max, float(np.linalg.norm(problem.wind_values, axis=-1).max()))
    if problem.physics["v_min"] <= wind_max + problem.physics["controllability_margin"]:
        raise ValueError(f"Strong controllability fails: v_min={problem.physics['v_min']}, wind bound={wind_max}")
    model = Potential(problem, config["network"]["widths"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    rho = np.zeros(grid.shape)
    start_outer = 0
    inherited_projection_mass = 0.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        physical_keys=("domain","target","source","obstacles","wind","physics","barrier","network","spatial_dimension")
        if checkpoint["grid"] != list(grid.shape) or any(checkpoint["config"].get(key)!=config.get(key) for key in physical_keys):
            raise ValueError("Resume checkpoint geometry, source, wind, physics or grid differs from requested problem")
        inherited_projection_mass = resume_projection_mass(checkpoint, resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rho = checkpoint["density"].cpu().numpy()
        start_outer = checkpoint["outer"] + 1
        initialization={"method":"resume", "checkpoint":str(resume)}
    else:
        initialization=initialize_potential(model,grid,rho,config["training"])
    config["seed"] = seed
    with (output / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    metadata = {"status": "running", "seed": seed, "device": str(device), "wind_bound": wind_max,
                "config": config, "environment": environment_metadata(), "resume": str(resume) if resume else None,
                "initialization":initialization,"inherited_projection_mass_removed":inherited_projection_mass,
                "setup_seconds":time.perf_counter()-started,
                "cpu_threads":torch.get_num_threads()}
    save_json(output / "metadata.json", metadata)
    history, iterations = [], []
    converged = False
    previous_velocity = None
    try:
        for outer in range(start_outer, start_outer + config["coupling"]["max_iterations"]):
            phase_start=time.perf_counter()
            for group in optimizer.param_groups:
                group["lr"] = config["training"]["learning_rate"] / (1 + 0.3 * outer)
            epochs = config["training"]["initial_epochs"] if outer == 0 else config["training"]["subsequent_epochs"]
            if outer>0 and config["training"].get("value_mean_tolerance"):
                _,_,existing_residual=evaluate(model,grid,rho,config["training"]["epsilon"])
                valid=grid.active & (problem.obstacle_distance(torch.from_numpy(grid.points)).numpy()>config["barrier"]["buffer"])
                if (np.abs(existing_residual[valid]).mean()<config["training"]["value_mean_tolerance"]
                        and np.quantile(np.abs(existing_residual[valid]),.95)<config["training"]["value_p95_tolerance"]):
                    epochs=0
            train_value(model, optimizer, grid, rho, config["training"], epochs, outer, history)
            value_seconds=time.perf_counter()-phase_start
            phase_start=time.perf_counter()
            phi, velocity, residual = evaluate(model, grid, rho, config["training"]["epsilon"])
            transport = Transport(grid, velocity)
            velocity_seconds=time.perf_counter()-phase_start
            phase_start=time.perf_counter()
            proposal, inner = transport.solve(rho, **config["transport"])
            transport_seconds=time.perf_counter()-phase_start
            clipped = np.clip(proposal, 0, problem.physics["rho_max"])
            alpha = config["coupling"]["alpha"]
            if not 0 < alpha <= 1:
                raise ValueError("Picard relaxation alpha must be in (0,1]")
            accepted = (1 - alpha) * rho + alpha * clipped
            relative_change = float(np.linalg.norm(accepted - rho) / max(np.linalg.norm(rho), 1e-30))
            raw_change = float(np.linalg.norm(proposal - rho) / max(np.linalg.norm(rho), 1e-30))
            projected_mass_removed = float((proposal - clipped).sum() * grid.cell_volume)
            row = {"outer": outer, "accepted_density_change": relative_change, "raw_density_change": raw_change,
                   "trained_value_epochs":epochs,"value_seconds":value_seconds,"velocity_seconds":velocity_seconds,
                   "transport_seconds":transport_seconds,"transport_backend":inner["backend"],
                   "velocity_change": float(np.linalg.norm(velocity - previous_velocity) / max(np.linalg.norm(previous_velocity), 1e-30)) if previous_velocity is not None else None,
                   "projection_mass_removed": projected_mass_removed, "accepted_projection_mass_removed": alpha * projected_mass_removed,
                   "inner_converged": inner["converged"], "inner_steps": inner["steps"],
                   "inner_pde_residual": inner["pde_relative_residual"], "inner_absorption_ratio": inner["absorption_over_source"],
                   "inner_max_budget_error": inner["max_step_mass_budget_error"],
                   "eikonal_mean": float(np.mean(np.abs(residual[grid.active])))}
            rho, previous_velocity = accepted, velocity
            final_phi, final_velocity, final_residual = evaluate(model, grid, rho, config["training"]["epsilon"])
            final_transport = Transport(grid, final_velocity)
            final_audit = final_transport.audit(rho)
            row["accepted_absorption_ratio"] = final_audit["absorption_over_source"]
            row["accepted_steady_balance"] = final_audit["relative_steady_balance"]
            row["accepted_pde_residual"] = final_audit["pde_relative_residual"]
            iterations.append(row)
            write_csv(output / "training.csv", history)
            write_csv(output / "picard.csv", iterations)
            print(f"{config['name']} k={outer + 1}: eik={row['eikonal_mean']:.3g}, "
                  f"d_rho={relative_change:.3g}, absorption/source={final_audit['absorption_over_source']:.6f}, "
                  f"inner={inner['converged']}, projection={projected_mass_removed:.3g}", flush=True)
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "density": torch.from_numpy(rho), "outer": outer, "grid": list(grid.shape), "target": problem.target,
                          "config": config,
                          "cumulative_projection_mass_removed": inherited_projection_mass + sum(
                              item["accepted_projection_mass_removed"] for item in iterations)}
            torch.save(checkpoint, output / "checkpoint.pt")
            if (relative_change < config["coupling"]["tolerance"] and inner["converged"]
                    and final_audit["relative_steady_balance"] < config["transport"]["balance_tolerance"]
                    and final_audit["pde_relative_residual"] < config["coupling"].get("continuity_tolerance",float("inf"))
                    and abs(projected_mass_removed) < 1e-10):
                converged = True
                break
        valid = grid.active & (problem.obstacle_distance(torch.from_numpy(grid.points)).numpy() > config["barrier"]["buffer"])
        diagnostics = {**final_audit, "picard_converged": converged, "last_inner_converged": inner["converged"],
                       "picard_iterations": len(iterations), "eikonal_mean": float(np.abs(final_residual[valid]).mean()),
                       "total_outer_iterations":outer+1,
                       "accepted_continuity_tolerance":config["coupling"].get("continuity_tolerance"),
                       "eikonal_p95": float(np.quantile(np.abs(final_residual[valid]), 0.95)),
                       "maximum_inner_mass_budget_error": max(row["inner_max_budget_error"] or 0 for row in iterations),
                       "total_projection_mass_removed": inherited_projection_mass + sum(
                           row["accepted_projection_mass_removed"] for row in iterations),
                       "duration_seconds": time.perf_counter() - started}
        # The zero-wind / zero-obstacle analytic value is an independent check;
        # density coupling may make it approximate, so do not call it an FSM result.
        if not problem.obstacles and wind_cfg["type"] == "none":
            analytic = problem.target_distance(torch.from_numpy(grid.points)).clamp_min(0).numpy() / problem.physics["v_free"]
            diagnostics["distance_over_free_speed_reference_relative_l2"] = float(np.linalg.norm((final_phi - analytic)[grid.active]) / np.linalg.norm(analytic[grid.active]))
        np.savez_compressed(output / "fields.npz", phi=final_phi, density=rho, velocity=final_velocity,
                            eikonal_residual=final_residual, source=grid.source, active=grid.active,
                            target=grid.target, obstacle=grid.obstacle, x=grid.axes[0], y=grid.axes[1], z=grid.axes[2])
        save_json(output / "diagnostics.json", diagnostics)
        metadata.update({"status": "converged" if converged else "completed_unconverged",
                         "duration_seconds": diagnostics["duration_seconds"], "diagnostics": diagnostics})
        save_json(output / "metadata.json", metadata)
        return diagnostics
    except (Exception, KeyboardInterrupt) as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}",
                         "duration_seconds": time.perf_counter() - started})
        save_json(output / "metadata.json", metadata)
        raise
