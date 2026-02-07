# UAV-MFG-Hybrid (UAVPINN++)

Hybrid physics-informed learning + conservative finite-volume solver for a 3D mean-field game (MFG) model of UAV traffic in wind.

中文说明见：[README_zh.md](./README_zh.md)

This repository provides two training routes:

- **Route B (recommended)**: **Fixed-point (Picard) iteration**
  - Train a PINN for the potential field **φ** with **ρ fixed**
  - Construct the velocity field **u** from φ
  - Update the density field **ρ** using a **conservative upwind finite-volume** solver
  - Apply relaxation **ρ ← (1−α)ρ + α ρ̃**
- **Route A (baseline)**: End-to-end dual-network PINN (jointly trains φ and ρ)

Outputs are written to `runs/<timestamp>_<exp_name>/` with logs, checkpoints, and plots.

## Features

- **3D domain** with configurable bounds
- **Target region (sink)**: sphere or AABB
- **Obstacle(s)**: AABB or union of multiple AABBs
- **Wind fields**:
  - `none`, `uniform`, `vortex`, `height_dependent`
  - Composable wind: **base + patches** via `CompositeWind` + `RegionConstantWind` (hard/smooth transition)
- **Visualization**:
  - 2D slices for φ and ρ
  - Optional **ParaView VTK** export for 3D fields

## Repository layout

- `scripts/train.py`: main CLI (Route A / Route B)
- `scripts/batch_train.py`: one-click batch reproduction for Route B
- `scripts/visualize.py`: visualize from a saved checkpoint
- `uavpinn/`: core library (geometry, wind, physics, solvers, training, viz)

## Installation

### 1) Create an environment

Python **3.10+** is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/Mac
# source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Quickstart

### Train (single config)

Route B (fixed-point hybrid, recommended):

```bash
python scripts/train.py --config path/to/config.yaml --route B --device auto
```

Route A (end-to-end baseline):

```bash
python scripts/train.py --config path/to/config.yaml --route A --device auto
```

Notes:

- **Outputs**: `runs/<timestamp>_<exp_name>/`
- To disable automatic plotting after training: add `--no-viz`
- If you intentionally run an **uncontrollable** case (e.g., wind exceeds max UAV speed), you may want:

```bash
python scripts/train.py --config path/to/config.yaml --route B --skip-physics-checks
```

### Batch reproduce (Route B)

Run multiple YAMLs:

```bash
python scripts/batch_train.py --config-dir path/to/configs --glob "*.yaml" --device auto
```

Or specify configs explicitly (repeatable):

```bash
python scripts/batch_train.py --config a.yaml --config b.yaml --device auto
```

Disable visualization:

```bash
python scripts/batch_train.py --config-dir path/to/configs --no-viz
```

### Visualize from a checkpoint

```bash
python scripts/visualize.py --checkpoint runs/<run_dir>/checkpoints/checkpoint_final.pt
```

ParaView export (VTK):

```bash
python scripts/visualize.py --checkpoint runs/<run_dir>/checkpoints/checkpoint_final.pt --paraview
```

You can also export VTK directly at the end of training:

```bash
python scripts/train.py --config path/to/config.yaml --route B --paraview
```

## Configuration

Training is driven by a YAML config (passed via `--config`).

Key sections include:

- `domain`: bounds in x/y/z
- `scenario`: `homing` or `p2p`
- `target`: geometry
- `obstacles`: optional list
- `wind`: base + optional patches
- `physics`: parameters for fundamental diagram and source term
- `network`, `training`, `fixed_point`, `sampling`, `numerical`, `rho_solver`

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE).
