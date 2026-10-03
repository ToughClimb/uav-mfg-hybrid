# Reproduce the independent result set

Run these commands from the repository root with the project `.venv` described
in [../README.md](../README.md). The sequence reconstructs the measured examples
in `validation/`; it does not assert exact numerical reproduction of the paper.
All configs use explicitly declared reconstructed coefficients. The GPU workflow
uses two independent case processes and eight CPU threads per process. On WSL
reuse the existing GPU driver. PyPI packages use the configured Tsinghua mirror.

CUDA hardware, PyTorch versions, and stochastic optimization can affect stopping
iterations. Check saved status and fresh audits rather than assuming a budget
will converge. Retain unsuccessful trials, and use fresh output directories.
The selection scripts encode the measured run history below and refuse to
silently substitute another case when an expected run has not passed.

## 1. Low-demand 3D runs and independent baselines

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/verified_demo.yaml --profile demo --workers 2 \
  --output results/eswa_rebuild --device cuda
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/run_baselines.py \
  --method both --epochs 2000 --profile demo --output results/eswa_rebuild/baselines
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/verify_results.py \
  --source results/eswa_rebuild --output results/eswa_verified --workers 2
.venv/bin/python scripts/benchmark_acceleration.py
.venv/bin/python scripts/build_report.py --root results/eswa_verified
```

The independent FSM is limited to the isotropic, no-wind/no-obstacle case.
The monolithic PINN is a 2,000-epoch demonstration. It failed the measured mass
audit and is still shown as a failed baseline; it is not the paper's strengthened
20,000-epoch baseline. These concurrent timings do not establish method speed
rankings. The acceleration benchmark compares identical CPU/CUDA transport work,
including transfers, graph capture and audits, excluding operator assembly.

## 2. Low-demand 2D figures, preserving unsuccessful load trials

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/paper_figures_2d.yaml --profile paper --workers 2 \
  --output results/paper_figures_2d --device cuda
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/paper_figures_verified.yaml --profile paper --workers 2 \
  --output results/paper_figures_verified --device cuda
.venv/bin/python scripts/select_paper_figures.py
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/build_paper_report.py \
  --planar-root results/paper_figures_verified --three-d-root results/eswa_verified \
  --comparison-root results/eswa_verified --output results/paper_style_low_load
```

The first P2P load q=0.01 exceeded the reconstructed density bound 0.05 and
failed. The second P2P load q=0.006 was recomputed from scratch. Homing uses
q=0.0001. The accepted set selects three Homing cases from the first trial and
three P2P cases from the second. These are explicit 100×100 XY equations with
one unit-thickness layer and disk-shaped source/target, not disguised 3D slices.

## 3. Increased demand and unchanged-parameter continuations

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/run_high_density.py --workers 2
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/high_density_2d.yaml --profile paper --cases homing_uniform \
  --output results/high_density_continuation_2d \
  --resume results/high_density_2d/homing_uniform/checkpoint.pt --device cuda
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/high_density_homing_obstacle.yaml --profile paper \
  --output results/high_density_homing_obstacle --device cuda
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/high_density_homing_obstacle.yaml --profile paper \
  --output results/high_density_homing_obstacle_continued \
  --resume results/high_density_homing_obstacle/homing_uniform_obstacle/checkpoint.pt --device cuda
.venv/bin/python scripts/select_high_density.py
```

Initial 2D P2P q=0.018 is three times the accepted low demand; the two unobstructed
Homing cases use q=0.0003. Initial obstacle-Homing at three times demand clips
density and is retained as an unsuccessful trial. Its separately accepted
reconstruction uses q=0.0002, starts from zero, and requires a continuation.
The 3D source q=0.01 is ten times the accepted low demand. The density bound
0.25 replaces the reconstruction-specific 0.05 bound; speed closure, rho_jam,
wind, and geometry remain unchanged. This bound is not a recovered paper value.

In the measured runs, uniform Homing needed 24 additional outer iterations,
and obstacle Homing at double demand needed 36 additional iterations. These
counts are measured outcomes, not cross-hardware guarantees. If a required
source run is already converged, a continuation is permitted with the same
physical parameters; if it remains unconverged, record further continuations
explicitly and update the selection record to their actual source directories.

Checkpoints carry cumulative projection mass. For older reconstruction
checkpoints without that field, adjacent `picard.csv` and `metadata.json` and
the referenced continuation chain are required. Missing history causes an
error. Checkpoints from the historical `uavpinn` implementation are unsupported.

## 4. Audit and render the increased-demand figures

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/build_paper_report.py \
  --planar-root results/high_density_verified_2d \
  --three-d-root results/high_density_verified_3d \
  --comparison-root results/eswa_verified \
  --baseline-planar-root results/paper_figures_verified \
  --baseline-three-d-root results/eswa_verified \
  --output results/paper_style
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory results
```

Open `http://127.0.0.1:8765/paper_style/report.html`. The renderer does not
require or distribute the manuscript. For an authorized private appearance
comparison, optionally add `--reference-paper reference/paper`, whose folder
must contain `main.tex` and all five named reference PNGs in `figures/`.
Missing explicitly requested reference files are errors.

The figure set is accepted only after reloading numeric fields and recomputing
actual shared face fluxes. Required checks include convergence status, zero
cumulative projection loss, zero outer/obstacle leakage, relative mass imbalance
and final continuity residual below 1e-3, hard target conditions, and unchanged
numeric-file hashes after rendering. Separate linear full-range and square-root
displays leave the numeric fields unchanged; density above the common 0–0.05
color range is marked by an upper colorbar arrow.

Method figure 5 continues to use the separately matched, low-demand 3D q=0.001
hybrid/baseline comparison. It is not compared against the increased-demand
fields as though they belonged to the same experiment. Density remains more
concentrated near the target than in the manuscript, and the outer spherical-source
height planes remain empty; both differences are retained and disclosed.

## 5. Representative main-grid 3D case

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m uav_mfg.cli run \
  --config configs/verified_demo.yaml --profile paper --cases p2p_none \
  --output results/eswa_main --device cuda
```

Only this representative 100×100×50 case is part of the recorded main-grid
validation. Do not describe all eight demonstration-grid cases as main-grid
reproductions. New runs save their own environments and source hashes; public
validation includes hashes of the code that actually generated earlier fields.
