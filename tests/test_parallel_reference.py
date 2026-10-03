from pathlib import Path

import numpy as np
import pytest

from uav_mfg.cli import load_settings
from uav_mfg.problem import Problem
from uav_mfg.reference import fast_sweeping
from uav_mfg.transport import Grid


@pytest.mark.skipif(not Path("build/fsm_kernel.so").exists(), reason="optional native kernel not built")
def test_parallel_sweeps_match_independent_python_solver():
    config = load_settings("configs/verified_demo.yaml")
    grid = Grid(Problem(config), (10, 10, 4))
    speed = 14 + 3 * grid.points[..., 0] / 100
    expected, original = fast_sweeping(grid, speed)
    actual, parallel = fast_sweeping(grid, speed, backend="openmp", threads=4)
    assert original["converged"] and parallel["converged"]
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-9)
