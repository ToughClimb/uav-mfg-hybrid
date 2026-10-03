import numpy as np
import pytest
import torch

from uav_mfg.cli import load_settings
from uav_mfg.experiment import case_config
from uav_mfg.problem import Problem
from uav_mfg.transport import Grid, Transport
from uav_mfg.value import Potential, eikonal, sample_points


def planar():
    return Problem(case_config(load_settings("configs/paper_figures_2d.yaml"), "p2p_none", "diagnostic"))


def test_planar_disk_potential_has_no_vertical_motion():
    p = planar()
    model = Potential(p)
    with torch.no_grad():
        model.network[-1].weight.zero_()
    x = torch.tensor([[30., 50., 4.6], [30., 50., 5.4]], requires_grad=True)
    values = model(x)
    assert values[0] == values[1]
    grad = torch.autograd.grad(values.sum(), x)[0]
    assert torch.equal(grad[:, 2], torch.zeros(2))
    assert eikonal(model, x.detach(), torch.zeros(2), 1e-4).abs().max() < 1e-4
    assert model(torch.tensor([[90., 90., 5.], [80., 90., 5.]])).eq(0).all()
    points = sample_points(p, 1000, device="cpu")
    assert p.free_mask(points).all()
    assert ((points[:, 2] >= 4.5) & (points[:, 2] <= 5.5)).all()


def test_planar_transport_keeps_source_and_absorption_balance():
    p = planar()
    grid = Grid(p, (40, 40, 1))
    direction = np.asarray(p.target["center"]) - grid.points
    direction[..., 2] = 0
    velocity = 20 * direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    transport = Transport(grid, velocity)
    density, info = transport.solve(np.zeros(grid.shape), method="sparse")
    audit = transport.audit(density)
    assert audit["relative_steady_balance"] < 1e-10
    assert audit["outer_leakage"] == 0
    assert audit["obstacle_leakage"] == 0
    with pytest.raises(ValueError, match="one z cell"):
        Grid(p, (20, 20, 2))
    velocity[..., 2] = 0.1
    with pytest.raises(ValueError, match="zero vertical velocity"):
        Transport(grid, velocity)
