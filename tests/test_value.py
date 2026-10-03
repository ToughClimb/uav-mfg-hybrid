import numpy as np
import torch
import yaml

from uav_mfg.problem import Problem
from uav_mfg.transport import Grid
from uav_mfg.value import Potential, eikonal, sample_points


def problem():
    with open("configs/reconstruction.yaml") as stream:
        return Problem(yaml.safe_load(stream))


def test_hard_target_zero_with_obstacle_barrier():
    p = problem()
    p.obstacles = [{"lower": [60,40,0], "upper": [70,50,30]}]
    model = Potential(p)
    center = torch.tensor(p.target["center"], dtype=torch.float32)
    x = torch.stack((center, center + torch.tensor([p.target["radius"],0,0])))
    assert torch.equal(model(x), torch.zeros(2))


def test_eikonal_analytic_distance_initialization():
    p = problem()
    model = Potential(p)
    with torch.no_grad():
        model.network[-1].weight.zero_()
    x = torch.tensor([[20,20,15],[30,50,10]],dtype=torch.float32)
    residual = eikonal(model, x, torch.zeros(2), 1e-4)
    assert residual.abs().max().item() < 1e-4


def test_sampler_respects_target_obstacle_and_buffer():
    p = problem()
    p.obstacles = [{"lower": [60,40,0], "upper": [70,50,30]}]
    points = sample_points(p, 1000, device="cpu", buffer=2)
    assert len(points) == 1000
    assert p.free_mask(points, buffer=2).all()


def test_single_height_grid_interpolation_and_affine_field():
    grid = Grid(problem(), (20,20,1))
    rho = 1 + 2*grid.points[...,0] + 3*grid.points[...,1]
    points = torch.tensor([[20,30,15],[45,80,15]],dtype=torch.float64)
    expected = 1 + 2*points[:,0] + 3*points[:,1]
    torch.testing.assert_close(grid.interpolate_torch(rho, points), expected)


def test_gridded_wind_interpolation_keeps_coordinate_derivatives(tmp_path):
    p = problem()
    axes = [np.linspace(0,100,3), np.linspace(0,100,4), np.linspace(0,30,2)]
    points = np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1)
    velocity = np.stack((0.02*points[...,0], 0.01*points[...,1], 0.1*points[...,2]),axis=-1)
    path = tmp_path/"wind.npz"
    np.savez(path,x=axes[0],y=axes[1],z=axes[2],velocity=velocity)
    p.config["wind"] = {"type":"gridded","path":str(path)}
    p = Problem(p.config)
    x = torch.tensor([[20.,40.,10.]],requires_grad=True)
    wind = p.wind(x)
    torch.testing.assert_close(wind, torch.tensor([[0.4,0.4,1.]]))
    torch.testing.assert_close(torch.autograd.grad(wind.sum(),x)[0], torch.tensor([[0.02,0.01,0.1]]))
