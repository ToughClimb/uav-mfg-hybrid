import numpy as np
import pytest
import torch
import yaml

from uav_mfg.problem import Problem
from uav_mfg.transport import Grid,Transport


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA not available")
def test_cuda_graph_matches_cpu_and_audits_each_step():
    with open("configs/reconstruction.yaml") as stream:
        config=yaml.safe_load(stream)
    config.update(domain=[[0,6],[0,1],[0,1]],target={"type":"sphere","center":[5.5,.5,.5],"radius":.4},
                  source={"type":"homing","rate":.01},obstacles=[])
    grid=Grid(Problem(config),(6,1,1));velocity=np.zeros(grid.shape+(3,));velocity[...,0]=1
    operator=Transport(grid,velocity);initial=np.zeros(grid.shape)
    cpu,a=operator.solve(initial,max_steps=1000,change_tolerance=1e-9,residual_tolerance=1e-8)
    gpu,b=operator.solve(initial,max_steps=1000,change_tolerance=1e-9,residual_tolerance=1e-8,backend="cuda",graph_steps=25)
    np.testing.assert_allclose(gpu,cpu,atol=1e-10)
    assert b["converged"] and b["mass_audit_scope"]=="every_step"
    assert b["max_step_mass_budget_error"]<1e-13
