"""Conservation, absorption, and positivity checks independent of training."""

import copy

import numpy as np
import pytest
import yaml

from uav_mfg.problem import Problem
from uav_mfg.transport import Grid, Transport


@pytest.fixture
def channel():
    with open("configs/reconstruction.yaml") as stream:
        config = yaml.safe_load(stream)
    config.update(domain=[[0, 6], [0, 1], [0, 1]], target={"type": "sphere", "center": [5.5, 0.5, 0.5], "radius": 0.4},
                  source={"type": "homing", "rate": 0.01}, obstacles=[])
    return Grid(Problem(config), (6, 1, 1))


def test_variable_velocity_shared_face_divergence(channel):
    velocity = np.zeros(channel.shape + (3,))
    velocity[:, 0, 0, 0] = [1, 2, 3, 2, -1, 0]
    transport = Transport(channel, velocity)
    rho = np.arange(1, 7, dtype=float).reshape(channel.shape)
    rho[~channel.active] = 0
    divergence = sum(np.diff(flux, axis=d) / channel.spacing[d] for d, flux in enumerate(transport.fluxes(rho)))
    np.testing.assert_allclose(divergence[channel.active], transport.operator @ rho[channel.active], atol=1e-14)
    integrated = divergence[channel.active].sum() * channel.cell_volume
    assert integrated == pytest.approx(transport.audit(rho)["target_absorption"], abs=1e-13)
    assert transport.audit(rho)["outer_leakage"] == 0


def test_analytic_steady_channel(channel):
    velocity = np.zeros(channel.shape + (3,))
    velocity[..., 0] = 1
    transport = Transport(channel, velocity)
    rho, diagnostics = transport.solve(np.zeros(channel.shape), max_steps=1000, change_tolerance=1e-9, residual_tolerance=1e-8)
    assert diagnostics["converged"]
    np.testing.assert_allclose(rho[:,0,0], [0.01,0.02,0.03,0.04,0.05,0], atol=1e-9)
    assert diagnostics["absorption_over_source"] == pytest.approx(1, abs=1e-8)
    assert diagnostics["max_step_mass_budget_error"] < 1e-14
    assert rho.min() >= 0


def test_closed_cell_mass_increase_not_false_steady(channel):
    transport = Transport(channel, np.zeros(channel.shape + (3,)))
    rho, diagnostics = transport.solve(np.zeros(channel.shape), max_steps=50)
    assert not diagnostics["converged"]
    assert diagnostics["target_absorption"] == 0
    assert rho[channel.active].sum() == pytest.approx(0.1 * 50 * channel.source.sum())
    assert diagnostics["max_step_mass_budget_error"] < 1e-14


def test_obstacle_interfaces_carry_no_flux(channel):
    channel.obstacle[2,0,0] = True
    channel.active[2,0,0] = False
    channel.source[2,0,0] = 0
    velocity = np.full(channel.shape + (3,), 3.0)
    transport = Transport(channel, velocity)
    rho = np.ones(channel.shape)
    rho[~channel.active] = 0
    audit = transport.audit(rho)
    assert audit["outer_leakage"] == 0
    assert audit["obstacle_leakage"] == 0
    flux = transport.fluxes(rho)[0]
    assert flux[2,0,0] == flux[3,0,0] == 0
