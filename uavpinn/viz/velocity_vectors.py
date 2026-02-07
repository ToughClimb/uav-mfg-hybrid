"""
Velocity vector field plots (u and v_w).
Copied from scripts/plot_velocity_vectors.py for viz module integration.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..utils import build_from_config
from ..models import PhiModel
from ..solvers import RhoSolver


def plot_velocity_vectors_2d(checkpoint_path: str, z_slice: float = 25.0,
                             resolution: int = 20):
    """
    Generate 2D velocity vector plots at given z height.
    Saves both UAV velocity u and wind velocity v_w.

    Args:
        checkpoint_path: path to checkpoint
        z_slice: z height for slice
        resolution: grid resolution for vectors
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    rho_grid = checkpoint['rho_grid']

    objects = build_from_config(config)
    domain_bounds = objects['domain_bounds']

    barrier_params = config.get('barrier', None)
    phi_model = PhiModel(
        hidden_layers=config['network']['hidden_layers'],
        target_shape=objects['target_shape'],
        domain_bounds=domain_bounds,
        p=config.get('phi_bc', {}).get('power', 1),
        obstacle_shape=objects['obstacle_shape'],
        barrier_params=barrier_params
    )
    phi_model.load_state_dict(checkpoint['phi_model_state'])
    phi_model.eval()

    rho_solver_cfg = config['rho_solver']
    rho_solver = RhoSolver(
        domain_bounds=domain_bounds,
        nx=rho_solver_cfg['grid']['nx'],
        ny=rho_solver_cfg['grid']['ny'],
        nz=rho_solver_cfg['grid']['nz'],
        target_shape=objects['target_shape'],
        obstacle_shape=objects['obstacle_shape'],
        dt=0.01
    )

    x_range = np.linspace(domain_bounds[0][0], domain_bounds[0][1], resolution)
    y_range = np.linspace(domain_bounds[1][0], domain_bounds[1][1], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, z_slice)

    x_grid = np.stack([X, Y, Z], axis=-1)
    x_tensor = torch.tensor(x_grid, dtype=torch.float64).reshape(-1, 3)

    with torch.no_grad():
        rho_flat = rho_solver.interpolate_to_points(rho_grid, x_tensor)
        v_max_flat = objects['fundamental_diagram'].v_max(rho_flat)
        v_w_flat = objects['wind_field'](x_tensor)

    x_grad = x_tensor.clone().detach().requires_grad_(True)
    phi_flat = phi_model.phi_total(x_grad)

    grad_phi = torch.autograd.grad(
        outputs=phi_flat,
        inputs=x_grad,
        grad_outputs=torch.ones_like(phi_flat),
        create_graph=False
    )[0]

    with torch.no_grad():
        eps_reg = config['numerical']['epsilon_reg']
        grad_norm_reg = torch.sqrt(torch.sum(grad_phi**2, dim=1, keepdim=True) + eps_reg**2)
        u_flat = v_w_flat - v_max_flat * (grad_phi / grad_norm_reg)

    u_x = u_flat[:, 0].numpy().reshape(resolution, resolution)
    u_y = u_flat[:, 1].numpy().reshape(resolution, resolution)
    v_w_x = v_w_flat[:, 0].numpy().reshape(resolution, resolution)
    v_w_y = v_w_flat[:, 1].numpy().reshape(resolution, resolution)

    if objects['obstacle_shape'] is not None:
        with torch.no_grad():
            sdf_obs = objects['obstacle_shape'].sdf(x_tensor).numpy().reshape(resolution, resolution)
            obstacle_mask = sdf_obs < 0
            u_x[obstacle_mask] = np.nan
            u_y[obstacle_mask] = np.nan
            v_w_x[obstacle_mask] = np.nan
            v_w_y[obstacle_mask] = np.nan

    target_params = config['target']['params']
    output_dir = Path(checkpoint_path).parent.parent / 'plots'

    fig1, ax1 = plt.subplots(figsize=(9, 8))
    ax1.quiver(X, Y, u_x, u_y, color='blue', alpha=0.8, scale=120, width=0.004)
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title(f'UAV Velocity Field u(x,y,z={z_slice}m)', fontsize=13)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2, linewidth=0.6, color='0.85')

    if z_slice >= target_params['center'][2] - target_params['radius'] and \
       z_slice <= target_params['center'][2] + target_params['radius']:
        circle = plt.Circle(
            (target_params['center'][0], target_params['center'][1]),
            target_params['radius'],
            fill=False, edgecolor='red', linewidth=2, linestyle='--'
        )
        ax1.add_patch(circle)

    plt.tight_layout()
    output_name = output_dir / f'velocity_u_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 8))
    ax2.quiver(X, Y, v_w_x, v_w_y, color='green', alpha=0.8, scale=120, width=0.004)
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)
    ax2.set_title(f'Wind Velocity Field v_w(x,y,z={z_slice}m)', fontsize=13)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2, linewidth=0.6, color='0.85')

    if z_slice >= target_params['center'][2] - target_params['radius'] and \
       z_slice <= target_params['center'][2] + target_params['radius']:
        circle2 = plt.Circle(
            (target_params['center'][0], target_params['center'][1]),
            target_params['radius'],
            fill=False, edgecolor='red', linewidth=2, linestyle='--'
        )
        ax2.add_patch(circle2)

    plt.tight_layout()
    output_name = output_dir / f'velocity_vw_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
