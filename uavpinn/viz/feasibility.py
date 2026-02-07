"""Feasibility/diagnostic plots related to wind dominance and reachability."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..utils import build_from_config
from ..models import PhiModel


def _compute_fields(checkpoint, config, objects, phi_model,
                    z_slice: float = 25.0, resolution: int = 100):
    from ..solvers import RhoSolver

    domain_bounds = objects['domain_bounds']
    rho_grid = checkpoint.get('rho_grid', checkpoint.get('rho_field'))

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
    x_tensor.requires_grad_(True)

    with torch.no_grad():
        if rho_grid is None:
            rho_grid = torch.zeros(
                (rho_solver_cfg['grid']['nz'], rho_solver_cfg['grid']['ny'], rho_solver_cfg['grid']['nx']),
                dtype=torch.float64
            )
        elif not torch.is_tensor(rho_grid):
            rho_grid = torch.tensor(rho_grid, dtype=torch.float64)
        else:
            rho_grid = rho_grid.to(dtype=torch.float64)

    rho_flat = rho_solver.interpolate_to_points(rho_grid, x_tensor)
    v_max_flat = objects['fundamental_diagram'].v_max(rho_flat)
    v_w_flat = objects['wind_field'](x_tensor)
    phi_flat = phi_model.phi_total(x_tensor)

    grad_phi = torch.autograd.grad(
        outputs=phi_flat,
        inputs=x_tensor,
        grad_outputs=torch.ones_like(phi_flat),
        create_graph=False
    )[0]

    with torch.no_grad():
        eps_reg = config['numerical']['epsilon_reg']
        grad_norm_reg = torch.sqrt(torch.sum(grad_phi**2, dim=1, keepdim=True) + eps_reg**2)
        n_hat = grad_phi / grad_norm_reg

        w_proj = torch.sum(v_w_flat * n_hat, dim=1, keepdim=True)
        feasibility = v_max_flat - w_proj

        u_flat = v_w_flat - v_max_flat * n_hat

        u_norm = torch.sqrt(torch.sum(u_flat**2, dim=1, keepdim=True) + 1e-12)
        grad_norm = torch.sqrt(torch.sum(grad_phi**2, dim=1, keepdim=True) + 1e-12)
        cos_sim = -torch.sum(u_flat * grad_phi, dim=1, keepdim=True) / (u_norm * grad_norm)
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)

        feasibility_np = feasibility.numpy().reshape(resolution, resolution)
        angle_deg_np = (torch.arccos(cos_sim) * 180.0 / np.pi).numpy().reshape(resolution, resolution)

    obstacle_mask = None
    if objects['obstacle_shape'] is not None:
        with torch.no_grad():
            sdf_obs = objects['obstacle_shape'].sdf(x_tensor.detach()).numpy().reshape(resolution, resolution)
            obstacle_mask = sdf_obs < 0

    return X, Y, feasibility_np, angle_deg_np, obstacle_mask


def plot_feasibility_indicator_2d(checkpoint, config, objects, phi_model, output_dir: Path,
                                 z_slice: float = 25.0, resolution: int = 100):
    X, Y, feasibility_np, _, obstacle_mask = _compute_fields(
        checkpoint, config, objects, phi_model, z_slice=z_slice, resolution=resolution
    )

    if obstacle_mask is not None:
        feasibility_np = feasibility_np.copy()
        feasibility_np[obstacle_mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    vmin = float(np.nanmin(feasibility_np))
    vmax = float(np.nanmax(feasibility_np))
    lim = max(abs(vmin), abs(vmax), 1e-6)

    im = ax.imshow(
        feasibility_np,
        origin='lower',
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap='coolwarm',
        vmin=-lim,
        vmax=lim,
        aspect='auto'
    )

    plt.colorbar(im, ax=ax, label='v_max - w·∇φ/||∇φ||_ε (m/s)')

    target_params = config['target']['params']
    if z_slice >= target_params['center'][2] - target_params['radius'] and \
       z_slice <= target_params['center'][2] + target_params['radius']:
        circle = plt.Circle(
            (target_params['center'][0], target_params['center'][1]),
            target_params['radius'],
            fill=False, edgecolor='black', linewidth=2, linestyle='--'
        )
        ax.add_patch(circle)

    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(f'Feasibility Indicator at z={z_slice}m', fontsize=14)

    plt.tight_layout()

    output_name = output_dir / f'feasibility_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_u_vs_neggradphi_angle_2d(checkpoint, config, objects, phi_model, output_dir: Path,
                                 z_slice: float = 25.0, resolution: int = 100):
    X, Y, _, angle_deg_np, obstacle_mask = _compute_fields(
        checkpoint, config, objects, phi_model, z_slice=z_slice, resolution=resolution
    )

    if obstacle_mask is not None:
        angle_deg_np = angle_deg_np.copy()
        angle_deg_np[obstacle_mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        angle_deg_np,
        origin='lower',
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap='viridis',
        vmin=0.0,
        vmax=180.0,
        aspect='auto'
    )

    plt.colorbar(im, ax=ax, label='Angle(u, -∇φ) (deg)')

    target_params = config['target']['params']
    if z_slice >= target_params['center'][2] - target_params['radius'] and \
       z_slice <= target_params['center'][2] + target_params['radius']:
        circle = plt.Circle(
            (target_params['center'][0], target_params['center'][1]),
            target_params['radius'],
            fill=False, edgecolor='red', linewidth=2, linestyle='--'
        )
        ax.add_patch(circle)

    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(f'Angle between u and -∇φ at z={z_slice}m', fontsize=14)

    plt.tight_layout()

    output_name = output_dir / f'angle_u_vs_neggradphi_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_feasibility_from_checkpoint(checkpoint_path: str, z_slice: float = 25.0,
                                     resolution: int = 100):
    """
    Convenience entry: generate feasibility heatmap directly from checkpoint path.

    Args:
        checkpoint_path: path to checkpoint_final.pt
        z_slice: z height for slice
        resolution: grid resolution
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    objects = build_from_config(config)

    barrier_params = config.get('barrier', None)
    phi_model = PhiModel(
        hidden_layers=config['network']['hidden_layers'],
        target_shape=objects['target_shape'],
        domain_bounds=objects['domain_bounds'],
        p=config.get('barrier', {}).get('p') or config.get('phi_bc', {}).get('power', 1),
        obstacle_shape=objects['obstacle_shape'],
        barrier_params=barrier_params
    )
    phi_model.load_state_dict(checkpoint['phi_model_state'])
    phi_model.eval()

    output_dir = checkpoint_path.parent.parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_feasibility_indicator_2d(
        checkpoint, config, objects, phi_model,
        output_dir, z_slice=z_slice, resolution=resolution
    )

    return output_dir
