"""
Diagnostic visualizations: diagonal profiles, residuals.
Strictly follows code spec section 11.2.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from pathlib import Path


def plot_diagonal_profile(checkpoint, config, objects, phi_model, output_dir: Path,
                          z_slice: float = 5.0, n_points: int = 200):
    """
    Plot diagonal profile: phi, rho, and velocity magnitude along diagonal.
    
    Args:
        checkpoint: loaded checkpoint dict
        config: configuration dict
        objects: objects from build_from_config
        phi_model: PhiModel instance
        output_dir: output directory
        z_slice: z height for diagonal
        n_points: number of points along diagonal
    """
    from ..solvers import RhoSolver
    
    domain_bounds = objects['domain_bounds']
    rho_grid = checkpoint['rho_grid']
    
    # Build rho solver
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
    
    # Diagonal line from (xmin, ymin) to (xmax, ymax)
    t = np.linspace(0, 1, n_points)
    x_diag = domain_bounds[0][0] + t * (domain_bounds[0][1] - domain_bounds[0][0])
    y_diag = domain_bounds[1][0] + t * (domain_bounds[1][1] - domain_bounds[1][0])
    z_diag = np.full_like(x_diag, z_slice)
    
    x_tensor = torch.tensor(np.column_stack([x_diag, y_diag, z_diag]), dtype=torch.float64)
    
    # Evaluate fields
    with torch.no_grad():
        phi_diag = phi_model.phi_total(x_tensor).numpy().flatten()
        rho_diag = rho_solver.interpolate_to_points(rho_grid, x_tensor).numpy().flatten()
        v_w_diag = objects['wind_field'](x_tensor).numpy()
        v_w_norm = np.linalg.norm(v_w_diag, axis=1)
    
    # Distance along diagonal
    distance = t * np.sqrt((domain_bounds[0][1] - domain_bounds[0][0])**2 + 
                          (domain_bounds[1][1] - domain_bounds[1][0])**2)
    
    # Plot phi profile separately
    fig1, ax_phi = plt.subplots(figsize=(10, 5))
    
    ax_phi.plot(distance, phi_diag, 'b-', linewidth=2)
    ax_phi.set_xlabel('Distance along diagonal (m)', fontsize=12)
    ax_phi.set_ylabel('φ (s)', fontsize=12)
    ax_phi.set_title(f'φ Diagonal Profile at z={z_slice}m', fontsize=13)
    ax_phi.grid(True, alpha=0.2, linewidth=0.6, color='0.85')
    plt.tight_layout()
    output_name = output_dir / f'diagonal_phi_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Rho profile
    fig2, ax_rho = plt.subplots(figsize=(10, 5))
    ax_rho.plot(distance, rho_diag, 'g-', linewidth=2)
    ax_rho.set_xlabel('Distance along diagonal (m)', fontsize=12)
    ax_rho.set_ylabel('ρ (UAVs/m³)', fontsize=12)
    ax_rho.set_title(f'ρ Diagonal Profile at z={z_slice}m', fontsize=13)
    ax_rho.grid(True, alpha=0.2, linewidth=0.6, color='0.85')
    plt.tight_layout()
    output_name = output_dir / f'diagonal_rho_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Wind magnitude profile
    fig3, ax_wind = plt.subplots(figsize=(10, 5))
    ax_wind.plot(distance, v_w_norm, 'r-', linewidth=2)
    ax_wind.set_xlabel('Distance along diagonal (m)', fontsize=12)
    ax_wind.set_ylabel('||v_w|| (m/s)', fontsize=12)
    ax_wind.set_title(f'Wind Magnitude Diagonal Profile at z={z_slice}m', fontsize=13)
    ax_wind.grid(True, alpha=0.2, linewidth=0.6, color='0.85')
    plt.tight_layout()
    output_name = output_dir / f'diagonal_wind_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"  Saved: diagonal_phi/rho/wind_z{int(z_slice)}.pdf and .png (3 files)")


def plot_diagonal_plane_slice(
    checkpoint,
    config,
    objects,
    phi_model,
    output_dir: Path,
    resolution_s: int = 200,
    resolution_z: int = 120
):
    """
    Plot diagonal plane (x=y) slice as 2D contour maps.

    Args:
        checkpoint: loaded checkpoint dict
        config: configuration dict
        objects: objects from build_from_config
        phi_model: PhiModel instance
        output_dir: output directory
        resolution_s: resolution along diagonal coordinate
        resolution_z: resolution along height (z)
    """
    from ..solvers import RhoSolver

    domain_bounds = objects['domain_bounds']
    rho_grid = checkpoint['rho_grid']

    s_min = max(domain_bounds[0][0], domain_bounds[1][0])
    s_max = min(domain_bounds[0][1], domain_bounds[1][1])
    if s_min >= s_max:
        raise ValueError("Diagonal plane x=y does not intersect domain bounds")

    s_vals = np.linspace(s_min, s_max, resolution_s)
    z_vals = np.linspace(domain_bounds[2][0], domain_bounds[2][1], resolution_z)
    S, Z = np.meshgrid(s_vals, z_vals)
    X = S
    Y = S

    x_grid = np.stack([X, Y, Z], axis=-1)
    x_tensor = torch.tensor(x_grid, dtype=torch.float64).reshape(-1, 3)

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

    with torch.no_grad():
        phi_plane = phi_model.phi_total(x_tensor).numpy().reshape(resolution_z, resolution_s)
        rho_plane = rho_solver.interpolate_to_points(rho_grid, x_tensor).numpy().reshape(
            resolution_z, resolution_s
        )
        wind_plane = objects['wind_field'](x_tensor).numpy().reshape(resolution_z, resolution_s, 3)
        wind_mag = np.linalg.norm(wind_plane, axis=-1)

    def _plot_plane(
        field,
        cmap,
        label,
        title,
        filename,
        add_contour=False,
        levels=20,
        vmin=None,
        vmax=None,
        norm=None
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        contour_kwargs = dict(levels=levels, cmap=cmap)
        if norm is None:
            contour_kwargs.update(vmin=vmin, vmax=vmax)
        else:
            contour_kwargs.update(norm=norm)

        im = ax.contourf(S, Z, field, **contour_kwargs)
        if add_contour:
            ax.contour(S, Z, field, levels=10, colors='white', alpha=0.35, linewidths=0.5)
        ax.set_xlabel('Diagonal coordinate s (x=y) (m)', fontsize=12)
        ax.set_ylabel('z (m)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.2, linewidth=0.6, color='0.85')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label, fontsize=12)
        plt.tight_layout()
        output_name = output_dir / filename
        plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    viz_cfg = config.get('viz', {})
    phi_vmin_cfg = viz_cfg.get('phi_vmin')
    phi_vmax_cfg = viz_cfg.get('phi_vmax')

    phi_min = np.nanmin(phi_plane)
    phi_max = np.nanmax(phi_plane)
    phi_vmin = (
        float(phi_vmin_cfg)
        if phi_vmin_cfg is not None
        else min(0.0, float(phi_min)) if np.isfinite(phi_min) else 0.0
    )
    phi_vmax = (
        float(phi_vmax_cfg)
        if phi_vmax_cfg is not None
        else float(phi_max) if np.isfinite(phi_max) else 1.0
    )
    if phi_vmax <= phi_vmin:
        phi_vmax = phi_vmin + 1.0
    phi_levels = np.linspace(phi_vmin, phi_vmax, 21)
    phi_gamma = viz_cfg.get('phi_gamma', 1.0)
    phi_gamma = float(phi_gamma) if phi_gamma is not None else 1.0
    phi_norm = None
    if phi_gamma and phi_gamma > 0 and phi_gamma != 1.0:
        phi_norm = PowerNorm(gamma=phi_gamma, vmin=phi_vmin, vmax=phi_vmax)

    _plot_plane(
        phi_plane,
        'viridis',
        'φ (s)',
        'φ Diagonal Plane Slice (x=y)',
        'diagonal_plane_phi',
        add_contour=True,
        levels=phi_levels,
        vmin=phi_vmin,
        vmax=phi_vmax,
        norm=phi_norm
    )
    _plot_plane(
        rho_plane,
        'viridis',
        'ρ (UAVs/m³)',
        'ρ Diagonal Plane Slice (x=y)',
        'diagonal_plane_rho'
    )
    _plot_plane(
        wind_mag,
        'plasma',
        '||v_w|| (m/s)',
        'Wind Magnitude Diagonal Plane Slice (x=y)',
        'diagonal_plane_wind'
    )

    print("  Saved: diagonal_plane_phi/rho/wind.pdf and .png (3 files)")


def plot_residual_heatmap(checkpoint, config, objects, phi_model, output_dir: Path,
                          z_slice: float = 25.0, resolution: int = 100):
    """
    Plot Eikonal residual heatmap at given z height.
    
    Args:
        checkpoint: loaded checkpoint dict
        config: configuration dict
        objects: objects from build_from_config
        phi_model: PhiModel instance
        output_dir: output directory
        z_slice: z height
        resolution: grid resolution
    """
    from ..solvers import RhoSolver
    
    domain_bounds = objects['domain_bounds']
    rho_grid = checkpoint['rho_grid']
    
    # Build rho solver
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
    
    # Create grid
    x_range = np.linspace(domain_bounds[0][0], domain_bounds[0][1], resolution)
    y_range = np.linspace(domain_bounds[1][0], domain_bounds[1][1], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, z_slice)
    
    x_grid = np.stack([X, Y, Z], axis=-1)
    x_tensor = torch.tensor(x_grid, dtype=torch.float64).reshape(-1, 3)
    x_tensor.requires_grad_(True)
    
    # Evaluate fields
    rho_flat = rho_solver.interpolate_to_points(rho_grid, x_tensor)
    v_max_flat = objects['fundamental_diagram'].v_max(rho_flat)
    v_w_flat = objects['wind_field'](x_tensor)
    phi_flat = phi_model.phi_total(x_tensor)
    
    # Compute gradient
    grad_phi = torch.autograd.grad(
        outputs=phi_flat,
        inputs=x_tensor,
        grad_outputs=torch.ones_like(phi_flat),
        create_graph=False
    )[0]
    
    # Compute residual
    with torch.no_grad():
        eps_reg = config['numerical']['epsilon_reg']
        grad_norm_reg = torch.sqrt(torch.sum(grad_phi**2, dim=1, keepdim=True) + eps_reg**2)
        wind_term = torch.sum(v_w_flat * grad_phi, dim=1, keepdim=True)
        residual = v_max_flat * grad_norm_reg - wind_term - 1.0
        residual_abs = torch.abs(residual).numpy().reshape(resolution, resolution)
        residual_log10 = np.log10(residual_abs + 1e-12)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(residual_log10, origin='lower', extent=[
        domain_bounds[0][0], domain_bounds[0][1],
        domain_bounds[1][0], domain_bounds[1][1]
    ], cmap='hot', aspect='auto')
    
    plt.colorbar(im, ax=ax, label='log10(|Eikonal Residual|)')
    
    # Target
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
    ax.set_title(f'Eikonal Residual at z={z_slice}m', fontsize=14)
    
    plt.tight_layout()
    
    output_name = output_dir / f'residual_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_name}.pdf and .png")
    plt.close(fig)
