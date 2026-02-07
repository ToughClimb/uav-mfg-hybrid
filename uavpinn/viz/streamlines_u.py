"""
Velocity streamline visualization (ground velocity u).
Additional plot that does not modify the original -∇φ streamlines.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import PowerNorm
from pathlib import Path

from ..utils import build_from_config
from ..models import PhiModel
from ..solvers import RhoSolver


def plot_velocity_streamlines_2d(checkpoint_path: str, z_slice: float = 25.0,
                                 resolution: int = 100):
    """
    Generate 2D streamline plot of ground velocity u at given z height.

    Args:
        checkpoint_path: path to checkpoint
        z_slice: z height for slice
        resolution: grid resolution
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    rho_grid = checkpoint.get('rho_grid', checkpoint.get('rho_field'))

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
        phi_values = phi_model.phi_total(x_tensor).numpy().reshape(resolution, resolution)

    viz_cfg = config.get('viz', {})
    phi_vmin_cfg = viz_cfg.get('phi_vmin')
    phi_vmax_cfg = viz_cfg.get('phi_vmax')

    phi_vmin = float(phi_vmin_cfg) if phi_vmin_cfg is not None else 0.0
    phi_vmax = float(phi_vmax_cfg) if phi_vmax_cfg is not None else 8.0
    if phi_vmax <= phi_vmin:
        phi_vmax = phi_vmin + 1.0
    phi_levels = np.linspace(phi_vmin, phi_vmax, 21)
    phi_gamma = viz_cfg.get('phi_gamma', 1.0)
    phi_gamma = float(phi_gamma) if phi_gamma is not None else 1.0
    phi_norm = None
    if phi_gamma and phi_gamma > 0 and phi_gamma != 1.0:
        phi_norm = PowerNorm(gamma=phi_gamma, vmin=phi_vmin, vmax=phi_vmax)
    streamline_buffer = viz_cfg.get('streamline_obstacle_buffer', 0.0)
    streamline_buffer = float(streamline_buffer) if streamline_buffer is not None else 0.0

    grad_phi_x = np.gradient(phi_values, x_range, axis=1)
    grad_phi_y = np.gradient(phi_values, y_range, axis=0)

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
        v_max_flat = objects['fundamental_diagram'].v_max(rho_flat).squeeze(-1)
        v_w_flat = objects['wind_field'](x_tensor)

    eps_reg = config.get('numerical', {}).get('epsilon_reg', 1e-6)
    grad_norm = np.sqrt(grad_phi_x**2 + grad_phi_y**2 + eps_reg**2)
    v_max_np = v_max_flat.numpy().reshape(resolution, resolution)
    v_w_np = v_w_flat.numpy().reshape(resolution, resolution, 3)

    U = v_w_np[..., 0] - v_max_np * (grad_phi_x / grad_norm)
    V = v_w_np[..., 1] - v_max_np * (grad_phi_y / grad_norm)

    obstacle_mask = None
    if objects['obstacle_shape'] is not None:
        with torch.no_grad():
            sdf_obs = objects['obstacle_shape'].sdf(x_tensor).numpy().reshape(resolution, resolution)
        obstacle_mask = sdf_obs < float(streamline_buffer)

    target_mask = None
    if objects.get('target_shape') is not None:
        with torch.no_grad():
            sdf_target = objects['target_shape'].sdf(x_tensor).numpy().reshape(resolution, resolution)
        target_mask = sdf_target <= 0.0

    if obstacle_mask is not None or target_mask is not None:
        combined_mask = np.zeros_like(U, dtype=bool)
        if obstacle_mask is not None:
            combined_mask |= obstacle_mask
        if target_mask is not None:
            combined_mask |= target_mask
        U = np.ma.array(U, mask=combined_mask)
        V = np.ma.array(V, mask=combined_mask)

    # Seed streamlines from source region for P2P scenarios
    start_points = None
    scenario_type = config.get('scenario', {}).get('type', '').lower()
    if scenario_type == 'p2p':
        source_cfg = config.get('physics', {}).get('source', {}).get('p2p_source', {})
        source_center = source_cfg.get('center')
        source_radius = source_cfg.get('radius')
        if source_center is not None and source_radius is not None:
            center = np.array(source_center, dtype=float)
            dz = float(z_slice) - center[2]
            if abs(dz) <= float(source_radius):
                r_xy = float(np.sqrt(max(float(source_radius) ** 2 - dz ** 2, 0.0)))
                if r_xy > 0.0:
                    viz_cfg = config.get('viz', {})
                    n_theta = int(viz_cfg.get('streamline_seed_theta', 8))
                    n_rings = int(viz_cfg.get('streamline_seed_rings', 1))
                    theta_offset = np.random.uniform(0.0, 2.0 * np.pi)
                    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False) + theta_offset
                    if n_rings <= 1:
                        radii = np.array([r_xy])
                    else:
                        radii = np.linspace(r_xy * 0.3, r_xy, n_rings)
                    seeds = []
                    for r in radii:
                        xs = center[0] + r * np.cos(thetas)
                        ys = center[1] + r * np.sin(thetas)
                        seeds.append(np.column_stack([xs, ys]))
                    start_points = np.vstack(seeds) if seeds else None
                else:
                    start_points = np.array([[center[0], center[1]]])

                if start_points is not None and start_points.size > 0:
                    x_min, x_max = domain_bounds[0]
                    y_min, y_max = domain_bounds[1]
                    inside_mask = (
                        (start_points[:, 0] >= x_min) & (start_points[:, 0] <= x_max) &
                        (start_points[:, 1] >= y_min) & (start_points[:, 1] <= y_max)
                    )
                    start_points = start_points[inside_mask]

                if start_points is not None and start_points.size > 0 and objects['obstacle_shape'] is not None:
                    z_vals = np.full((start_points.shape[0], 1), float(z_slice))
                    seed_xyz = np.hstack([start_points, z_vals])
                    with torch.no_grad():
                        sdf_seed = objects['obstacle_shape'].sdf(
                            torch.tensor(seed_xyz, dtype=torch.float64)
                        ).numpy().reshape(-1)
                    start_points = start_points[sdf_seed >= float(streamline_buffer)]

                if start_points is not None and start_points.size == 0:
                    start_points = None

    fig, ax = plt.subplots(figsize=(10, 9))
    contour_kwargs = dict(levels=phi_levels, cmap='viridis', alpha=0.6)
    if phi_norm is None:
        contour_kwargs.update(vmin=phi_vmin, vmax=phi_vmax)
    else:
        contour_kwargs.update(norm=phi_norm)

    contour = ax.contourf(X, Y, phi_values, **contour_kwargs)
    cbar = plt.colorbar(contour, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('φ (s)', fontsize=13)

    target_params = config['target']['params']

    stream_kwargs = dict(
        color='white',
        linewidth=1.8,
        arrowsize=1.5,
        arrowstyle='->',
        minlength=0.5,
        broken_streamlines=False
    )
    if start_points is None:
        stream_kwargs['density'] = 1.2
    else:
        stream_kwargs['start_points'] = start_points

    stream = ax.streamplot(X, Y, U, V, **stream_kwargs)
    stroke_width = stream_kwargs['linewidth'] + 1.2
    line_effects = [pe.Stroke(linewidth=stroke_width, foreground='black', alpha=0.6), pe.Normal()]
    stream.lines.set_path_effects(line_effects)
    if hasattr(stream, 'arrows') and stream.arrows is not None:
        stream.arrows.set_path_effects(line_effects)

    if z_slice >= target_params['center'][2] - target_params['radius'] and \
       z_slice <= target_params['center'][2] + target_params['radius']:
        circle = plt.Circle(
            (target_params['center'][0], target_params['center'][1]),
            target_params['radius'],
            fill=False, edgecolor='red', linewidth=2, linestyle='--',
            label='Target (sink)'
        )
        ax.add_patch(circle)

    # Source circle for P2P scenarios
    if scenario_type == 'p2p':
        source_cfg = config.get('physics', {}).get('source', {}).get('p2p_source', {})
        source_center = source_cfg.get('center')
        source_radius = source_cfg.get('radius')
        if source_center is not None and source_radius is not None:
            if z_slice >= source_center[2] - source_radius and z_slice <= source_center[2] + source_radius:
                source_circle = plt.Circle(
                    (source_center[0], source_center[1]),
                    source_radius,
                    fill=False, edgecolor='green', linewidth=2, linestyle='-',
                    label='Source'
                )
                ax.add_patch(source_circle)

    if 'obstacles' in config and len(config['obstacles']) > 0:
        for obs in config['obstacles']:
            if obs['type'] == 'aabb':
                params = obs['params']
                if z_slice >= params['min'][2] and z_slice <= params['max'][2]:
                    rect = plt.Rectangle(
                        (params['min'][0], params['min'][1]),
                        params['max'][0] - params['min'][0],
                        params['max'][1] - params['min'][1],
                        fill=True, facecolor='gray', alpha=0.8,
                        edgecolor='black', linewidth=2, label='Obstacle'
                    )
                    ax.add_patch(rect)

    ax.set_xlabel('x (m)', fontsize=14)
    ax.set_ylabel('y (m)', fontsize=14)
    ax.set_title(f'u at z={z_slice}m', fontsize=15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=0.6, color='0.85')
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(
            fontsize=12,
            loc='lower center',
            bbox_to_anchor=(0.75, 1.06),
            bbox_transform=cbar.ax.transAxes,
            borderaxespad=0.0
        )

    plt.tight_layout()

    output_dir = Path(checkpoint_path).parent.parent / 'plots'
    output_name = output_dir / f'streamlines_u_z{int(z_slice)}'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
