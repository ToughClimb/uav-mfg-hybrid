"""3D visualization functions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def plot_3d_streamlines(checkpoint, config, objects, phi_model, output_dir: Path,
                        n_streamlines: int = 20):
    """3D streamlines of -∇φ."""
    domain_bounds = objects['domain_bounds']

    resolution = 30
    x_range = np.linspace(domain_bounds[0][0], domain_bounds[0][1], resolution)
    y_range = np.linspace(domain_bounds[1][0], domain_bounds[1][1], resolution)
    z_range = np.linspace(domain_bounds[2][0], domain_bounds[2][1], resolution)
    
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    x_grid = np.stack([X, Y, Z], axis=-1)
    x_tensor = torch.tensor(x_grid, dtype=torch.float64).reshape(-1, 3)
    x_tensor.requires_grad_(True)

    phi_flat = phi_model.phi_total(x_tensor)

    grad_phi = torch.autograd.grad(
        outputs=phi_flat,
        inputs=x_tensor,
        grad_outputs=torch.ones_like(phi_flat),
        create_graph=False
    )[0]
    
    with torch.no_grad():
        grad_phi_np = grad_phi.numpy().reshape(resolution, resolution, resolution, 3)
    
    U = -grad_phi_np[..., 0]
    V = -grad_phi_np[..., 1]
    W = -grad_phi_np[..., 2]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    np.random.seed(42)
    seed_points = np.random.rand(n_streamlines, 3)
    seed_points[:, 0] = seed_points[:, 0] * (domain_bounds[0][1] - domain_bounds[0][0]) + domain_bounds[0][0]
    seed_points[:, 1] = seed_points[:, 1] * (domain_bounds[1][1] - domain_bounds[1][0]) + domain_bounds[1][0]
    seed_points[:, 2] = seed_points[:, 2] * (domain_bounds[2][1] - domain_bounds[2][0]) + domain_bounds[2][0]
    
    for seed in seed_points[:min(n_streamlines, len(seed_points))]:
        trajectory = [seed]
        pos = seed.copy()
        
        for step in range(100):
            idx_x = np.clip(int((pos[0] - domain_bounds[0][0]) / (domain_bounds[0][1] - domain_bounds[0][0]) * (resolution-1)), 0, resolution-2)
            idx_y = np.clip(int((pos[1] - domain_bounds[1][0]) / (domain_bounds[1][1] - domain_bounds[1][0]) * (resolution-1)), 0, resolution-2)
            idx_z = np.clip(int((pos[2] - domain_bounds[2][0]) / (domain_bounds[2][1] - domain_bounds[2][0]) * (resolution-1)), 0, resolution-2)
            
            vel = np.array([U[idx_x, idx_y, idx_z], V[idx_x, idx_y, idx_z], W[idx_x, idx_y, idx_z]])
            vel_norm = np.linalg.norm(vel)
            
            if vel_norm < 1e-6:
                break
            
            dt = 0.5
            pos = pos + dt * vel / (vel_norm + 1e-6)
            
            if not (domain_bounds[0][0] <= pos[0] <= domain_bounds[0][1] and
                    domain_bounds[1][0] <= pos[1] <= domain_bounds[1][1] and
                    domain_bounds[2][0] <= pos[2] <= domain_bounds[2][1]):
                break
            
            trajectory.append(pos.copy())
        
        if len(trajectory) > 2:
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   'b-', linewidth=1.5, alpha=0.7)
    
    target_params = config['target']['params']
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = target_params['center'][0] + target_params['radius'] * np.outer(np.cos(u), np.sin(v))
    y_sphere = target_params['center'][1] + target_params['radius'] * np.outer(np.sin(u), np.sin(v))
    z_sphere = target_params['center'][2] + target_params['radius'] * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='red', alpha=0.3)
    
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_zlabel('z (m)', fontsize=11)
    ax.set_title('3D Streamlines of -∇φ', fontsize=14)
    
    plt.tight_layout()
    
    output_name = output_dir / 'streamlines_3d'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_name}.pdf and .png")
    plt.close(fig)


def plot_3d_isosurface(checkpoint, config, objects, phi_model, output_dir: Path,
                       phi_levels=[2, 4, 6, 8], resolution: int = 30):
    """
    Plot 3D isosurfaces of phi (equal-time surfaces).
    
    Args:
        checkpoint: loaded checkpoint dict
        config: configuration dict  
        objects: objects from build_from_config
        phi_model: PhiModel instance
        output_dir: output directory
        phi_levels: list of phi values for isosurfaces
        resolution: grid resolution
    """
    domain_bounds = objects['domain_bounds']

    x_range = np.linspace(domain_bounds[0][0], domain_bounds[0][1], resolution)
    y_range = np.linspace(domain_bounds[1][0], domain_bounds[1][1], resolution)
    z_range = np.linspace(domain_bounds[2][0], domain_bounds[2][1], resolution)
    
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    x_grid = np.stack([X, Y, Z], axis=-1)
    x_tensor = torch.tensor(x_grid, dtype=torch.float64).reshape(-1, 3)

    with torch.no_grad():
        phi_values = phi_model.phi_total(x_tensor).numpy().reshape(resolution, resolution, resolution)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    from skimage import measure
    colors = plt.cm.viridis(np.linspace(0, 1, len(phi_levels)))
    
    for level, color in zip(phi_levels, colors):
        try:
            verts, faces, _, _ = measure.marching_cubes(phi_values, level=level)
            verts[:, 0] = verts[:, 0] / (resolution-1) * (domain_bounds[0][1] - domain_bounds[0][0]) + domain_bounds[0][0]
            verts[:, 1] = verts[:, 1] / (resolution-1) * (domain_bounds[1][1] - domain_bounds[1][0]) + domain_bounds[1][0]
            verts[:, 2] = verts[:, 2] / (resolution-1) * (domain_bounds[2][1] - domain_bounds[2][0]) + domain_bounds[2][0]
            
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                           color=color, alpha=0.4, label=f'φ={level}s')
        except:
            print(f"  Warning: Could not generate isosurface for φ={level}")
    
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_zlabel('z (m)', fontsize=11)
    ax.set_title('3D Phi Isosurfaces', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    output_name = output_dir / 'phi_isosurfaces_3d'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_name}.pdf and .png")
    plt.close(fig)


def plot_3d_vectors(checkpoint, config, objects, phi_model, output_dir: Path,
                    resolution: int = 12):
    """
    Plot 3D vector field: UAV velocity (u) and wind velocity (v_w).
    
    Args:
        checkpoint: loaded checkpoint dict
        config: configuration dict
        objects: objects from build_from_config
        phi_model: PhiModel instance
        output_dir: output directory
        resolution: grid resolution (sparse for clarity)
    """
    from ..solvers import RhoSolver
    
    domain_bounds = objects['domain_bounds']
    rho_grid = checkpoint['rho_grid']

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
    z_range = np.linspace(domain_bounds[2][0], domain_bounds[2][1], resolution)
    
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    x_grid = np.stack([X, Y, Z], axis=-1)
    x_tensor = torch.tensor(x_grid, dtype=torch.float64).reshape(-1, 3)
    x_tensor_grad = x_tensor.clone().detach().requires_grad_(True)

    rho_flat = rho_solver.interpolate_to_points(rho_grid, x_tensor)
    v_max_flat = objects['fundamental_diagram'].v_max(rho_flat)
    v_w_flat = objects['wind_field'](x_tensor)
    phi_flat = phi_model.phi_total(x_tensor_grad)

    grad_phi = torch.autograd.grad(
        outputs=phi_flat,
        inputs=x_tensor_grad,
        grad_outputs=torch.ones_like(phi_flat),
        create_graph=False
    )[0]
    
    with torch.no_grad():
        eps_reg = config['numerical']['epsilon_reg']
        grad_norm_reg = torch.sqrt(torch.sum(grad_phi**2, dim=1, keepdim=True) + eps_reg**2)
        u_flat = v_w_flat - v_max_flat * (grad_phi / grad_norm_reg)

    u_np = u_flat.numpy().reshape(resolution, resolution, resolution, 3)
    v_w_np = v_w_flat.numpy().reshape(resolution, resolution, resolution, 3)

    u_stride = 2
    X_u = X[::u_stride, ::u_stride, ::u_stride]
    Y_u = Y[::u_stride, ::u_stride, ::u_stride]
    Z_u = Z[::u_stride, ::u_stride, ::u_stride]
    u_plot = u_np[::u_stride, ::u_stride, ::u_stride]

    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.quiver(X_u, Y_u, Z_u, u_plot[..., 0], u_plot[..., 1], u_plot[..., 2],
              length=2.5, color='blue', alpha=0.6, arrow_length_ratio=0.25)
    ax1.set_xlabel('x (m)', fontsize=11)
    ax1.set_ylabel('y (m)', fontsize=11)
    ax1.set_zlabel('z (m)', fontsize=11)
    ax1.set_title('UAV Velocity Field u (3D)', fontsize=12)
    
    plt.tight_layout()
    
    output_name = output_dir / 'vectors_u_3d'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_name}.pdf and .png")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.quiver(X, Y, Z, v_w_np[..., 0], v_w_np[..., 1], v_w_np[..., 2],
              length=4, color='green', alpha=0.6, arrow_length_ratio=0.35)
    ax2.set_xlabel('x (m)', fontsize=11)
    ax2.set_ylabel('y (m)', fontsize=11)
    ax2.set_zlabel('z (m)', fontsize=11)
    ax2.set_title('Wind Velocity Field v_w (3D)', fontsize=12)
    
    plt.tight_layout()
    
    output_name = output_dir / 'vectors_vw_3d'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_name}.pdf and .png")
    plt.close(fig2)


def plot_3d_density_volume(checkpoint, config, objects, output_dir: Path,
                           resolution: int = 50, threshold: float = 1e-5):
    """
    Plot 3D volume rendering of density field.
    
    Args:
        checkpoint: loaded checkpoint dict
        config: configuration dict
        objects: objects from build_from_config
        output_dir: output directory
        resolution: grid resolution
        threshold: density threshold for visualization
    """
    from ..solvers import RhoSolver
    
    domain_bounds = objects['domain_bounds']
    rho_grid = checkpoint['rho_grid']

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
    z_range = np.linspace(domain_bounds[2][0], domain_bounds[2][1], resolution)
    
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    x_grid = np.stack([X, Y, Z], axis=-1)
    x_tensor = torch.tensor(x_grid, dtype=torch.float64).reshape(-1, 3)

    with torch.no_grad():
        rho_interp = rho_solver.interpolate_to_points(rho_grid, x_tensor)
    
    if rho_interp.numel() == 0:
        print("  Warning: 3D density volume skipped (empty rho interpolation)")
        return

    rho_volume = rho_interp.numpy().reshape(resolution, resolution, resolution)
    rho_volume = np.nan_to_num(rho_volume, nan=0.0, posinf=0.0, neginf=0.0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    voxels = rho_volume > threshold
    if not np.any(voxels):
        print("  Warning: 3D density volume skipped (no voxels above threshold)")
        return
    rho_normalized = np.clip(rho_volume / config['physics']['rho_max'], 0, 1)

    colors_rgba = plt.cm.viridis(rho_normalized)
    colors_rgba[..., 3] = 0.3
    if colors_rgba.shape[:3] != voxels.shape:
        print("  Warning: 3D density volume skipped (color/voxel shape mismatch)")
        return
    ax.voxels(voxels, facecolors=colors_rgba, edgecolors='none')
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('z', fontsize=11)
    ax.set_title('3D Density Volume Rendering', fontsize=14)
    
    plt.tight_layout()
    
    output_name = output_dir / 'density_volume_3d'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_name}.pdf and .png")
    plt.close(fig)
