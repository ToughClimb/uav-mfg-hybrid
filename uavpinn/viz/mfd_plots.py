"""
MFD (Macroscopic Fundamental Diagram) visualization.
Strictly follows code spec section 11.3.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_mfd_validation(checkpoint, config, objects, output_dir: Path, 
                        n_samples: int = 2000):
    """
    Generate MFD validation plot with density distribution.
    
    Args:
        checkpoint: loaded checkpoint dict
        config: configuration dict
        objects: objects from build_from_config
        output_dir: output directory
        n_samples: number of samples
    """
    from ..solvers import RhoSolver
    
    rho_grid = checkpoint['rho_grid']
    domain_bounds = objects['domain_bounds']
    
    # Build rho solver for sampling
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
    
    # Sample random points
    x_samples = torch.rand(n_samples, 3, dtype=torch.float64)
    x_samples[:, 0] = x_samples[:, 0] * (domain_bounds[0][1] - domain_bounds[0][0]) + domain_bounds[0][0]
    x_samples[:, 1] = x_samples[:, 1] * (domain_bounds[1][1] - domain_bounds[1][0]) + domain_bounds[1][0]
    x_samples[:, 2] = x_samples[:, 2] * (domain_bounds[2][1] - domain_bounds[2][0]) + domain_bounds[2][0]
    
    # Interpolate rho and compute v_max
    with torch.no_grad():
        rho_samples = rho_solver.interpolate_to_points(rho_grid, x_samples)
        v_max_samples = objects['fundamental_diagram'].v_max(rho_samples)
    
    rho_np = rho_samples.numpy().flatten()
    v_max_np = v_max_samples.numpy().flatten()
    
    # Theoretical curve
    rho_theory = np.linspace(0, config['physics']['rho_max'], 200)
    rho_theory_tensor = torch.tensor(rho_theory, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        v_max_theory = objects['fundamental_diagram'].v_max(rho_theory_tensor).numpy().flatten()
    
    # Create figure (single panel)
    fig, ax_main = plt.subplots(figsize=(12, 7))
    
    # Theoretical curve (behind)
    ax_main.plot(rho_theory, v_max_theory, 'r-', linewidth=2.5, alpha=0.9,
                label='Theoretical MFD (Softmax-Greenshields)', zorder=3)

    # Scatter points (on top, lighter)
    ax_main.scatter(rho_np, v_max_np, s=18, alpha=0.35, c='dodgerblue',
                   edgecolors='none', linewidths=0.0,
                   label='Simulation samples', zorder=4, rasterized=True)
    
    # Reference lines
    ax_main.axvline(config['physics']['rho_jam'], color='orange', linestyle='--', 
                   linewidth=1.5, alpha=0.6, label=f"ρ_jam = {config['physics']['rho_jam']}", zorder=1)
    ax_main.axhline(config['physics']['v_min'], color='purple', linestyle='--',
                   linewidth=1.5, alpha=0.6, label=f"v_min = {config['physics']['v_min']}", zorder=1)
    
    ax_main.set_xlabel('Density ρ (UAVs/m³)', fontsize=13)
    ax_main.set_ylabel('Max Airspeed v_max(ρ) (m/s)', fontsize=13)
    ax_main.set_title('Macroscopic Fundamental Diagram (MFD) Validation',
                     fontsize=14)
    ax_main.set_xlim(0, config['physics']['rho_max'] * 1.1)
    ax_main.set_ylim(config['physics']['v_min'] * 0.95, config['physics']['v_max_0'] * 1.05)
    ax_main.grid(True, alpha=0.3, zorder=0)
    ax_main.legend(
        fontsize=11,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0
    )
    
    plt.tight_layout()
    
    # Save
    output_name = output_dir / 'mfd_validation'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_name}.pdf and .png")
    plt.close(fig)
