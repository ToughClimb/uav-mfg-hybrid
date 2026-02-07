"""
Enhanced MFD visualization with multiple display options.
Copied from scripts/plot_mfd_enhanced.py for viz module integration.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from ..utils import build_from_config
from ..solvers import RhoSolver


def plot_mfd_enhanced(checkpoint_path: str, n_samples: int = 2000, style: str = 'all'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    rho_grid = checkpoint['rho_grid']

    objects = build_from_config(config)
    domain_bounds = objects['domain_bounds']

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

    x_samples = torch.rand(n_samples, 3, dtype=torch.float64)
    x_samples[:, 0] = x_samples[:, 0] * (domain_bounds[0][1] - domain_bounds[0][0]) + domain_bounds[0][0]
    x_samples[:, 1] = x_samples[:, 1] * (domain_bounds[1][1] - domain_bounds[1][0]) + domain_bounds[1][0]
    x_samples[:, 2] = x_samples[:, 2] * (domain_bounds[2][1] - domain_bounds[2][0]) + domain_bounds[2][0]

    with torch.no_grad():
        rho_samples = rho_solver.interpolate_to_points(rho_grid, x_samples)
        v_max_samples = objects['fundamental_diagram'].v_max(rho_samples)

    rho_np = rho_samples.numpy().flatten()
    v_max_np = v_max_samples.numpy().flatten()

    rho_theory = np.linspace(0, config['physics']['rho_max'], 200)
    rho_theory_tensor = torch.tensor(rho_theory, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        v_max_theory = objects['fundamental_diagram'].v_max(rho_theory_tensor).numpy().flatten()

    output_dir = Path(checkpoint_path).parent.parent / 'plots'

    if style in ['standard', 'all']:
        _plot_standard(rho_np, v_max_np, rho_theory, v_max_theory, config, output_dir)

    if style in ['hexbin', 'all']:
        _plot_hexbin(rho_np, v_max_np, rho_theory, v_max_theory, config, output_dir)

    if style in ['contour', 'all']:
        _plot_contour(rho_np, v_max_np, rho_theory, v_max_theory, config, output_dir)

    if style == 'all':
        _plot_combined(rho_np, v_max_np, rho_theory, v_max_theory, config, output_dir)


def _plot_standard(rho_np, v_max_np, rho_theory, v_max_theory, config, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(rho_theory, v_max_theory, 'r-', linewidth=2, alpha=0.7,
            label='Theoretical MFD', zorder=2)
    ax.scatter(rho_np, v_max_np, s=40, alpha=0.6, c='dodgerblue',
               edgecolors='navy', linewidths=0.8,
               label='Simulation samples', zorder=3)
    ax.axvline(config['physics']['rho_jam'], color='orange', linestyle='--',
               linewidth=1.5, alpha=0.6, label=f"ρ_jam = {config['physics']['rho_jam']}", zorder=1)
    ax.axhline(config['physics']['v_min'], color='purple', linestyle='--',
               linewidth=1.5, alpha=0.6, label=f"v_min = {config['physics']['v_min']}", zorder=1)
    ax.set_xlabel('Density ρ (UAVs/m³)', fontsize=13)
    ax.set_ylabel('Max Airspeed v_max(ρ) (m/s)', fontsize=13)
    ax.set_title('MFD Validation: Scatter Plot', fontsize=14)
    ax.set_xlim(0, config['physics']['rho_max'] * 1.1)
    ax.set_ylim(config['physics']['v_min'] * 0.95, config['physics']['v_max_0'] * 1.05)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    output_name = output_dir / 'mfd_standard'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_hexbin(rho_np, v_max_np, rho_theory, v_max_theory, config, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    hexbin = ax.hexbin(rho_np, v_max_np, gridsize=30, cmap='Blues',
                       mincnt=1, alpha=0.8, zorder=2)
    ax.plot(rho_theory, v_max_theory, 'r-', linewidth=3,
            label='Theoretical MFD', zorder=3)
    ax.axvline(config['physics']['rho_jam'], color='orange', linestyle='--',
               linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(config['physics']['v_min'], color='purple', linestyle='--',
               linewidth=1.5, alpha=0.7, zorder=1)
    ax.set_xlabel('Density ρ (UAVs/m³)', fontsize=13)
    ax.set_ylabel('Max Airspeed v_max(ρ) (m/s)', fontsize=13)
    ax.set_title('MFD Validation: Density Hexbin', fontsize=14)
    ax.set_xlim(0, config['physics']['rho_max'] * 1.1)
    ax.set_ylim(config['physics']['v_min'] * 0.95, config['physics']['v_max_0'] * 1.05)
    plt.colorbar(hexbin, ax=ax, label='Sample Count')
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    output_name = output_dir / 'mfd_hexbin'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_contour(rho_np, v_max_np, rho_theory, v_max_theory, config, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    H, xedges, yedges = np.histogram2d(rho_np, v_max_np, bins=40)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(H.T, origin='lower', extent=extent, aspect='auto',
                   cmap='Blues', alpha=0.8, zorder=2)
    ax.plot(rho_theory, v_max_theory, 'r-', linewidth=3,
            label='Theoretical MFD', zorder=3)
    ax.axvline(config['physics']['rho_jam'], color='orange', linestyle='--',
               linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(config['physics']['v_min'], color='purple', linestyle='--',
               linewidth=1.5, alpha=0.7, zorder=1)
    ax.set_xlabel('Density ρ (UAVs/m³)', fontsize=13)
    ax.set_ylabel('Max Airspeed v_max(ρ) (m/s)', fontsize=13)
    ax.set_title('MFD Validation: 2D Histogram', fontsize=14)
    ax.set_xlim(0, config['physics']['rho_max'] * 1.1)
    ax.set_ylim(config['physics']['v_min'] * 0.95, config['physics']['v_max_0'] * 1.05)
    plt.colorbar(im, ax=ax, label='Sample Density')
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    output_name = output_dir / 'mfd_contour'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_combined(rho_np, v_max_np, rho_theory, v_max_theory, config, output_dir):
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rho_theory, v_max_theory, 'r-', linewidth=2, alpha=0.7, zorder=2)
    ax1.scatter(rho_np, v_max_np, s=25, alpha=0.5, c='dodgerblue',
                edgecolors='navy', linewidths=0.5, zorder=3)
    ax1.set_xlabel('Density ρ (UAVs/m³)', fontsize=11)
    ax1.set_ylabel('v_max(ρ) (m/s)', fontsize=11)
    ax1.set_title('(a) Scatter Plot', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, config['physics']['rho_max'] * 1.1)
    ax1.set_ylim(config['physics']['v_min'] * 0.95, config['physics']['v_max_0'] * 1.05)

    ax2 = fig.add_subplot(gs[0, 1])
    hexbin = ax2.hexbin(rho_np, v_max_np, gridsize=25, cmap='Blues', mincnt=1, alpha=0.8)
    ax2.plot(rho_theory, v_max_theory, 'r-', linewidth=2.5, zorder=3)
    ax2.set_xlabel('Density ρ (UAVs/m³)', fontsize=11)
    ax2.set_ylabel('v_max(ρ) (m/s)', fontsize=11)
    ax2.set_title('(b) Hexbin Density', fontsize=12)
    ax2.set_xlim(0, config['physics']['rho_max'] * 1.1)
    ax2.set_ylim(config['physics']['v_min'] * 0.95, config['physics']['v_max_0'] * 1.05)
    plt.colorbar(hexbin, ax=ax2, label='Count')

    ax3 = fig.add_subplot(gs[0, 2])
    H, xedges, yedges = np.histogram2d(rho_np, v_max_np, bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax3.imshow(H.T, origin='lower', extent=extent, aspect='auto', cmap='Blues', alpha=0.8)
    ax3.plot(rho_theory, v_max_theory, 'r-', linewidth=2.5, zorder=3)
    ax3.set_xlabel('Density ρ (UAVs/m³)', fontsize=11)
    ax3.set_ylabel('v_max(ρ) (m/s)', fontsize=11)
    ax3.set_title('(c) 2D Histogram', fontsize=12)
    ax3.set_xlim(0, config['physics']['rho_max'] * 1.1)
    ax3.set_ylim(config['physics']['v_min'] * 0.95, config['physics']['v_max_0'] * 1.05)
    plt.colorbar(im, ax=ax3, label='Density')

    plt.suptitle('MFD Validation: Multiple Views', fontsize=14, y=1.02)

    output_name = output_dir / 'mfd_combined'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
