"""
Phase diagram plotting for parameter sweeps.
Copied from scripts/plot_phase_diagram.py for viz module integration.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


def extract_metrics_from_run(run_dir: Path) -> dict:
    checkpoint_path = run_dir / 'checkpoints' / 'checkpoint_final.pt'
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        rho_grid = checkpoint['rho_grid']

        metrics = {
            'rho_jam': config['physics']['rho_jam'],
            'q_source': config['physics']['source'].get('p2p_source', {}).get('q_source') or
                        config['physics']['source'].get('homing_q0', 0),
            'max_rho': rho_grid.max().item(),
            'mean_rho': rho_grid.mean().item(),
            'std_rho': rho_grid.std().item(),
            'converged': True
        }

        if torch.isnan(rho_grid).any():
            metrics['converged'] = False
            metrics['max_rho'] = np.nan
            metrics['mean_rho'] = np.nan

        return metrics

    except Exception:
        return None


def plot_phase_diagram_2d(runs_dir: Path, output_dir: Path):
    phase_runs = sorted(runs_dir.glob('*_phase_rj*'))
    if len(phase_runs) == 0:
        return

    data = []
    for run_dir in phase_runs:
        metrics = extract_metrics_from_run(run_dir)
        if metrics:
            data.append(metrics)

    if len(data) == 0:
        return

    rho_jam_values = sorted(set(d['rho_jam'] for d in data))
    q_source_values = sorted(set(d['q_source'] for d in data))

    rho_jam_grid, q_source_grid = np.meshgrid(rho_jam_values, q_source_values)
    mean_rho_grid = np.full_like(rho_jam_grid, np.nan)
    converged_grid = np.zeros_like(rho_jam_grid, dtype=bool)

    for d in data:
        i = q_source_values.index(d['q_source'])
        j = rho_jam_values.index(d['rho_jam'])
        mean_rho_grid[i, j] = d['mean_rho']
        converged_grid[i, j] = d['converged']

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2166ac', '#4393c3', '#92c5de', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap = LinearSegmentedColormap.from_list('congestion', colors, N=100)

    levels = [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    contourf = ax.contourf(rho_jam_grid, q_source_grid, mean_rho_grid,
                           levels=levels, cmap=cmap, extend='both')

    for i, j in zip(*np.where(~converged_grid)):
        ax.plot(rho_jam_grid[i, j], q_source_grid[i, j], 'kx',
                markersize=15, markeredgewidth=3, label='Unstable (NaN)' if i == 0 and j == 0 else '')

    contour = ax.contour(rho_jam_grid, q_source_grid, mean_rho_grid,
                         levels=levels, colors='black', linewidths=0.5, alpha=0.3)
    ax.clabel(contour, inline=True, fontsize=9, fmt='%.3f')

    plt.colorbar(contourf, ax=ax, label='Mean Density ρ (UAVs/m³)')

    ax.set_xlabel('Congestion Threshold ρ_jam (UAVs/m³)', fontsize=13)
    ax.set_ylabel('Source Injection Rate q (UAVs/(m³·s))', fontsize=13)
    ax.set_title('Phase Diagram: Congestion Regimes', fontsize=14)

    ax.text(0.3, 0.0008, 'Free Flow', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if np.nanmax(mean_rho_grid) > 0.05:
        ax.text(0.15, 0.004, 'Congested', fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    output_dir.mkdir(exist_ok=True, parents=True)
    output_name = output_dir / 'phase_diagram_congestion'
    plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    generate_robustness_table(data, output_dir)


def generate_robustness_table(data: list, output_dir: Path):
    rho_jam_values = sorted(set(d['rho_jam'] for d in data))
    q_source_values = sorted(set(d['q_source'] for d in data))

    latex = r"""\begin{table}[htbp]
\centering
\caption{Parameter Robustness Analysis: Mean Density vs. $\rho_{jam}$ and $q$}
\label{tab:robustness}
\begin{tabular}{c|""" + "c" * len(q_source_values) + r"""}
\hline
$\rho_{jam}$ & """ + " & ".join([f"$q={q:.4f}$" for q in q_source_values]) + r""" \\
\hline
"""

    for rho_jam in rho_jam_values:
        row = [f"{rho_jam:.2f}"]
        for q_source in q_source_values:
            match = [d for d in data if d['rho_jam'] == rho_jam and d['q_source'] == q_source]
            if match and match[0]['converged']:
                mean_rho = match[0]['mean_rho']
                row.append(f"{mean_rho:.4f}")
            else:
                row.append("NaN")
        latex += " & ".join(row) + r" \\" + "\n"

    latex += r"""\hline
\end{tabular}
\end{table}
"""

    table_path = output_dir / 'robustness_table.tex'
    with open(table_path, 'w') as f:
        f.write(latex)
