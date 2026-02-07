"""
Training curve visualization.
Strictly follows code spec section 11.2.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _prepare_epoch_axis(df: pd.DataFrame):
    if 'outer_iter' in df.columns and 'phi_epoch' in df.columns:
        phi_epoch = pd.to_numeric(df['phi_epoch'], errors='coerce')
        outer_iter = pd.to_numeric(df['outer_iter'], errors='coerce')
        numeric_mask = phi_epoch.notna() & outer_iter.notna()
        df_numeric = df.loc[numeric_mask].copy()
        if not df_numeric.empty:
            df_numeric['phi_epoch'] = phi_epoch[numeric_mask]
            df_numeric['outer_iter'] = outer_iter[numeric_mask]

            diffs = df_numeric['phi_epoch'].diff()
            positive_diffs = diffs[diffs > 0]
            epoch_step = positive_diffs.median()
            if pd.isna(epoch_step) or epoch_step <= 0:
                epoch_step = 1.0

            max_per_outer = df_numeric.groupby('outer_iter')['phi_epoch'].max()
            offsets = {}
            offset = 0.0
            for outer in sorted(max_per_outer.index):
                offsets[outer] = offset
                offset += max_per_outer[outer] + epoch_step

            df_numeric['epoch_plot'] = df_numeric['phi_epoch'] + df_numeric['outer_iter'].map(offsets)

            summary_mask = phi_epoch.isna() & outer_iter.notna()
            df_summary = df.loc[summary_mask].copy()
            if not df_summary.empty:
                df_summary['outer_iter'] = outer_iter[summary_mask]
                df_summary['phi_epoch'] = df_summary['outer_iter'].map(max_per_outer)
                df_summary = df_summary[df_summary['phi_epoch'].notna()].copy()
                df_summary['phi_epoch'] = df_summary['phi_epoch'] + epoch_step
                df_summary['epoch_plot'] = df_summary['phi_epoch'] + df_summary['outer_iter'].map(offsets)
                df_plot = pd.concat([df_numeric, df_summary], ignore_index=True)
            else:
                df_plot = df_numeric

            df_plot = df_plot.sort_values('epoch_plot')
            return df_plot, 'epoch_plot'

    df_plot = df.copy()
    if 'epoch' not in df_plot.columns:
        df_plot['epoch'] = range(len(df_plot))
    epoch_numeric = pd.to_numeric(df_plot['epoch'], errors='coerce')
    epoch_mask = epoch_numeric.notna()
    df_plot = df_plot.loc[epoch_mask].copy()
    df_plot['epoch_plot'] = epoch_numeric[epoch_mask]
    df_plot = df_plot.sort_values('epoch_plot')
    return df_plot, 'epoch_plot'


def plot_training_curves(run_dir: Path, output_dir: Path = None):
    """
    Plot training curves from metrics.csv.
    
    Args:
        run_dir: run directory containing metrics.csv
        output_dir: output directory (default: run_dir/plots)
    """
    if output_dir is None:
        output_dir = run_dir / 'plots'
    
    output_dir.mkdir(exist_ok=True)
    
    metrics_file = run_dir / 'metrics.csv'
    if not metrics_file.exists():
        print(f"  Warning: metrics.csv not found in {run_dir}")
        return
    
    # Load metrics
    df = pd.read_csv(metrics_file)
    
    df_plot, epoch_col = _prepare_epoch_axis(df)
    epoch_label = 'Epoch (global)' if 'outer_iter' in df.columns else 'Epoch'
    
    # Determine loss and residual column names
    loss_candidates = ['loss', 'loss_eik', 'phi_loss']
    residual_candidates = ['residual', 'mean_residual', 'phi_residual']
    loss_col = next((col for col in loss_candidates if col in df_plot.columns), None)
    residual_col = next((col for col in residual_candidates if col in df_plot.columns), None)
    label_map = {
        'loss': 'Loss',
        'loss_eik': 'Eikonal loss (L_eik)',
        'phi_loss': 'Phi loss (L_phi)',
        'residual': 'Residual',
        'mean_residual': 'Mean eikonal residual',
        'phi_residual': 'Phi residual'
    }
    
    # Plot loss separately
    fig1, ax_loss = plt.subplots(figsize=(10, 6))
    
    # Plot loss
    if loss_col is not None and df_plot[loss_col].notna().any():
        ax_loss.plot(df_plot[epoch_col], df_plot[loss_col], 'b-', linewidth=2)
        ax_loss.set_xlabel(epoch_label, fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_loss.set_yscale('log')
        ax_loss.set_title('Training Loss', fontsize=14)
        ax_loss.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_name = output_dir / 'training_loss'
        plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_name}.pdf and .png")
        plt.close(fig1)

    # Plot loss + residual in the same figure
    if (
        loss_col is not None and residual_col is not None and
        df_plot[loss_col].notna().any() and df_plot[residual_col].notna().any()
    ):
        fig_combo, ax_combo = plt.subplots(figsize=(10, 6))
        loss_label = label_map.get(loss_col, loss_col)
        residual_label = label_map.get(residual_col, residual_col)
        ax_combo.plot(df_plot[epoch_col], df_plot[loss_col], 'b-', linewidth=2, label=loss_label)
        ax_combo.plot(df_plot[epoch_col], df_plot[residual_col], 'r--', linewidth=2, label=residual_label)
        ax_combo.set_xlabel(epoch_label, fontsize=12)
        ax_combo.set_ylabel('Loss / Residual', fontsize=12)
        ax_combo.set_yscale('log')
        ax_combo.set_title('Training Loss & Residual', fontsize=14)
        ax_combo.grid(True, alpha=0.3)
        ax_combo.legend(fontsize=11)

        plt.tight_layout()

        output_name = output_dir / 'training_loss_residual'
        plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_name}.pdf and .png")
        plt.close(fig_combo)

    # Plot residual separately
    if residual_col is not None and df_plot[residual_col].notna().any():
        fig2, ax_res = plt.subplots(figsize=(10, 6))
        
        ax_res.plot(df_plot[epoch_col], df_plot[residual_col], 'r-', linewidth=2)
        ax_res.set_xlabel(epoch_label, fontsize=12)
        ax_res.set_ylabel('Mean Residual', fontsize=12)
        ax_res.set_yscale('log')
        ax_res.set_title('Training Residual', fontsize=14)
        ax_res.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_name = output_dir / 'training_residual'
        plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_name}.pdf and .png")
        plt.close(fig2)


def plot_fixed_point_convergence(run_dir: Path, output_dir: Path = None):
    """
    Plot fixed-point iteration convergence.
    
    Args:
        run_dir: run directory containing metrics.csv
        output_dir: output directory
    """
    if output_dir is None:
        output_dir = run_dir / 'plots'
    
    output_dir.mkdir(exist_ok=True)
    
    metrics_file = run_dir / 'metrics.csv'
    if not metrics_file.exists():
        return
    
    df = pd.read_csv(metrics_file)
    
    # Ensure epoch column exists
    if 'epoch' not in df.columns:
        if 'phi_epoch' in df.columns:
            df['epoch'] = df['phi_epoch']
        else:
            df['epoch'] = range(len(df))
    
    # Group by outer iteration
    if 'outer_iter' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        loss_candidates = ['loss', 'loss_eik', 'phi_loss']
        loss_col = next((col for col in loss_candidates if col in df.columns), None)
        if loss_col is None:
            return

        for outer_iter in df['outer_iter'].unique():
            df_iter = df[df['outer_iter'] == outer_iter]
            ax.plot(df_iter['epoch'], df_iter[loss_col], 
                   linewidth=1.5, alpha=0.7, label=f'Outer iter {outer_iter}')
        
        ax.set_xlabel('Epoch (within outer iteration)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_yscale('log')
        ax.set_title('Fixed-Point Iteration Convergence', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
        
        plt.tight_layout()
        
        output_name = output_dir / 'fixed_point_convergence'
        plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_name}.pdf and .png")
        plt.close(fig)


def plot_rho_solver_convergence(run_dir: Path, output_dir: Path = None):
    """
    Plot rho solver convergence metrics across outer iterations.

    Generates three plots:
    - rho_solver_residual
    - rho_solver_iters
    - rho_change
    """
    if output_dir is None:
        output_dir = run_dir / 'plots'

    output_dir.mkdir(exist_ok=True)

    metrics_file = run_dir / 'metrics.csv'
    if not metrics_file.exists():
        return

    df = pd.read_csv(metrics_file)
    if 'outer_iter' not in df.columns:
        return

    outer_iter = pd.to_numeric(df['outer_iter'], errors='coerce')
    df = df.loc[outer_iter.notna()].copy()
    df['outer_iter'] = outer_iter[outer_iter.notna()]

    # Prefer outer_summary rows if present
    if 'phi_epoch' in df.columns:
        df_summary = df[df['phi_epoch'] == 'outer_summary'].copy()
        if not df_summary.empty:
            df = df_summary

    df = df.sort_values('outer_iter')

    def _plot_series(col: str, ylabel: str, title: str, filename: str, logy: bool = True):
        if col not in df.columns:
            return
        series = pd.to_numeric(df[col], errors='coerce')
        if series.notna().sum() == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['outer_iter'], series, marker='o', linewidth=1.8, color='tab:blue')
        ax.set_xlabel('Outer iteration', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if logy:
            ax.set_yscale('log')
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_name = output_dir / filename
        plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_name}.pdf and .png")
        plt.close(fig)

    _plot_series(
        'rho_residual',
        ylabel='ρ residual',
        title='ρ solver residual',
        filename='rho_solver_residual',
        logy=True
    )
    _plot_series(
        'rho_solver_iters',
        ylabel='Iterations',
        title='ρ solver iterations',
        filename='rho_solver_iters',
        logy=False
    )
    _plot_series(
        'rho_change',
        ylabel='Relative change',
        title='ρ update change',
        filename='rho_change',
        logy=True
    )
