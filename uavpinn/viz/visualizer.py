"""Visualization utilities for UAVPINN++ results."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm
from pathlib import Path
from typing import Optional

from .streamlines import plot_streamlines_2d
from .streamlines_u import plot_velocity_streamlines_2d


class Visualizer:
    """Plot φ/ρ/velocity fields and diagnostics."""
    
    def __init__(self, checkpoint_path: str, config: dict, objects: dict,
                 output_dir: Optional[Path] = None):
        """Initialize visualizer."""
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.config = config
        self.objects = objects
        
        if output_dir is None:
            self.output_dir = Path(checkpoint_path).parent.parent / 'plots'
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        from ..models import PhiModel
        
        domain_bounds = objects['domain_bounds']
        barrier_params = config.get('barrier', None)
        
        p_value = config.get('barrier', {}).get('p') or config.get('phi_bc', {}).get('power', 1)
        self.phi_model = PhiModel(
            hidden_layers=config['network']['hidden_layers'],
            target_shape=objects['target_shape'],
            domain_bounds=domain_bounds,
            p=p_value,
            obstacle_shape=objects['obstacle_shape'],
            barrier_params=barrier_params
        )
        self.phi_model.load_state_dict(self.checkpoint['phi_model_state'])
        self.phi_model.eval()
        
        if 'rho_grid' in self.checkpoint:
            self.rho_grid = self.checkpoint['rho_grid']
            self.rho_model = None
        elif 'rho_model_state' in self.checkpoint:
            from ..models import RhoModel
            self.rho_model = RhoModel(
                hidden_layers=config['network']['hidden_layers'],
                rho_max=config['physics']['rho_max'],
                domain_bounds=domain_bounds
            )
            self.rho_model.load_state_dict(self.checkpoint['rho_model_state'])
            self.rho_model.eval()
            
            from ..solvers import RhoSolver
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
                rho_flat = self.rho_model(rho_solver.grid_coords_flat)
                self.rho_grid = rho_flat.reshape(rho_solver.nz, rho_solver.ny, rho_solver.nx)
        else:
            raise ValueError("Checkpoint must contain either 'rho_grid' (Route B) or 'rho_model_state' (Route A)")

        self.domain_bounds = domain_bounds
    
    def create_2d_slice_grid(self, z_slice: float, resolution: int = 100):
        """Create a 2D grid at a given z height."""
        x_range = np.linspace(self.domain_bounds[0][0], self.domain_bounds[0][1], resolution)
        y_range = np.linspace(self.domain_bounds[1][0], self.domain_bounds[1][1], resolution)
        
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.full_like(X, z_slice)
        
        x_grid = np.stack([X, Y, Z], axis=-1)
        x_grid_tensor = torch.tensor(x_grid, dtype=torch.float64).reshape(-1, 3)
        
        return x_grid_tensor, X, Y
    
    def plot_phi_slices(self, z_slices=[5, 15, 25, 35]):
        """
        Plot phi potential field at multiple z heights.
        Each height saved as separate file for flexible composition.
        Follows code spec section 11.2.
        """
        print(f"\nGenerating phi slices at z={z_slices}...")
        
        phi_slices = []
        phi_min = np.inf
        phi_max = -np.inf
        
        for z in z_slices:
            x_grid, X, Y = self.create_2d_slice_grid(z, resolution=100)

            with torch.no_grad():
                phi_values = self.phi_model.phi_total(x_grid).numpy().reshape(100, 100)
            
            phi_slices.append((z, X, Y, phi_values))
            phi_min = min(phi_min, np.nanmin(phi_values))
            phi_max = max(phi_max, np.nanmax(phi_values))
        
        viz_cfg = self.config.get('viz', {})
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
        
        for z, X, Y, phi_values in phi_slices:
            fig, ax = plt.subplots(figsize=(8, 7))

            contour_kwargs = dict(levels=phi_levels, cmap='viridis')
            if phi_norm is None:
                contour_kwargs.update(vmin=phi_vmin, vmax=phi_vmax)
            else:
                contour_kwargs.update(norm=phi_norm)

            target_params = self.config['target']['params']
            in_target_slice = (
                z >= target_params['center'][2] - target_params['radius'] and
                z <= target_params['center'][2] + target_params['radius']
            )
            target_mask = None
            if in_target_slice:
                dx = X - target_params['center'][0]
                dy = Y - target_params['center'][1]
                target_mask = (dx * dx + dy * dy) <= target_params['radius'] ** 2

            obstacles_in_slice = []
            obstacle_mask = None
            if 'obstacles' in self.config and len(self.config['obstacles']) > 0:
                for obs in self.config['obstacles']:
                    if obs['type'] == 'aabb':
                        params = obs['params']
                        if z >= params['min'][2] and z <= params['max'][2]:
                            obstacles_in_slice.append(params)
                            obs_mask = (
                                (X >= params['min'][0]) & (X <= params['max'][0]) &
                                (Y >= params['min'][1]) & (Y <= params['max'][1])
                            )
                            obstacle_mask = obs_mask if obstacle_mask is None else (obstacle_mask | obs_mask)

            im = ax.contourf(X, Y, phi_values, **contour_kwargs)
            if target_mask is not None or obstacle_mask is not None:
                combined_mask = np.zeros_like(phi_values, dtype=bool)
                if target_mask is not None:
                    combined_mask |= target_mask
                if obstacle_mask is not None:
                    combined_mask |= obstacle_mask
                phi_contour = np.ma.array(phi_values, mask=combined_mask)
            else:
                phi_contour = phi_values
            ax.contour(X, Y, phi_contour, levels=10, colors='white', alpha=0.3, linewidths=0.5)

            if in_target_slice:
                circle = plt.Circle(
                    (target_params['center'][0], target_params['center'][1]),
                    target_params['radius'],
                    fill=False, edgecolor='red', linewidth=2, linestyle='--',
                    label='Target (sink)'
                )
                ax.add_patch(circle)

            if self.config['scenario']['type'] == 'p2p':
                source_cfg = self.config['physics']['source']['p2p_source']
                if z >= source_cfg['center'][2] - source_cfg['radius'] and \
                   z <= source_cfg['center'][2] + source_cfg['radius']:
                    source_circle = plt.Circle(
                        (source_cfg['center'][0], source_cfg['center'][1]),
                        source_cfg['radius'],
                        fill=False, edgecolor='green', linewidth=2, linestyle='-',
                        label='Source'
                    )
                    ax.add_patch(source_circle)

            if obstacles_in_slice:
                obstacle_labeled = False
                for params in obstacles_in_slice:
                    rect = plt.Rectangle(
                        (params['min'][0], params['min'][1]),
                        params['max'][0] - params['min'][0],
                        params['max'][1] - params['min'][1],
                        fill=True, facecolor='0.25', alpha=0.85,
                        edgecolor='0.1', linewidth=1.5,
                        label='Obstacle' if not obstacle_labeled else None
                    )
                    ax.add_patch(rect)
                    obstacle_labeled = True
            
            ax.set_xlabel('x (m)', fontsize=14)
            ax.set_ylabel('y (m)', fontsize=14)
            ax.set_title(f'φ at z={z}m', fontsize=15)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, linewidth=0.6, color='0.85')
            cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cbar.set_label('φ (s)', fontsize=14)
            cbar_pos = cbar.ax.get_position()
            cbar.ax.set_position([
                cbar_pos.x0,
                cbar_pos.y0 - 0.02,
                cbar_pos.width,
                cbar_pos.height
            ])
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(
                    loc='lower center',
                    bbox_to_anchor=(1.1, 1.03),
                    bbox_transform=cbar.ax.transAxes,
                    borderaxespad=0.0,
                    fontsize=12
                )
            
            plt.tight_layout()

            output_name = self.output_dir / f'phi_z{int(z)}'
            plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"  Saved: phi_z*.pdf and .png ({len(z_slices)} files)")
    
    def plot_rho_slices(self, z_slices=[5, 15, 25, 35], use_interpolation=True):
        """
        Plot rho density field at multiple z heights.
        Each height saved as separate file for flexible composition.
        
        Args:
            z_slices: list of z heights
            use_interpolation: if True, interpolate to higher resolution
        """
        print(f"\nGenerating rho slices at z={z_slices}...")
        
        rho_vmin = 0.0
        rho_vmax = float(self.config.get('physics', {}).get('rho_max', np.nan))
        if not np.isfinite(rho_vmax):
            rho_vmax = float(np.nanmax(self.rho_grid.detach().cpu().numpy()))
        if rho_vmax <= rho_vmin:
            rho_vmax = rho_vmin + 1e-6
        rho_levels = np.linspace(rho_vmin, rho_vmax, 21)
        
        for z in z_slices:
            fig, ax = plt.subplots(figsize=(8, 7))

            nz, ny, nx = self.rho_grid.shape
            z_coords = np.linspace(self.domain_bounds[2][0], self.domain_bounds[2][1], nz)

            if use_interpolation:
                from ..solvers import RhoSolver

                solver = RhoSolver(
                    domain_bounds=self.domain_bounds,
                    nx=nx, ny=ny, nz=nz,
                    target_shape=self.objects['target_shape'],
                    obstacle_shape=self.objects['obstacle_shape'],
                    dt=0.01
                )

                x_grid, X, Y = self.create_2d_slice_grid(z, resolution=200)

                with torch.no_grad():
                    rho_interp = solver.interpolate_to_points(self.rho_grid, x_grid)
                    rho_slice = rho_interp.numpy().reshape(200, 200)
            else:
                x_coords = np.linspace(self.domain_bounds[0][0], self.domain_bounds[0][1], nx)
                y_coords = np.linspace(self.domain_bounds[1][0], self.domain_bounds[1][1], ny)
                X, Y = np.meshgrid(x_coords, y_coords)

                z_idx = np.argmin(np.abs(z_coords - z))
                rho_slice = self.rho_grid[z_idx, :, :].cpu().numpy()

            im = ax.contourf(
                X,
                Y,
                rho_slice,
                levels=rho_levels,
                cmap='viridis',
                vmin=rho_vmin,
                vmax=rho_vmax
            )

            target_params = self.config['target']['params']
            if z >= target_params['center'][2] - target_params['radius'] and \
               z <= target_params['center'][2] + target_params['radius']:
                circle = plt.Circle(
                    (target_params['center'][0], target_params['center'][1]),
                    target_params['radius'],
                    fill=False, edgecolor='red', linewidth=2, linestyle='--',
                    label='Target (sink)'
                )
                ax.add_patch(circle)

            if self.config['scenario']['type'] == 'p2p':
                source_cfg = self.config['physics']['source']['p2p_source']
                if z >= source_cfg['center'][2] - source_cfg['radius'] and \
                   z <= source_cfg['center'][2] + source_cfg['radius']:
                    source_circle = plt.Circle(
                        (source_cfg['center'][0], source_cfg['center'][1]),
                        source_cfg['radius'],
                        fill=False, edgecolor='green', linewidth=2, linestyle='-',
                        label='Source'
                    )
                    ax.add_patch(source_circle)

            if 'obstacles' in self.config and len(self.config['obstacles']) > 0:
                obstacle_labeled = False
                for obs in self.config['obstacles']:
                    if obs['type'] == 'aabb':
                        params = obs['params']
                        if z >= params['min'][2] and z <= params['max'][2]:
                            rect = plt.Rectangle(
                                (params['min'][0], params['min'][1]),
                                params['max'][0] - params['min'][0],
                                params['max'][1] - params['min'][1],
                                fill=True, facecolor='0.25', alpha=0.8,
                                edgecolor='0.1', linewidth=1.5,
                                label='Obstacle' if not obstacle_labeled else None
                            )
                            ax.add_patch(rect)
                            obstacle_labeled = True
            
            ax.set_xlabel('x (m)', fontsize=14)
            ax.set_ylabel('y (m)', fontsize=14)
            ax.set_title(f'ρ at z={z}m', fontsize=15)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, linewidth=0.6, color='0.85')
            cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cbar.set_label('ρ (UAVs/m³)', fontsize=14)
            cbar_pos = cbar.ax.get_position()
            cbar.ax.set_position([
                cbar_pos.x0,
                cbar_pos.y0 - 0.02,
                cbar_pos.width,
                cbar_pos.height
            ])
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(
                    loc='lower center',
                    bbox_to_anchor=(1.1, 1.03),
                    bbox_transform=cbar.ax.transAxes,
                    borderaxespad=0.0,
                    fontsize=12
                )
            
            plt.tight_layout()

            suffix = '_interp' if use_interpolation else ''
            output_name = self.output_dir / f'rho_z{int(z)}{suffix}'
            plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        res_info = ' (interpolated 200x200)' if use_interpolation else ' (original 50x50)'
        print(f"  Saved: rho_z*.pdf and .png ({len(z_slices)} files){res_info}")
    
    def plot_training_curves(self):
        """Plot training loss curves from metrics.csv."""
        print(f"\nGenerating training curves...")
        
        import csv
        metrics_file = Path(self.checkpoint['config']['run']['exp_name'])
        
        checkpoint_dir = Path(self.checkpoint_path).parent.parent
        metrics_path = checkpoint_dir / 'metrics.csv'

        if not metrics_path.exists():
            print(f"  ⚠️  metrics.csv not found, skipping")
            return

        data = {'outer_iter': [], 'phi_epoch': [], 'loss': [], 'lr': []}
        with open(metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['phi_epoch'] != 'outer_summary':
                    data['outer_iter'].append(int(row['outer_iter']))
                    data['phi_epoch'].append(int(row['phi_epoch']))
                    data['loss'].append(float(row['loss_eik']))
                    data['lr'].append(float(row['lr']))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.semilogy(data['loss'], linewidth=1.5, color='blue', alpha=0.7)
        ax1.set_xlabel('Training Step', fontsize=12)
        ax1.set_ylabel('Eikonal Loss', fontsize=12)
        ax1.set_title('Training Loss Curve', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        ax2.semilogy(data['lr'], linewidth=1.5, color='red', alpha=0.7)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_name = self.output_dir / 'training_curves'
        plt.savefig(f'{output_name}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_name}.pdf and .png")
        
        plt.show()

    def export_paraview(
        self,
        output_dir: Optional[Path] = None,
        grid_resolution: Optional[tuple] = None,
        batch_size: int = 50000,
        prefix: str = 'final',
        iteration: Optional[int] = None
    ):
        from .paraview_export import ParaViewExporter

        if output_dir is None:
            paraview_dir = Path(self.checkpoint_path).parent.parent / 'paraview'
        else:
            paraview_dir = Path(output_dir)
        paraview_dir.mkdir(parents=True, exist_ok=True)

        rho_nz, rho_ny, rho_nx = self.rho_grid.shape
        if grid_resolution is None:
            grid_resolution = (rho_nx, rho_ny, rho_nz)

        rho_field = self.rho_grid
        if isinstance(rho_field, torch.Tensor):
            rho_field = rho_field.detach().cpu()

        if rho_field.ndim == 3 and tuple(rho_field.shape) == (
            grid_resolution[2], grid_resolution[1], grid_resolution[0]
        ):
            rho_field = rho_field.permute(2, 1, 0).contiguous()

        obs_shape = self.objects.get('obstacle_shape')
        obstacle_shapes = [obs_shape] if obs_shape is not None else []

        if iteration is None:
            it_raw = self.checkpoint.get('iteration', self.checkpoint.get('outer_iter', 0))
            if isinstance(it_raw, (int, np.integer)):
                iteration = int(it_raw)
            else:
                try:
                    iteration = int(it_raw)
                except Exception:
                    iteration = 0

        exporter = ParaViewExporter(
            output_dir=str(paraview_dir),
            domain_bounds=self.domain_bounds,
            grid_resolution=tuple(grid_resolution)
        )

        exporter.export_fields(
            phi_model=self.phi_model,
            rho_field=rho_field,
            wind_field=self.objects['wind_field'],
            fundamental_diagram=self.objects['fundamental_diagram'],
            target_shape=self.objects['target_shape'],
            obstacle_shapes=obstacle_shapes,
            iteration=iteration,
            prefix=prefix,
            batch_size=batch_size
        )

        return paraview_dir
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "="*70)
        print(" "*20 + "GENERATING VISUALIZATIONS")
        print("="*70)
        
        self.plot_phi_slices()
        self.plot_rho_slices()

        viz_cfg = self.config.get('viz', {})
        enable_streamlines = bool(viz_cfg.get('enable_streamlines', True))
        enable_u_streamlines = bool(viz_cfg.get('enable_u_streamlines', True))
        z_slices = viz_cfg.get('z_slices', [5, 15, 25, 35])
        if z_slices is None:
            z_slices = [5, 15, 25, 35]

        if enable_streamlines or enable_u_streamlines:
            print(f"\nGenerating streamline plots at z={z_slices}...")
            for z in z_slices:
                if enable_streamlines:
                    try:
                        plot_streamlines_2d(
                            checkpoint=self.checkpoint,
                            config=self.config,
                            objects=self.objects,
                            phi_model=self.phi_model,
                            output_dir=self.output_dir,
                            z_slice=float(z)
                        )
                    except Exception as e:
                        print(f"  ⚠️  streamlines_z{int(float(z))} failed: {e}")
                if enable_u_streamlines:
                    try:
                        plot_velocity_streamlines_2d(
                            checkpoint_path=str(self.checkpoint_path),
                            z_slice=float(z)
                        )
                    except Exception as e:
                        print(f"  ⚠️  streamlines_u_z{int(float(z))} failed: {e}")
        
        print("\n" + "="*70)
        print(f"✅ All plots saved to: {self.output_dir}")
        print("="*70 + "\n")
