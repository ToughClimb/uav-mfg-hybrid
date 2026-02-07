"""VTK export (structured grid) for ParaView visualization."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging


class ParaViewExporter:
    """Export simulation results to VTK format for ParaView visualization."""
    
    def __init__(
        self,
        output_dir: str,
        domain_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        grid_resolution: Optional[Tuple[int, int, int]] = None
    ):
        """Initialize exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.domain_bounds = domain_bounds
        self.grid_resolution = grid_resolution or (100, 100, 50)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ParaView exporter initialized: {self.output_dir}")
        self.logger.info(f"Grid resolution: {self.grid_resolution}")
    
    def create_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create structured grid coordinates."""
        nx, ny, nz = self.grid_resolution
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.domain_bounds
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        
        return np.meshgrid(x, y, z, indexing='ij')
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def _sanitize_array(self, array: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
        return arr
    
    def _evaluate_batched(
        self,
        model_or_func,
        input_tensor: torch.Tensor,
        batch_size: int
    ) -> np.ndarray:
        """Evaluate model/function in batches."""
        n_points = input_tensor.shape[0]
        n_batches = (n_points + batch_size - 1) // batch_size
        
        outputs = []
        target_device = None
        if isinstance(model_or_func, torch.nn.Module):
            try:
                target_device = next(model_or_func.parameters()).device
            except StopIteration:
                target_device = input_tensor.device
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_points)
                batch = input_tensor[start_idx:end_idx]
                if target_device is not None and batch.device != target_device:
                    batch = batch.to(target_device)
                
                output_batch = model_or_func(batch)
                outputs.append(output_batch.detach().cpu().numpy())
        
        # Concatenate all batches
        result = np.concatenate(outputs, axis=0)
        
        # Handle scalar outputs (squeeze if needed)
        if result.shape[-1] == 1 and len(result.shape) > 1:
            result = result.squeeze(-1)
        
        return result
    
    def export_fields(
        self,
        phi_model,
        rho_field: torch.Tensor,
        wind_field,
        fundamental_diagram,
        target_shape,
        obstacle_shapes: list,
        iteration: int,
        prefix: str = "fields",
        batch_size: int = 50000
    ):
        """
        Export all fields to VTK format.
        
        Args:
            phi_model: Trained phi neural network model
            rho_field: Density field tensor (nx, ny, nz)
            wind_field: Wind field object
            fundamental_diagram: Fundamental diagram object
            target_shape: Target region shape
            obstacle_shapes: List of obstacle shapes
            iteration: Current iteration number
            prefix: Filename prefix
        """
        self.logger.info(f"Exporting fields at iteration {iteration}...")
        
        # Create grid
        X, Y, Z = self.create_grid()
        nx, ny, nz = self.grid_resolution
        
        # Flatten grid for evaluation
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        n_points = grid_tensor.shape[0]

        phi_device = None
        if isinstance(phi_model, torch.nn.Module):
            try:
                phi_device = next(phi_model.parameters()).device
            except StopIteration:
                phi_device = grid_tensor.device
        else:
            phi_device = grid_tensor.device
        grid_tensor_phi = grid_tensor.to(phi_device)
        
        self.logger.info(f"Evaluating fields on {n_points} grid points...")
        
        # Evaluate phi field in batches
        phi_values = self._evaluate_batched(phi_model, grid_tensor_phi, batch_size)
        phi_field = phi_values.reshape(nx, ny, nz)
        
        # Compute gradient of phi (velocity direction) - already batched internally
        grad_phi = self._compute_gradient_field(phi_model, grid_tensor_phi, (nx, ny, nz), batch_size=10000)
        
        # Evaluate wind field in batches
        wind_values = self._evaluate_batched(wind_field, grid_tensor, batch_size)
        wind_field_grid = wind_values.reshape(nx, ny, nz, 3)
        
        # Compute density-dependent speed
        rho_grid = self._to_numpy(rho_field)
        if rho_grid.shape != (nx, ny, nz):
            self.logger.warning(f"Rho field shape mismatch: {rho_grid.shape} vs {(nx, ny, nz)}")
            rho_grid = np.zeros((nx, ny, nz))
        
        # Vectorized v_max computation
        self.logger.info("Computing v_max field...")
        rho_flat = torch.tensor(rho_grid.ravel(), dtype=torch.float32).unsqueeze(1)
        v_max_flat = self._evaluate_batched(fundamental_diagram.v_max, rho_flat, batch_size)
        v_max_field = v_max_flat.reshape(nx, ny, nz)
        
        # Compute velocity field: v = v_max * (-grad_phi / |grad_phi|) + v_w
        grad_phi_norm = np.linalg.norm(grad_phi, axis=-1, keepdims=True) + 1e-8
        velocity_field = v_max_field[..., np.newaxis] * (-grad_phi / grad_phi_norm) + wind_field_grid
        
        # Compute speed (magnitude)
        speed_field = np.linalg.norm(velocity_field, axis=-1)
        
        # Compute SDF for target and obstacles in batches
        self.logger.info("Computing SDF fields...")
        target_sdf = self._evaluate_batched(target_shape.sdf, grid_tensor, batch_size)
        target_sdf = target_sdf.reshape(nx, ny, nz)
        
        obstacle_sdf = np.full((nx, ny, nz), 1.0e6, dtype=np.float32)
        for obs in obstacle_shapes:
            obs_sdf = self._evaluate_batched(obs.sdf, grid_tensor, batch_size)
            obs_sdf = obs_sdf.reshape(nx, ny, nz)
            obstacle_sdf = np.minimum(obstacle_sdf, obs_sdf)
        
        # Create mask fields
        free_space_mask = (obstacle_sdf > 0).astype(float)
        target_mask = (target_sdf <= 0).astype(float)
        
        # Export to VTK
        filename = self.output_dir / f"{prefix}_iter_{iteration:04d}.vtk"
        scalar_fields = {
            'phi': self._sanitize_array(phi_field, fill_value=0.0),
            'rho': self._sanitize_array(rho_grid, fill_value=0.0),
            'speed': self._sanitize_array(speed_field, fill_value=0.0),
            'v_max': self._sanitize_array(v_max_field, fill_value=0.0),
            'target_sdf': self._sanitize_array(target_sdf, fill_value=1.0e6),
            'obstacle_sdf': self._sanitize_array(obstacle_sdf, fill_value=1.0e6),
            'free_space': self._sanitize_array(free_space_mask, fill_value=1.0),
            'target_region': self._sanitize_array(target_mask, fill_value=0.0)
        }
        vector_fields = {
            'velocity': self._sanitize_array(velocity_field, fill_value=0.0),
            'wind': self._sanitize_array(wind_field_grid, fill_value=0.0),
            'grad_phi': self._sanitize_array(grad_phi, fill_value=0.0)
        }

        self._write_vtk_structured_grid(
            filename,
            X, Y, Z,
            scalar_fields,
            vector_fields
        )
        
        self.logger.info(f"Exported to {filename}")
    
    def _compute_gradient_field(
        self,
        phi_model,
        grid_tensor: torch.Tensor,
        shape: Tuple[int, int, int],
        batch_size: int = 10000
    ) -> np.ndarray:
        """
        Compute gradient of phi using automatic differentiation with batching.
        
        Args:
            phi_model: Neural network model
            grid_tensor: Grid points (N, 3)
            shape: Output shape (nx, ny, nz)
            batch_size: Batch size for gradient computation (default: 10000)
                       Reduce if OOM, increase for faster computation
        
        Returns:
            Gradient field (nx, ny, nz, 3)
        """
        nx, ny, nz = shape
        n_points = grid_tensor.shape[0]
        
        # Allocate output array
        grad_phi_all = np.zeros((n_points, 3), dtype=np.float32)
        
        # Compute gradients in batches to avoid OOM
        n_batches = (n_points + batch_size - 1) // batch_size
        
        self.logger.info(f"Computing gradients for {n_points} points in {n_batches} batches...")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_points)
            
            # Get batch
            batch = grid_tensor[start_idx:end_idx].clone().requires_grad_(True)
            
            # Compute gradient for this batch
            with torch.enable_grad():
                phi_batch = phi_model(batch)
                grad_outputs = torch.ones_like(phi_batch)
                grad_phi_batch = torch.autograd.grad(
                    outputs=phi_batch,
                    inputs=batch,
                    grad_outputs=grad_outputs,
                    create_graph=False,
                    retain_graph=False
                )[0]
            
            # Store result
            grad_phi_all[start_idx:end_idx] = grad_phi_batch.detach().cpu().numpy()
            
            # Progress logging
            if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
                progress = (i + 1) / n_batches * 100
                self.logger.info(f"  Progress: {progress:.1f}% ({i+1}/{n_batches} batches)")
        
        return grad_phi_all.reshape(nx, ny, nz, 3)
    
    def _write_vtk_structured_grid(
        self,
        filename: Path,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        scalar_fields: Dict[str, np.ndarray],
        vector_fields: Dict[str, np.ndarray]
    ):
        """
        Write VTK structured grid file.
        
        Args:
            filename: Output filename
            X, Y, Z: Grid coordinates (nx, ny, nz)
            scalar_fields: Dictionary of scalar fields
            vector_fields: Dictionary of vector fields
        """
        nx, ny, nz = X.shape
        n_points = nx * ny * nz
        
        with open(filename, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("UAV PINN Simulation Results\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
            f.write(f"POINTS {n_points} float\n")
            
            # Write points
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{X[i,j,k]:.6f} {Y[i,j,k]:.6f} {Z[i,j,k]:.6f}\n")
            
            # Write point data
            f.write(f"\nPOINT_DATA {n_points}\n")
            
            # Write scalar fields
            for name, field in scalar_fields.items():
                f.write(f"\nSCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            f.write(f"{field[i,j,k]:.6e}\n")
            
            # Write vector fields
            for name, field in vector_fields.items():
                f.write(f"\nVECTORS {name} float\n")
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            f.write(f"{field[i,j,k,0]:.6e} {field[i,j,k,1]:.6e} {field[i,j,k,2]:.6e}\n")
    
    def export_time_series(
        self,
        phi_model,
        rho_fields: list,
        wind_field,
        fundamental_diagram,
        target_shape,
        obstacle_shapes: list,
        prefix: str = "timeseries"
    ):
        """
        Export time series of fields (for animation in ParaView).
        
        Args:
            phi_model: Trained phi model
            rho_fields: List of density fields at different iterations
            wind_field: Wind field object
            fundamental_diagram: Fundamental diagram object
            target_shape: Target shape
            obstacle_shapes: List of obstacles
            prefix: Filename prefix
        """
        for idx, rho_field in enumerate(rho_fields):
            self.export_fields(
                phi_model,
                rho_field,
                wind_field,
                fundamental_diagram,
                target_shape,
                obstacle_shapes,
                iteration=idx,
                prefix=prefix
            )
        
        # Create PVD file for time series
        self._write_pvd_file(prefix, len(rho_fields))
    
    def _write_pvd_file(self, prefix: str, n_timesteps: int):
        """Write ParaView Data (PVD) file for time series."""
        pvd_filename = self.output_dir / f"{prefix}.pvd"
        
        with open(pvd_filename, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            
            for i in range(n_timesteps):
                vtk_file = f"{prefix}_iter_{i:04d}.vtk"
                f.write(f'    <DataSet timestep="{i}" group="" part="0" file="{vtk_file}"/>\n')
            
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        self.logger.info(f"Created PVD file: {pvd_filename}")
