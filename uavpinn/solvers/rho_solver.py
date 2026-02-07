"""
Finite-volume upwind solver for density.

Solves (∂ρ/∂t + ∇·(ρu) = q).
"""

import torch
import numpy as np
import warnings
from typing import Optional, Tuple


class RhoSolver:
    """Finite-volume solver on a regular Cartesian grid.

    BC semantics:
    - No-penetration on ∂Ω and obstacle interfaces: (ρu)·n = 0
    - Absorbing target: cells inside D are masked (ρ=0)
    """
    
    def __init__(self, domain_bounds: Tuple[Tuple[float, float], ...],
                 nx: int, ny: int, nz: int,
                 target_shape, obstacle_shape: Optional[object] = None,
                 dt: float = 0.01, max_iters: int = 5000, tol: float = 1e-6,
                 device: str = 'cpu', dtype: torch.dtype = torch.float64):
        """Initialize solver."""
        self.domain_bounds = domain_bounds
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.target_shape = target_shape
        self.obstacle_shape = obstacle_shape
        self.dt = dt
        self.max_iters = max_iters
        self.tol = tol
        self.device = device
        self.dtype = dtype
        
        self.dx = (domain_bounds[0][1] - domain_bounds[0][0]) / nx
        self.dy = (domain_bounds[1][1] - domain_bounds[1][0]) / ny
        self.dz = (domain_bounds[2][1] - domain_bounds[2][0]) / nz

        x_centers = torch.linspace(
            domain_bounds[0][0] + 0.5*self.dx,
            domain_bounds[0][1] - 0.5*self.dx,
            nx, dtype=dtype, device=device
        )
        y_centers = torch.linspace(
            domain_bounds[1][0] + 0.5*self.dy,
            domain_bounds[1][1] - 0.5*self.dy,
            ny, dtype=dtype, device=device
        )
        z_centers = torch.linspace(
            domain_bounds[2][0] + 0.5*self.dz,
            domain_bounds[2][1] - 0.5*self.dz,
            nz, dtype=dtype, device=device
        )

        Z, Y, X = torch.meshgrid(z_centers, y_centers, x_centers, indexing='ij')
        self.grid_coords = torch.stack([X, Y, Z], dim=-1)  # (nz, ny, nx, 3)

        self.grid_coords_flat = self.grid_coords.reshape(-1, 3)  # (nz*ny*nx, 3)

        self._compute_masks()
    
    def _compute_masks(self):
        """Compute masks for target/obstacle interiors."""
        sdf_target = self.target_shape.sdf(self.grid_coords_flat)
        self.mask_target = (sdf_target > 0.0).reshape(self.nz, self.ny, self.nx)

        if self.obstacle_shape is not None:
            sdf_obs = self.obstacle_shape.sdf(self.grid_coords_flat)
            self.mask_obs = (sdf_obs > 0.0).reshape(self.nz, self.ny, self.nx)
        else:
            self.mask_obs = torch.ones(self.nz, self.ny, self.nx, 
                                       dtype=torch.bool, device=self.device)

        self.mask_free = self.mask_target & self.mask_obs

        n_target_cells = torch.count_nonzero(~self.mask_target).item()
        if n_target_cells == 0:
            warnings.warn(
                f"Target region does not contain any rho-grid cell centers (n_target_cells=0). "
                f"This usually means the target is too small or misaligned for the current grid. "
                f"Grid spacing: dx={self.dx:.4f}, dy={self.dy:.4f}, dz={self.dz:.4f}.",
                UserWarning
            )
    
    def upwind_flux_1d(self, rho: torch.Tensor, u: torch.Tensor, 
                       dx: float, dim: int) -> torch.Tensor:
        """Upwind flux divergence contribution in one dimension."""
        if dim == 2:  # x direction
            rho_left = torch.roll(rho, shifts=1, dims=2)
            rho_right = torch.roll(rho, shifts=-1, dims=2)
            u_left = torch.roll(u, shifts=1, dims=2)
            u_right = torch.roll(u, shifts=-1, dims=2)

            rho_left[:, :, 0] = rho[:, :, 0]
            rho_right[:, :, -1] = rho[:, :, -1]
            u_left[:, :, 0] = 0.0
            u_right[:, :, -1] = 0.0

        elif dim == 1:  # y direction
            rho_left = torch.roll(rho, shifts=1, dims=1)
            rho_right = torch.roll(rho, shifts=-1, dims=1)
            u_left = torch.roll(u, shifts=1, dims=1)
            u_right = torch.roll(u, shifts=-1, dims=1)

            rho_left[:, 0, :] = rho[:, 0, :]
            rho_right[:, -1, :] = rho[:, -1, :]
            u_left[:, 0, :] = 0.0
            u_right[:, -1, :] = 0.0

        else:  # dim == 0, z direction
            rho_left = torch.roll(rho, shifts=1, dims=0)
            rho_right = torch.roll(rho, shifts=-1, dims=0)
            u_left = torch.roll(u, shifts=1, dims=0)
            u_right = torch.roll(u, shifts=-1, dims=0)

            rho_left[0, :, :] = rho[0, :, :]
            rho_right[-1, :, :] = rho[-1, :, :]
            u_left[0, :, :] = 0.0
            u_right[-1, :, :] = 0.0

        if self.obstacle_shape is not None:
            mask_left_obs = torch.roll(~self.mask_obs, shifts=1, dims=dim)
            mask_right_obs = torch.roll(~self.mask_obs, shifts=-1, dims=dim)

            u_left = torch.where(self.mask_obs & mask_left_obs, 
                                torch.zeros_like(u_left), u_left)
            u_right = torch.where(self.mask_obs & mask_right_obs,
                                 torch.zeros_like(u_right), u_right)

        flux_left = torch.where(u_left > 0, u_left * rho_left, u_left * rho)
        flux_right = torch.where(u_right > 0, u_right * rho, u_right * rho_right)
        div_flux = (flux_right - flux_left) / dx

        return div_flux
    
    def solve(self, u_field: torch.Tensor, q_field: torch.Tensor,
              rho_init: Optional[torch.Tensor] = None) -> dict:
        """Solve ∂ρ/∂t + ∇·(ρu) = q to steady state by pseudo-time marching."""
        def _finite_stats(name: str, t: torch.Tensor) -> str:
            t_detached = t.detach()
            n_nan = torch.isnan(t_detached).sum().item()
            n_inf = torch.isinf(t_detached).sum().item()
            if t_detached.numel() == 0:
                t_min = float('nan')
                t_max = float('nan')
            else:
                finite_mask = torch.isfinite(t_detached)
                if finite_mask.any():
                    finite_vals = t_detached[finite_mask]
                    t_min = torch.min(finite_vals).item()
                    t_max = torch.max(finite_vals).item()
                else:
                    t_min = float('nan')
                    t_max = float('nan')
            return (
                f"{name}: shape={tuple(t_detached.shape)}, dtype={t_detached.dtype}, device={t_detached.device}, "
                f"nan={n_nan}, inf={n_inf}, min={t_min:.6e}, max={t_max:.6e}"
            )

        if rho_init is None:
            rho = torch.full((self.nz, self.ny, self.nx), 0.01, 
                           dtype=self.dtype, device=self.device)
        else:
            rho = rho_init.clone()

        if not torch.isfinite(u_field).all():
            raise FloatingPointError(
                "Non-finite values detected in u_field before rho solve.\n" + _finite_stats("u_field", u_field)
            )
        if not torch.isfinite(q_field).all():
            raise FloatingPointError(
                "Non-finite values detected in q_field before rho solve.\n" + _finite_stats("q_field", q_field)
            )
        if not torch.isfinite(rho).all():
            raise FloatingPointError(
                "Non-finite values detected in rho_init before rho solve.\n" + _finite_stats("rho_init", rho)
            )
        
        ux = u_field[..., 0]  # (nz, ny, nx)
        uy = u_field[..., 1]
        uz = u_field[..., 2]

        u_mag = torch.sqrt(ux**2 + uy**2 + uz**2)
        if not torch.isfinite(u_mag).all():
            raise FloatingPointError(
                "Non-finite values detected in u_mag during CFL check.\n" + _finite_stats("u_mag", u_mag)
            )
        u_max = torch.max(u_mag).item()
        h_min = min(self.dx, self.dy, self.dz)
        cfl_loose = u_max * self.dt / h_min
        sum_u_over_h = (
            torch.abs(ux) / self.dx + torch.abs(uy) / self.dy + torch.abs(uz) / self.dz
        )
        max_u_over_h = torch.max(sum_u_over_h).item()
        cfl_sum_max = self.dt * max_u_over_h
        dt_safe_sum = 0.5 / (max_u_over_h + 1e-10)
        dt_eff = min(self.dt, dt_safe_sum)

        if cfl_loose > 1.0:
            dt_safe = 0.5 * h_min / (u_max + 1e-10)
            warnings.warn(
                f"\n"
                f"  ⚠️  CFL_loose = {cfl_loose:.3f} > 1.0 (coarse u_max/h_min check)\n"
                f"  Max |u| = {u_max:.4f} m/s, h_min = {h_min:.4f} m\n"
                f"  Suggested dt ≤ {dt_safe:.6f}\n",
                UserWarning
            )
        
        if cfl_sum_max > 1.0:
            warnings.warn(
                f"\n"
                f"  ⚠️  CFL_sum = {cfl_sum_max:.3f} > 1.0 (stricter stability check)\n"
                f"  Max (|ux|/dx + |uy|/dy + |uz|/dz) = {max_u_over_h:.4f}\n"
                f"  Suggested dt ≤ {dt_safe_sum:.6f}\n",
                UserWarning
            )
        
        if dt_eff < self.dt:
            warnings.warn(
                f"\n"
                f"  ℹ️  Using reduced dt={dt_eff:.6e} (from {self.dt:.6e}) to satisfy CFL_sum <= 1.0\n",
                UserWarning
            )
        
        q_field = q_field * self.mask_free.float()

        residual_history = []

        for iter in range(self.max_iters):
            rho_old = rho.clone()

            div_flux_x = self.upwind_flux_1d(rho, ux, self.dx, dim=2)
            div_flux_y = self.upwind_flux_1d(rho, uy, self.dy, dim=1)
            div_flux_z = self.upwind_flux_1d(rho, uz, self.dz, dim=0)

            div_flux_total = div_flux_x + div_flux_y + div_flux_z

            if not torch.isfinite(div_flux_total).all():
                raise FloatingPointError(
                    "Non-finite values detected in div_flux_total during rho solve.\n"
                    f"iter={iter+1}, dt={dt_eff}, dx={self.dx}, dy={self.dy}, dz={self.dz}, "
                    f"cfl_sum_max={cfl_sum_max:.6e}, cfl_loose={cfl_loose:.6e}, u_max={u_max:.6e}\n"
                    + _finite_stats("div_flux_total", div_flux_total)
                    + "\n"
                    + _finite_stats("rho_old", rho_old)
                )
            
            rho = rho - dt_eff * (div_flux_total - q_field)
            rho = rho * self.mask_free.float()
            rho = torch.clamp(rho, min=0.0)

            if not torch.isfinite(rho).all():
                raise FloatingPointError(
                    "Non-finite values detected in rho during rho solve.\n"
                    f"iter={iter+1}, dt={dt_eff}, dx={self.dx}, dy={self.dy}, dz={self.dz}, "
                    f"cfl_sum_max={cfl_sum_max:.6e}, cfl_loose={cfl_loose:.6e}, u_max={u_max:.6e}\n"
                    + _finite_stats("rho", rho)
                    + "\n"
                    + _finite_stats("div_flux_total", div_flux_total)
                    + "\n"
                    + _finite_stats("q_field", q_field)
                )
            
            num = torch.norm(rho - rho_old)
            den = torch.norm(rho_old)
            if not torch.isfinite(num) or not torch.isfinite(den):
                raise FloatingPointError(
                    "Non-finite values detected in residual norm computation during rho solve.\n"
                    f"iter={iter+1}, num={num.item()}, den={den.item()}\n"
                    + _finite_stats("rho", rho)
                    + "\n"
                    + _finite_stats("rho_old", rho_old)
                )
            residual = num / (den + 1e-10)
            residual_history.append(residual.item())
            
            if residual < self.tol:
                return {
                    'rho': rho,
                    'iterations': iter + 1,
                    'residual_history': residual_history,
                    'converged': True
                }
            
            if (iter + 1) % 500 == 0:
                print(f"  Iteration {iter+1}/{self.max_iters}, residual={residual.item():.2e}")

        print(f"  Warning: Max iterations reached, residual={residual.item():.2e}")
        return {
            'rho': rho,
            'iterations': self.max_iters,
            'residual_history': residual_history,
            'converged': False
        }
    
    def interpolate_to_points(self, rho_grid: torch.Tensor, 
                             points: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation from grid to arbitrary points."""
        x_norm = (points[:, 0] - self.domain_bounds[0][0]) / self.dx - 0.5
        y_norm = (points[:, 1] - self.domain_bounds[1][0]) / self.dy - 0.5
        z_norm = (points[:, 2] - self.domain_bounds[2][0]) / self.dz - 0.5

        x_norm = torch.clamp(x_norm, 0, self.nx - 1.001)
        y_norm = torch.clamp(y_norm, 0, self.ny - 1.001)
        z_norm = torch.clamp(z_norm, 0, self.nz - 1.001)

        ix = torch.floor(x_norm).long()
        iy = torch.floor(y_norm).long()
        iz = torch.floor(z_norm).long()

        fx = (x_norm - ix.to(x_norm.dtype)).unsqueeze(-1)
        fy = (y_norm - iy.to(y_norm.dtype)).unsqueeze(-1)
        fz = (z_norm - iz.to(z_norm.dtype)).unsqueeze(-1)

        ix1 = torch.clamp(ix + 1, max=self.nx - 1)
        iy1 = torch.clamp(iy + 1, max=self.ny - 1)
        iz1 = torch.clamp(iz + 1, max=self.nz - 1)

        c000 = rho_grid[iz, iy, ix].unsqueeze(-1)
        c001 = rho_grid[iz, iy, ix1].unsqueeze(-1)
        c010 = rho_grid[iz, iy1, ix].unsqueeze(-1)
        c011 = rho_grid[iz, iy1, ix1].unsqueeze(-1)
        c100 = rho_grid[iz1, iy, ix].unsqueeze(-1)
        c101 = rho_grid[iz1, iy, ix1].unsqueeze(-1)
        c110 = rho_grid[iz1, iy1, ix].unsqueeze(-1)
        c111 = rho_grid[iz1, iy1, ix1].unsqueeze(-1)

        c00 = c000 * (1 - fx) + c001 * fx
        c01 = c010 * (1 - fx) + c011 * fx
        c10 = c100 * (1 - fx) + c101 * fx
        c11 = c110 * (1 - fx) + c111 * fx

        c0 = c00 * (1 - fy) + c01 * fy
        c1 = c10 * (1 - fy) + c11 * fy

        rho_interp = c0 * (1 - fz) + c1 * fz

        return rho_interp
