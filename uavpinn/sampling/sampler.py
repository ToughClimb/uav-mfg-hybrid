"""Sampling for PINN training."""

import torch
import numpy as np
from typing import Optional, Tuple


class Sampler:
    """Samples PDE points excluding target/obstacle interiors."""
    
    def __init__(self, domain_bounds: Tuple[Tuple[float, float], ...],
                 target_shape, obstacle_shape: Optional[object] = None,
                 n_pde: int = 5000, n_boundary: int = 1000, 
                 n_depot: int = 200, n_obstacle: int = 0,
                 device: str = 'cpu', dtype: torch.dtype = torch.float64):
        """Initialize sampler."""
        self.domain_bounds = domain_bounds
        self.target_shape = target_shape
        self.obstacle_shape = obstacle_shape
        
        self.n_pde = n_pde
        self.n_boundary = n_boundary
        self.n_depot = n_depot
        self.n_obstacle = n_obstacle
        
        self.device = device
        self.dtype = dtype
        
        self.x_min = torch.tensor([b[0] for b in domain_bounds], dtype=dtype, device=device)
        self.x_max = torch.tensor([b[1] for b in domain_bounds], dtype=dtype, device=device)
    
    def sample_uniform_domain(self, n: int) -> torch.Tensor:
        """Uniform samples in Ω."""
        x = torch.rand(n, 3, dtype=self.dtype, device=self.device)
        x = self.x_min + x * (self.x_max - self.x_min)
        return x
    
    def sample_pde_points(self) -> torch.Tensor:
        """PDE collocation points excluding target/obstacle interiors."""
        n_oversample = self.n_pde * 3
        x = self.sample_uniform_domain(n_oversample)

        sdf_target = self.target_shape.sdf(x)
        mask_outside_target = (sdf_target > 0.0).squeeze()

        if self.obstacle_shape is not None:
            sdf_obs = self.obstacle_shape.sdf(x)
            mask_outside_obs = (sdf_obs > 0.0).squeeze()
            mask = mask_outside_target & mask_outside_obs
        else:
            mask = mask_outside_target

        x_valid = x[mask]

        if x_valid.shape[0] < self.n_pde:
            print(f"Warning: only {x_valid.shape[0]} valid PDE points, requested {self.n_pde}")
            return x_valid
        else:
            return x_valid[:self.n_pde]
    
    def sample_domain_boundary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples on ∂Ω with outward normals."""
        n_per_face = self.n_boundary // 6
        points_list = []
        normals_list = []

        for dim in range(3):
            for side in [0, 1]:
                x = torch.rand(n_per_face, 3, dtype=self.dtype, device=self.device)
                x = self.x_min + x * (self.x_max - self.x_min)

                if side == 0:
                    x[:, dim] = self.x_min[dim]
                    normal = torch.zeros(n_per_face, 3, dtype=self.dtype, device=self.device)
                    normal[:, dim] = -1.0
                else:
                    x[:, dim] = self.x_max[dim]
                    normal = torch.zeros(n_per_face, 3, dtype=self.dtype, device=self.device)
                    normal[:, dim] = 1.0
                
                points_list.append(x)
                normals_list.append(normal)
        
        points = torch.cat(points_list, dim=0)
        normals = torch.cat(normals_list, dim=0)
        
        return points, normals
    
    def sample_obstacle_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples near ∂Ω_obs and returns normals."""
        if self.obstacle_shape is None or self.n_obstacle == 0:
            return (torch.empty(0, 3, dtype=self.dtype, device=self.device),
                    torch.empty(0, 3, dtype=self.dtype, device=self.device))
        
        n_oversample = self.n_obstacle * 100
        x = self.sample_uniform_domain(n_oversample)
        
        sdf = self.obstacle_shape.sdf(x)
        
        band_width = 2.0  # meters
        mask_near_surface = (torch.abs(sdf) < band_width).squeeze()
        x_near = x[mask_near_surface]
        
        if x_near.shape[0] < self.n_obstacle:
            print(f"Warning: only {x_near.shape[0]} obstacle surface points, requested {self.n_obstacle}")
            n_actual = x_near.shape[0]
        else:
            n_actual = self.n_obstacle
            x_near = x_near[:n_actual]
        
        if n_actual > 0:
            normals = self.obstacle_shape.normal(x_near)
        else:
            normals = torch.empty(0, 3, dtype=self.dtype, device=self.device)
        
        return x_near, normals
    
    def sample_target_band(self) -> torch.Tensor:
        """Samples in a narrow band near ∂D."""
        n_oversample = self.n_depot * 10
        x = self.sample_uniform_domain(n_oversample)
        
        sdf_target = self.target_shape.sdf(x)
        
        band_width = 2.0  # meters
        mask_near_boundary = (torch.abs(sdf_target) < band_width).squeeze()
        x_near = x[mask_near_boundary]
        
        if x_near.shape[0] < self.n_depot:
            return x_near
        else:
            return x_near[:self.n_depot]
    
    def sample_all(self) -> dict:
        """Samples all point types."""
        pde_points = self.sample_pde_points()
        boundary_points, boundary_normals = self.sample_domain_boundary()
        obstacle_points, obstacle_normals = self.sample_obstacle_surface()
        target_band_points = self.sample_target_band()
        
        return {
            'pde_points': pde_points,
            'boundary_points': boundary_points,
            'boundary_normals': boundary_normals,
            'obstacle_points': obstacle_points,
            'obstacle_normals': obstacle_normals,
            'target_band_points': target_band_points
        }
