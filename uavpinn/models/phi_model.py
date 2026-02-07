"""Φ model with hard absorbing BC and optional obstacle barrier."""

import torch
import torch.nn as nn
from .mlp import MLP
from typing import Optional


class PhiModel(nn.Module):
    """Potential network with hard BC.

    - phi_bc(x) = d_D(x)^p * softplus(nn_phi(x_nn))
    - phi_total(x) = phi_bc(x) + d_D(x)^p * Barrier_obs(x)
    - Barrier_obs(x) = C_height * softplus(-k * sdf_obs(x))
    """
    
    def __init__(self, hidden_layers: list, target_shape,
                 domain_bounds: tuple, p: int = 1,
                 obstacle_shape: Optional[object] = None,
                 barrier_params: Optional[dict] = None):
        """Initialize model."""
        super().__init__()
        
        self.target_shape = target_shape
        self.domain_bounds = domain_bounds
        self.p = p
        self.obstacle_shape = obstacle_shape
        
        if barrier_params is not None:
            self.C_height = barrier_params.get('C_height', 0.02)
            self.k = barrier_params.get('k', 0.5)
        else:
            self.C_height = 0.0
            self.k = 0.0

        self.nn_phi = MLP(
            input_dim=3,
            hidden_layers=hidden_layers,
            output_dim=1,
            activation='tanh',
            dtype=torch.float64
        )
        
        self.register_buffer('x_min', torch.tensor([domain_bounds[0][0], 
                                                     domain_bounds[1][0], 
                                                     domain_bounds[2][0]], dtype=torch.float64))
        self.register_buffer('x_max', torch.tensor([domain_bounds[0][1], 
                                                     domain_bounds[1][1], 
                                                     domain_bounds[2][1]], dtype=torch.float64))
    
    def normalize_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [-1, 1] for network input."""
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
    
    def phi_bc(self, x: torch.Tensor) -> torch.Tensor:
        """phi_bc(x) = d_D(x)^p * softplus(nn_phi(x_nn))."""
        d_D = self.target_shape.distance_nonneg(x)
        x_nn = self.normalize_coords(x)
        nn_out = self.nn_phi(x_nn)
        phi_nn = torch.nn.functional.softplus(nn_out)
        phi_bc_val = (d_D ** self.p) * phi_nn
        return phi_bc_val
    
    def barrier_obs(self, x: torch.Tensor) -> torch.Tensor:
        """Barrier_obs(x) = C_height * softplus(-k * sdf_obs(x))."""
        if self.obstacle_shape is None or self.C_height == 0.0:
            return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

        sdf_obs = self.obstacle_shape.sdf(x)
        sdf_clipped = torch.clamp(-self.k * sdf_obs, min=-50.0, max=50.0)
        barrier = self.C_height * torch.nn.functional.softplus(sdf_clipped)
        return barrier
    
    def phi_total(self, x: torch.Tensor) -> torch.Tensor:
        """phi_total(x) = phi_bc(x) + d_D(x)^p * Barrier_obs(x)."""
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"

        phi_bc_val = self.phi_bc(x)

        if self.obstacle_shape is not None and self.C_height > 0.0:
            d_D = self.target_shape.distance_nonneg(x)
            barrier = self.barrier_obs(x)

            assert phi_bc_val.shape == d_D.shape == barrier.shape, \
                f"Shape mismatch: phi_bc={phi_bc_val.shape}, d_D={d_D.shape}, barrier={barrier.shape}"

            barrier_term = (d_D ** self.p) * barrier
            phi_total_val = phi_bc_val + barrier_term
        else:
            phi_total_val = phi_bc_val

        return phi_total_val
    
    def forward(self, x: torch.Tensor, return_bc: bool = False) -> torch.Tensor:
        """Forward pass."""
        if return_bc:
            return self.phi_bc(x)
        else:
            return self.phi_total(x)
