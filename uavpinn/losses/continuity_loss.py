"""
Continuity equation residual loss.
Strictly follows code spec section 6.2 and mathematical model section 4.
"""

import torch
import torch.nn as nn
from typing import Optional


class ContinuityLoss(nn.Module):
    """
    Continuity equation residual loss:
    r_cont = ∇·(ρu) - κΔρ - q
    
    where u = v_w - v_max(ρ) * ∇φ / ||∇φ||_ε
    """
    
    def __init__(self, epsilon_reg: float = 1e-4, kappa: float = 0.0):
        """
        Args:
            epsilon_reg: regularization for gradient norm
            kappa: diffusion coefficient (default 0 for pure advection)
        """
        super().__init__()
        self.epsilon_reg = epsilon_reg
        self.kappa = kappa
    
    def compute_gradient(self, field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient using autograd."""
        grad = torch.autograd.grad(
            outputs=field,
            inputs=x,
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            retain_graph=True
        )[0]
        return grad
    
    def compute_divergence(self, vector_field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute divergence of a vector field.
        
        Args:
            vector_field: (N, 3) vector field
            x: (N, 3) coordinates
            
        Returns:
            (N, 1) divergence
        """
        div = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        
        for i in range(3):
            component = vector_field[:, i:i+1]
            grad_component = self.compute_gradient(component, x)
            div += grad_component[:, i:i+1]
        
        return div
    
    def compute_laplacian(self, field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of a scalar field."""
        grad = self.compute_gradient(field, x)  # (N, 3)
        
        laplacian = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        for i in range(3):
            grad_i = grad[:, i:i+1]
            grad2_i = self.compute_gradient(grad_i, x)
            laplacian += grad2_i[:, i:i+1]
        
        return laplacian
    
    def forward(self, rho: torch.Tensor, phi: torch.Tensor, x: torch.Tensor,
                v_max: torch.Tensor, v_w: torch.Tensor, q: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> dict:
        """
        Compute continuity residual loss.
        
        Args:
            rho: (N, 1) density
            phi: (N, 1) potential
            x: (N, 3) coordinates (must have requires_grad=True)
            v_max: (N, 1) maximum airspeed
            v_w: (N, 3) wind velocity
            q: (N, 1) source term
            mask: (N, 1) optional mask
            
        Returns:
            dict with 'loss', 'residual', 'mean_residual'
        """
        # Compute ∇φ
        grad_phi = self.compute_gradient(phi, x)  # (N, 3)
        grad_norm_sq = torch.sum(grad_phi**2, dim=1, keepdim=True)
        grad_norm_reg = torch.sqrt(grad_norm_sq + self.epsilon_reg**2)
        
        # Velocity field: u = v_w - v_max(ρ) * ∇φ / ||∇φ||_ε
        u = v_w - v_max * (grad_phi / grad_norm_reg)  # (N, 3)
        
        # Flux: ρu
        flux = rho * u  # (N, 3)
        
        # Divergence: ∇·(ρu)
        div_flux = self.compute_divergence(flux, x)  # (N, 1)
        
        # Continuity residual: ∇·(ρu) - q
        residual = div_flux - q
        
        # Add diffusion term if kappa > 0
        if self.kappa > 0.0:
            laplacian_rho = self.compute_laplacian(rho, x)
            residual = residual - self.kappa * laplacian_rho
        
        # Apply mask if provided
        if mask is not None:
            residual_masked = residual * mask
            mask_sum = torch.sum(mask) + 1e-8
            loss = torch.sum(residual_masked**2) / mask_sum
            mean_residual = torch.sum(torch.abs(residual_masked)) / mask_sum
        else:
            loss = torch.mean(residual**2)
            mean_residual = torch.mean(torch.abs(residual))
        
        return {
            'loss': loss,
            'residual': residual,
            'mean_residual': mean_residual
        }
