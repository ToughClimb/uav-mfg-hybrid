"""Eikonal residual loss with obstacle masking."""

import torch
import torch.nn as nn
from typing import Optional


class EikonalLoss(nn.Module):
    """r = v_max(ρ)||∇φ||_ε - v_w·∇φ - 1 (optional mask)."""
    
    def __init__(self, epsilon_reg: float = 1e-4, nu: float = 0.0):
        """Initialize loss."""
        super().__init__()
        self.epsilon_reg = epsilon_reg
        self.nu = nu
    
    def compute_gradient(self, phi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute ∇φ by autograd (x must require grad)."""
        grad_phi = torch.autograd.grad(
            outputs=phi,
            inputs=x,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
            retain_graph=True
        )[0]
        return grad_phi
    
    def compute_laplacian(self, phi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute Δφ (for viscosity term)."""
        grad_phi = self.compute_gradient(phi, x)  # (N, 3)
        
        laplacian = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        for i in range(3):
            grad_i = grad_phi[:, i:i+1]
            grad2_i = torch.autograd.grad(
                outputs=grad_i,
                inputs=x,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True,
                retain_graph=True
            )[0][:, i:i+1]
            laplacian += grad2_i
        
        return laplacian
    
    def forward(self, phi: torch.Tensor, x: torch.Tensor, 
                v_max: torch.Tensor, v_w: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> dict:
        """Compute loss from r = v_max||∇φ||_ε - v_w·∇φ - 1."""
        grad_phi = self.compute_gradient(phi, x)  # (N, 3)

        grad_norm_sq = torch.sum(grad_phi**2, dim=1, keepdim=True)
        grad_norm_reg = torch.sqrt(grad_norm_sq + self.epsilon_reg**2)

        wind_term = torch.sum(v_w * grad_phi, dim=1, keepdim=True)

        residual = v_max * grad_norm_reg - wind_term - 1.0

        if self.nu > 0.0:
            laplacian = self.compute_laplacian(phi, x)
            residual = residual - self.nu * laplacian

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
