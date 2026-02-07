"""
Zero-flux boundary condition loss.
Strictly follows code spec section 6.3 and mathematical model section 6.2.
"""

import torch
import torch.nn as nn


class FluxLoss(nn.Module):
    """
    Zero-flux boundary condition loss:
    flux = (ρu - κ∇ρ) · n = 0
    
    Applied on ∂Ω ∪ ∂Ω_obs
    """
    
    def __init__(self, epsilon_reg: float = 1e-4, kappa: float = 0.0):
        """
        Args:
            epsilon_reg: regularization for gradient norm
            kappa: diffusion coefficient
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
    
    def forward(self, rho: torch.Tensor, phi: torch.Tensor, x: torch.Tensor,
                v_max: torch.Tensor, v_w: torch.Tensor, 
                normals: torch.Tensor) -> dict:
        """
        Compute flux boundary condition loss.
        
        Args:
            rho: (N, 1) density at boundary points
            phi: (N, 1) potential at boundary points
            x: (N, 3) boundary coordinates (must have requires_grad=True)
            v_max: (N, 1) maximum airspeed
            v_w: (N, 3) wind velocity
            normals: (N, 3) outward normal vectors
            
        Returns:
            dict with 'loss', 'flux', 'mean_flux'
        """
        # Compute ∇φ
        grad_phi = self.compute_gradient(phi, x)  # (N, 3)
        grad_norm_sq = torch.sum(grad_phi**2, dim=1, keepdim=True)
        grad_norm_reg = torch.sqrt(grad_norm_sq + self.epsilon_reg**2)
        
        # Velocity field: u = v_w - v_max(ρ) * ∇φ / ||∇φ||_ε
        u = v_w - v_max * (grad_phi / grad_norm_reg)  # (N, 3)
        
        # Advective flux: ρu
        advective_flux = rho * u  # (N, 3)
        
        # Total flux: ρu - κ∇ρ
        if self.kappa > 0.0:
            grad_rho = self.compute_gradient(rho, x)  # (N, 3)
            total_flux = advective_flux - self.kappa * grad_rho
        else:
            total_flux = advective_flux
        
        # Normal flux: (ρu - κ∇ρ) · n
        flux_normal = torch.sum(total_flux * normals, dim=1, keepdim=True)  # (N, 1)
        
        # Loss: mean squared flux
        loss = torch.mean(flux_normal**2)
        mean_flux = torch.mean(torch.abs(flux_normal))
        
        return {
            'loss': loss,
            'flux': flux_normal,
            'mean_flux': mean_flux
        }
