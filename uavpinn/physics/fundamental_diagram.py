"""
Fundamental Diagram: Softmax-Greenshields congestion model.
Strictly follows mathematical model section 5.
"""

import torch
import torch.nn as nn


class FundamentalDiagram(nn.Module):
    """
    Softmax-Greenshields fundamental diagram:
    v_max(ρ) = Softmax_β(v_min, v_max^0 * (1 - ρ/ρ_jam))
    
    where Softmax_β(a, b) = (1/β) * log(exp(β*a) + exp(β*b))
    """
    
    def __init__(self, v_max_0: float, v_min: float, rho_jam: float, 
                 beta: float, clip_to_bounds: bool = False):
        """
        Args:
            v_max_0: free-flow maximum airspeed in m/s
            v_min: minimum allowed speed (congestion saturation) in m/s
            rho_jam: congestion threshold/scale parameter in UAVs/m³
            beta: smoothness parameter for Softmax (dimensionless)
            clip_to_bounds: if True, clip v_max to [v_min, v_max_0]
        """
        super().__init__()
        self.v_max_0 = float(v_max_0)
        self.v_min = float(v_min)
        self.rho_jam = float(rho_jam)
        self.beta = float(beta)
        self.clip_to_bounds = clip_to_bounds
    
    def softmax_beta(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Smooth maximum: Softmax_β(a, b) = (1/β) * log(exp(β*a) + exp(β*b))
        
        Args:
            a, b: tensors of same shape
            
        Returns:
            smooth maximum of a and b
        """
        # Use log-sum-exp trick for numerical stability
        max_val = torch.max(a, b)
        return max_val + (1.0 / self.beta) * torch.log(
            torch.exp(self.beta * (a - max_val)) + 
            torch.exp(self.beta * (b - max_val))
        )
    
    def v_max(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum airspeed as function of density.
        
        Args:
            rho: (N, 1) density field in UAVs/m³
            
        Returns:
            (N, 1) maximum airspeed in m/s
        """
        assert rho.shape[-1] == 1, f"Expected (N, 1) density, got {rho.shape}"
        
        # Greenshields term: v_max^0 * (1 - ρ/ρ_jam)
        greenshields = self.v_max_0 * (1.0 - rho / self.rho_jam)
        
        # Softmax with v_min
        v_min_tensor = torch.full_like(rho, self.v_min)
        v_max_val = self.softmax_beta(v_min_tensor, greenshields)
        
        # Optional clipping to strict bounds
        if self.clip_to_bounds:
            v_max_val = torch.clamp(v_max_val, min=self.v_min, max=self.v_max_0)
        
        return v_max_val
    
    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Alias for v_max."""
        return self.v_max(rho)
    
    def check_properties(self, rho_max: float, n_samples: int = 100) -> dict:
        """
        Check fundamental diagram properties for validation.
        
        Args:
            rho_max: maximum density to check
            n_samples: number of samples
            
        Returns:
            dict with check results
        """
        rho_samples = torch.linspace(0, rho_max, n_samples).reshape(-1, 1)
        v_samples = self.v_max(rho_samples)
        
        # Check 1: v_max(ρ) >= v_min
        min_speed = v_samples.min().item()
        check_min = min_speed >= self.v_min - 1e-6
        
        # Check 2: v_max(0) ≈ v_max^0
        v_at_zero = v_samples[0].item()
        check_zero = abs(v_at_zero - self.v_max_0) < 0.1 * self.v_max_0
        
        # Check 3: monotonically non-increasing
        dv = v_samples[1:] - v_samples[:-1]
        check_monotone = torch.all(dv <= 1e-6).item()
        
        return {
            'min_speed': min_speed,
            'v_at_zero': v_at_zero,
            'check_min_bound': check_min,
            'check_zero_approx': check_zero,
            'check_monotone': check_monotone,
            'all_passed': check_min and check_zero and check_monotone
        }
