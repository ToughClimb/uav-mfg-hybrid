"""
Source term q(x) implementations.
Strictly follows 复现实验补充.md section 2.
"""

import torch
import torch.nn as nn
from typing import Optional


class SourceTerm(nn.Module):
    """
    Source term q(x) for continuity equation.
    Supports Homing (uniform) and P2P (source sphere) scenarios.
    
    IMPORTANT: Source term is only defined on Ω_free \\ D (excludes target region).
    """
    
    def __init__(self, scenario: str, target_shape, q0: float = 0.0,
                 source_sphere_center: Optional[tuple] = None,
                 source_sphere_radius: Optional[float] = None,
                 q_source: Optional[float] = None):
        """
        Args:
            scenario: 'homing' or 'p2p'
            target_shape: Shape object for target region D
            q0: uniform source term in UAVs/(m³·s) for homing
            source_sphere_center: (x, y, z) for P2P source sphere
            source_sphere_radius: radius in meters for P2P source sphere
            q_source: injection rate in UAVs/(m³·s) for P2P source sphere
        """
        super().__init__()
        self.scenario = scenario.lower()
        self.target_shape = target_shape
        self.q0 = float(q0)
        
        if self.scenario == 'p2p':
            assert source_sphere_center is not None, "P2P requires source_sphere_center"
            assert source_sphere_radius is not None, "P2P requires source_sphere_radius"
            assert q_source is not None, "P2P requires q_source"
            
            self.source_center = torch.tensor(source_sphere_center, dtype=torch.float64).reshape(1, 3)
            self.source_radius = float(source_sphere_radius)
            self.q_source = float(q_source)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate source term at given points.
        Source is only active in Ω_free \\ D (excludes target region).
        
        Args:
            x: (N, 3) coordinates in meters
            
        Returns:
            (N, 1) source term in UAVs/(m³·s)
        """
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"
        
        # Check if points are outside target region
        sdf_target = self.target_shape.sdf(x)  # (N, 1)
        outside_target = (sdf_target > 0.0).float()  # 1 if outside, 0 if inside
        
        if self.scenario == 'homing':
            # Uniform source in Ω_free \ D: q(x) = q0 if outside target, else 0
            q_base = torch.full((x.shape[0], 1), self.q0, dtype=x.dtype, device=x.device)
            return q_base * outside_target
        
        elif self.scenario == 'p2p':
            # Source sphere in Ω_free \ D: q(x) = q_source if x in sphere AND outside target
            self.source_center = self.source_center.to(x.device).to(x.dtype)
            dist_to_center = torch.sqrt(torch.sum((x - self.source_center)**2, dim=1, keepdim=True))
            inside_sphere = (dist_to_center <= self.source_radius).float()
            # Only inject if outside target region
            return inside_sphere * self.q_source * outside_target
        
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
