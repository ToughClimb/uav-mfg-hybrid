"""
Wind fields (analytical).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import math


class WindField(ABC):
    """Abstract base class."""
    
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate wind velocity at points `x` (N, 3)."""
        pass


class ZeroWind(WindField):
    """v_w = 0."""
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"
        return torch.zeros_like(x)


class UniformWind(WindField):
    """v_w = constant vector."""
    
    def __init__(self, velocity: Tuple[float, float, float]):
        """`velocity` is (vx, vy, vz) in m/s."""
        self.velocity = torch.tensor(velocity, dtype=torch.float64).reshape(1, 3)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"
        self.velocity = self.velocity.to(x.device).to(x.dtype)
        return self.velocity.expand(x.shape[0], 3)


class VortexWind(WindField):
    """Vortex wind around z-axis.

    v_w(x,y,z) = s * g(r) * h(Δz) * (-Δy, Δx, 0) / (r + r_core)
    g(r) = (r/r0) * exp(1 - r/r0)
    h(Δz) = exp(-Δz^2 / (2*z0^2))
    r = sqrt(Δx^2 + Δy^2)
    """
    
    def __init__(self, center: Tuple[float, float, float], strength: float):
        """`center` in meters; `strength` in m/s."""
        self.center = torch.tensor(center, dtype=torch.float64).reshape(1, 3)
        self.strength = float(strength)

        self.r0 = 25.0
        self.z0 = 12.5
        self.r_core = 1.0
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"
        
        self.center = self.center.to(x.device).to(x.dtype)
        
        delta = x - self.center  # (N, 3)
        dx = delta[:, 0:1]  # (N, 1)
        dy = delta[:, 1:2]  # (N, 1)
        dz = delta[:, 2:3]  # (N, 1)

        r = torch.sqrt(dx**2 + dy**2)  # (N, 1)

        g_r = (r / self.r0) * torch.exp(1.0 - r / self.r0)

        h_z = torch.exp(-dz**2 / (2.0 * self.z0**2))

        tangent_x = -dy / (r + self.r_core)
        tangent_y = dx / (r + self.r_core)
        tangent_z = torch.zeros_like(dx)

        v_w = self.strength * g_r * h_z * torch.cat([tangent_x, tangent_y, tangent_z], dim=1)

        return v_w


class HeightDependentWind(WindField):
    """Height-dependent wind.

    v_w(z) = v0 * exp(-alpha * (z - z_min) / H), H = z_max - z_min.
    """
    
    def __init__(self, v0: Tuple[float, float, float], alpha: float,
                 z_min: float = 0.0, z_max: float = 50.0):
        """`v0` in m/s; `alpha` dimensionless."""
        self.v0 = torch.tensor(v0, dtype=torch.float64).reshape(1, 3)
        self.alpha = float(alpha)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.H = self.z_max - self.z_min
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"
        
        self.v0 = self.v0.to(x.device).to(x.dtype)
        
        z = x[:, 2:3]  # (N, 1)
        
        f_z = torch.exp(-self.alpha * (z - self.z_min) / self.H)
        v_w = self.v0 * f_z

        return v_w


class RegionConstantWind(WindField):
    """Constant wind patch inside a region (hard/smooth transition)."""
    
    def __init__(self, region_shape, value: Tuple[float, float, float],
                 combine: str = 'override', transition: str = 'hard',
                 sharpness: float = 50.0):
        """Initialize a patch."""
        self.region_shape = region_shape
        self.value = torch.tensor(value, dtype=torch.float64).reshape(1, 3)
        self.combine = combine
        self.transition = transition
        self.sharpness = float(sharpness)
        
        assert combine in ['override', 'add'], f"combine must be 'override' or 'add', got {combine}"
        assert transition in ['hard', 'smooth'], f"transition must be 'hard' or 'smooth', got {transition}"
    
    def indicator(self, x: torch.Tensor) -> torch.Tensor:
        sdf = self.region_shape.sdf(x)  # (N, 1)

        if self.transition == 'hard':
            return (sdf <= 0.0).float()
        else:  # smooth
            return torch.sigmoid(-self.sharpness * sdf)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("RegionConstantWind should be used within CompositeWind")


class CompositeWind(WindField):
    """Composite wind: base + patches."""
    
    def __init__(self, base: WindField, patches: Optional[list] = None):
        self.base = base
        self.patches = patches if patches is not None else []
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"

        v_w = self.base(x)  # (N, 3)

        for patch in self.patches:
            indicator = patch.indicator(x)  # (N, 1)
            patch.value = patch.value.to(x.device).to(x.dtype)

            if patch.combine == 'override':
                v_w = v_w * (1.0 - indicator) + patch.value * indicator
            elif patch.combine == 'add':
                v_w = v_w + patch.value * indicator

        return v_w
