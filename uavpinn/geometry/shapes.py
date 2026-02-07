"""SDF-based shapes for target regions and obstacles."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class Shape(ABC):
    """Geometric shape with SDF."""
    
    @abstractmethod
    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """Signed distance (positive outside, negative inside, zero on surface)."""
        pass
    
    def distance_nonneg(self, x: torch.Tensor) -> torch.Tensor:
        """Non-negative distance: max(sdf(x), 0)."""
        return torch.clamp(self.sdf(x), min=0.0)
    
    def normal(self, x_surface: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Outward unit normal at surface points via normalized ∇sdf."""
        x_surface = x_surface.clone().requires_grad_(True)
        sdf_val = self.sdf(x_surface)
        grad_sdf = torch.autograd.grad(
            outputs=sdf_val,
            inputs=x_surface,
            grad_outputs=torch.ones_like(sdf_val),
            create_graph=True,
            retain_graph=True
        )[0]
        norm = torch.sqrt(torch.sum(grad_sdf**2, dim=1, keepdim=True) + eps**2)
        return grad_sdf / norm


class Sphere(Shape):
    """Sphere: |x-center| - radius."""
    
    def __init__(self, center: Tuple[float, float, float], radius: float):
        """`center` in meters; `radius` in meters."""
        self.center = torch.tensor(center, dtype=torch.float64).reshape(1, 3)
        self.radius = float(radius)
    
    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """SDF: |x-center| - radius."""
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"
        self.center = self.center.to(x.device).to(x.dtype)
        dist_to_center = torch.sqrt(torch.sum((x - self.center)**2, dim=1, keepdim=True))
        return dist_to_center - self.radius


class AABB(Shape):
    """Axis-Aligned Bounding Box."""
    
    def __init__(self, min_corner: Tuple[float, float, float], 
                 max_corner: Tuple[float, float, float]):
        """`min_corner`/`max_corner` in meters."""
        self.min_corner = torch.tensor(min_corner, dtype=torch.float64).reshape(1, 3)
        self.max_corner = torch.tensor(max_corner, dtype=torch.float64).reshape(1, 3)
        
        assert torch.all(self.max_corner > self.min_corner), \
            "max_corner must be greater than min_corner"
    
    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """SDF for AABB. Reference: https://iquilezles.org/articles/distfunctions/"""
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"
        
        self.min_corner = self.min_corner.to(x.device).to(x.dtype)
        self.max_corner = self.max_corner.to(x.device).to(x.dtype)
        
        center = (self.min_corner + self.max_corner) / 2.0
        half_size = (self.max_corner - self.min_corner) / 2.0
        
        q = torch.abs(x - center) - half_size
        
        outside_dist = torch.sqrt(
            torch.sum(torch.clamp(q, min=0.0)**2, dim=1, keepdim=True)
        )
        
        inside_dist = torch.clamp(torch.max(q, dim=1, keepdim=True)[0], max=0.0)
        
        return outside_dist + inside_dist


class Union(Shape):
    """Union of multiple shapes: sdf = min(sdf_i)."""
    
    def __init__(self, shapes: list):
        """`shapes` is a non-empty list of Shape."""
        assert len(shapes) > 0, "Union requires at least one shape"
        self.shapes = shapes
    
    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """SDF: min_i sdf_i."""
        assert x.shape[-1] == 3, f"Expected (N, 3) coordinates, got {x.shape}"
        
        sdf_values = torch.stack([shape.sdf(x) for shape in self.shapes], dim=-1)  # (N, 1, num_shapes)
        min_sdf = torch.min(sdf_values, dim=-1, keepdim=False)[0]  # (N, 1)
        return min_sdf.reshape(-1, 1)  # Ensure (N, 1) output shape
