"""
Density model for end-to-end PINN (Route A comparison).
Strictly follows experimental configuration section 4.3.
"""

import torch
import torch.nn as nn
from .mlp import MLP


class RhoModel(nn.Module):
    """
    Density network with sigmoid output activation.
    
    Implements:
    rho(x) = rho_max * sigmoid(nn_rho(x_nn))
    """
    
    def __init__(self, hidden_layers: list, rho_max: float, domain_bounds: tuple):
        """
        Args:
            hidden_layers: list of hidden layer widths
            rho_max: maximum density in UAVs/m³
            domain_bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        """
        super().__init__()
        
        self.rho_max = float(rho_max)
        self.domain_bounds = domain_bounds
        
        # Neural network
        self.nn_rho = MLP(
            input_dim=3,
            hidden_layers=hidden_layers,
            output_dim=1,
            activation='tanh',
            dtype=torch.float64
        )
        
        # Store domain bounds for normalization
        self.register_buffer('x_min', torch.tensor([domain_bounds[0][0], 
                                                     domain_bounds[1][0], 
                                                     domain_bounds[2][0]], dtype=torch.float64))
        self.register_buffer('x_max', torch.tensor([domain_bounds[0][1], 
                                                     domain_bounds[1][1], 
                                                     domain_bounds[2][1]], dtype=torch.float64))
    
    def normalize_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [-1, 1]."""
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (N, 3) physical coordinates
            
        Returns:
            (N, 1) density values in [0, rho_max]
        """
        x_nn = self.normalize_coords(x)
        nn_out = self.nn_rho(x_nn)
        rho = self.rho_max * torch.sigmoid(nn_out)
        return rho
