"""
Multi-layer perceptron with configurable architecture.
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Fully-connected neural network with tanh activation.
    Follows experimental configuration section 4.2.
    """
    
    def __init__(self, input_dim: int, hidden_layers: List[int], 
                 output_dim: int, activation: str = 'tanh',
                 dtype: torch.dtype = torch.float64):
        """
        Args:
            input_dim: input dimension (typically 3 for normalized coordinates)
            hidden_layers: list of hidden layer widths
            output_dim: output dimension (1 for scalar fields)
            activation: activation function ('tanh' recommended for PINN)
            dtype: data type (float64 recommended for PINN gradient stability)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dtype = dtype
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Xavier initialization
        self._initialize_weights()
        
        # Convert to specified dtype
        self.to(dtype)
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (N, input_dim) input tensor
            
        Returns:
            (N, output_dim) output tensor
        """
        return self.network(x)
