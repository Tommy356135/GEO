"""Small label-independent encoder for frozen 12-D geometry."""
from __future__ import annotations
import torch
from torch import nn

class GeometryEncoder(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 32, output_dim: int = 16,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim), nn.ReLU(inplace=True),
        )
    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.net(geometry)
