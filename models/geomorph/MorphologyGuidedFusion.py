"""Coordinate-preserving residual geometry gate."""
from __future__ import annotations
import torch
from torch import nn
import MinkowskiEngine as ME

class MorphologyGuidedFusion(nn.Module):
    def __init__(self, semantic_channels: int, geometry_channels: int) -> None:
        super().__init__()
        self.eta = nn.Linear(geometry_channels, semantic_channels)
        self.psi = nn.Linear(geometry_channels, semantic_channels)
        # Small nonzero projection keeps the residual near identity while allowing
        # geometry-encoder gradients from the first diagnostic step.
        nn.init.normal_(self.eta.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.eta.bias)
        nn.init.zeros_(self.psi.weight); nn.init.constant_(self.psi.bias, -4.0)
    def forward(self, semantic: ME.SparseTensor, geometry: ME.SparseTensor) -> ME.SparseTensor:
        if semantic.coordinate_map_key != geometry.coordinate_map_key:
            raise ValueError("semantic and geometry sparse tensors do not share coordinate map key")
        residual = self.eta(geometry.F)
        gate = torch.sigmoid(self.psi(geometry.F))
        return ME.SparseTensor(semantic.F + gate * residual,
                               coordinate_map_key=semantic.coordinate_map_key,
                               coordinate_manager=semantic.coordinate_manager)
