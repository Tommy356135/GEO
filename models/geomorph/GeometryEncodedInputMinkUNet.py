import torch
import MinkowskiEngine as ME
from models.backbone.minkowski.MinkUNet34C import MinkUNet34C
from models.geomorph.GeometryEncoder import GeometryEncoder
class GeometryEncodedInputMinkUNet(MinkUNet34C):
    """B2: frozen geometry is encoded to 16 channels then concatenated."""
    def __init__(self, **kwargs):
        super().__init__(in_channels=22, **{k:v for k,v in kwargs.items() if k!='in_channels'})
        self.geometry_encoder = GeometryEncoder()
    def forward(self, sparse_input, geometry):
        merged = ME.SparseTensor(torch.cat((sparse_input.F, self.geometry_encoder(geometry)), dim=1),
                                 coordinate_map_key=sparse_input.coordinate_map_key,
                                 coordinate_manager=sparse_input.coordinate_manager)
        return super().forward(merged)
