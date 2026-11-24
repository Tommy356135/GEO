import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.backbone.minkowski.MinkUNet34C import MinkUNet34C
from models.geomorph.GeometryEncoder import GeometryEncoder
from models.geomorph.MorphologyGuidedFusion import MorphologyGuidedFusion
class GeometrySingleFusionMinkUNet(MinkUNet34C):
    """B3: one residual gate after encoder1, with explicit sparse alignment."""
    def __init__(self, **kwargs):
        super().__init__(in_channels=6, **{k:v for k,v in kwargs.items() if k!='in_channels'})
        self.geometry_encoder = GeometryEncoder()
        self.geometry_stem = nn.Sequential(
            ME.MinkowskiConvolution(16,32,kernel_size=5,dimension=3),
            ME.MinkowskiBatchNorm(32), ME.MinkowskiReLU(inplace=True))
        self.geometry_down1 = ME.MinkowskiConvolution(32,32,kernel_size=2,stride=2,dimension=3)
        self.fusion1 = MorphologyGuidedFusion(32,32)
    def forward(self, sparse_input, geometry):
        encoded = self.geometry_encoder(geometry)
        gs = ME.SparseTensor(encoded, coordinate_map_key=sparse_input.coordinate_map_key,
                             coordinate_manager=sparse_input.coordinate_manager)
        semantic = self.stem(sparse_input)
        geom = self.geometry_down1(self.geometry_stem(gs))
        e1 = self.encoder1(self.down1(semantic))
        e1 = self.fusion1(e1, geom)
        e2=self.encoder2(self.down2(e1)); e3=self.encoder3(self.down3(e2)); e4=self.encoder4(self.down4(e3))
        return self.decode_features({'stem':semantic,'encoder1':e1,'encoder2':e2,'encoder3':e3,'encoder4':e4})
