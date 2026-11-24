import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.backbone.minkowski.MinkUNet34C import MinkUNet34C
from models.geomorph.GeometryEncoder import GeometryEncoder
from models.geomorph.MorphologyGuidedFusion import MorphologyGuidedFusion

class GeoMorphMinkUNet(MinkUNet34C):
    """GeoMorph-Seg: residual geometry gates after encoder1 and encoder2."""
    def __init__(self, **kwargs):
        super().__init__(in_channels=6, **{k:v for k,v in kwargs.items() if k!='in_channels'})
        self.geometry_encoder = GeometryEncoder()
        self.geometry_stem = nn.Sequential(
            ME.MinkowskiConvolution(16,32,kernel_size=5,dimension=3),
            ME.MinkowskiBatchNorm(32), ME.MinkowskiReLU(inplace=True))
        self.geometry_down1 = ME.MinkowskiConvolution(32,32,kernel_size=2,stride=2,dimension=3)
        self.geometry_down2 = ME.MinkowskiConvolution(32,64,kernel_size=2,stride=2,dimension=3)
        self.fusion1 = MorphologyGuidedFusion(32,32)
        self.fusion2 = MorphologyGuidedFusion(64,64)
    def forward(self, sparse_input, geometry):
        encoded = self.geometry_encoder(geometry)
        gs = ME.SparseTensor(encoded, coordinate_map_key=sparse_input.coordinate_map_key,
                             coordinate_manager=sparse_input.coordinate_manager)
        semantic = self.stem(sparse_input)
        geom = self.geometry_stem(gs)
        geom1 = self.geometry_down1(geom)
        e1 = self.encoder1(self.down1(semantic))
        e1 = self.fusion1(e1, geom1)
        geom2 = self.geometry_down2(geom1)
        e2 = self.encoder2(self.down2(e1))
        e2 = self.fusion2(e2, geom2)
        e3=self.encoder3(self.down3(e2)); e4=self.encoder4(self.down4(e3))
        return self.decode_features({'stem':semantic,'encoder1':e1,'encoder2':e2,'encoder3':e3,'encoder4':e4})
