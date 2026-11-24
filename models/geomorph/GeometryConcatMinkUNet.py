from models.backbone.minkowski.MinkUNet34C import MinkUNet34C
class GeometryConcatMinkUNet(MinkUNet34C):
    """B1: only input concatenation, with the unchanged residual U-Net."""
    def __init__(self, **kwargs):
        super().__init__(in_channels=18, **{k:v for k,v in kwargs.items() if k!='in_channels'})
