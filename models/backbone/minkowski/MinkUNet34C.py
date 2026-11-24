"""Baseline-Strong residual four-level Minkowski U-Net without geometry modules."""

from __future__ import annotations

from typing import Any, Sequence

try:
    import torch.nn as nn
    import MinkowskiEngine as ME
except ImportError:
    nn = None  # type: ignore[assignment]
    ME = None  # type: ignore[assignment]


if nn is not None and ME is not None:

    class ResidualBlock(nn.Module):
        """Two-convolution sparse residual block."""

        def __init__(self, in_channels: int, out_channels: int, dimension: int) -> None:
            super().__init__()
            self.conv1 = ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=3, dimension=dimension
            )
            self.bn1 = ME.MinkowskiBatchNorm(out_channels)
            self.relu = ME.MinkowskiReLU(inplace=True)
            self.conv2 = ME.MinkowskiConvolution(
                out_channels, out_channels, kernel_size=3, dimension=dimension
            )
            self.bn2 = ME.MinkowskiBatchNorm(out_channels)
            self.projection = (
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels, out_channels, kernel_size=1, dimension=dimension
                    ),
                    ME.MinkowskiBatchNorm(out_channels),
                )
                if in_channels != out_channels else nn.Identity()
            )

        def forward(self, sparse_input: Any) -> Any:
            residual = self.projection(sparse_input)
            output = self.conv1(sparse_input)
            output = self.relu(self.bn1(output))
            output = self.bn2(self.conv2(output))
            return self.relu(output + residual)


    def residual_stage(
        in_channels: int, out_channels: int, blocks: int, dimension: int
    ) -> nn.Sequential:
        if blocks <= 0:
            raise ValueError("residual stage must contain at least one block")
        modules = [ResidualBlock(in_channels, out_channels, dimension)]
        modules.extend(
            ResidualBlock(out_channels, out_channels, dimension)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*modules)


    class MinkUNet34C(nn.Module):
        """Standard residual 4-level Sparse U-Net used as Baseline-Strong."""

        DEFAULT_PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
        DEFAULT_LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

        def __init__(
            self,
            in_channels: int = 6,
            num_classes: int = 9,
            dimension: int = 3,
            stem_channels: int = 32,
            planes: Sequence[int] = DEFAULT_PLANES,
            layers: Sequence[int] = DEFAULT_LAYERS,
        ) -> None:
            super().__init__()
            if len(planes) != 8 or len(layers) != 8:
                raise ValueError("MinkUNet34C requires eight plane and layer entries")
            p = tuple(int(value) for value in planes)
            depth = tuple(int(value) for value in layers)
            self.stem = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels, stem_channels, kernel_size=5, dimension=dimension
                ),
                ME.MinkowskiBatchNorm(stem_channels),
                ME.MinkowskiReLU(inplace=True),
            )
            self.down1 = ME.MinkowskiConvolution(
                stem_channels, p[0], kernel_size=2, stride=2, dimension=dimension
            )
            self.encoder1 = residual_stage(p[0], p[0], depth[0], dimension)
            self.down2 = ME.MinkowskiConvolution(
                p[0], p[1], kernel_size=2, stride=2, dimension=dimension
            )
            self.encoder2 = residual_stage(p[1], p[1], depth[1], dimension)
            self.down3 = ME.MinkowskiConvolution(
                p[1], p[2], kernel_size=2, stride=2, dimension=dimension
            )
            self.encoder3 = residual_stage(p[2], p[2], depth[2], dimension)
            self.down4 = ME.MinkowskiConvolution(
                p[2], p[3], kernel_size=2, stride=2, dimension=dimension
            )
            self.encoder4 = residual_stage(p[3], p[3], depth[3], dimension)

            self.up4 = ME.MinkowskiGenerativeConvolutionTranspose(
                p[3], p[4], kernel_size=2, stride=2, dimension=dimension
            )
            self.decoder4 = residual_stage(p[4] + p[2], p[4], depth[4], dimension)
            self.up3 = ME.MinkowskiGenerativeConvolutionTranspose(
                p[4], p[5], kernel_size=2, stride=2, dimension=dimension
            )
            self.decoder3 = residual_stage(p[5] + p[1], p[5], depth[5], dimension)
            self.up2 = ME.MinkowskiGenerativeConvolutionTranspose(
                p[5], p[6], kernel_size=2, stride=2, dimension=dimension
            )
            self.decoder2 = residual_stage(p[6] + p[0], p[6], depth[6], dimension)
            self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
                p[6], p[7], kernel_size=2, stride=2, dimension=dimension
            )
            self.decoder1 = residual_stage(
                p[7] + stem_channels, p[7], depth[7], dimension
            )
            self.classifier = ME.MinkowskiConvolution(
                p[7], num_classes, kernel_size=1, bias=True, dimension=dimension
            )

        def forward(self, sparse_input: Any) -> Any:
            features = self.forward_features(sparse_input)
            return self.decode_features(features)

        def forward_features(self, sparse_input: Any) -> dict[str, Any]:
            stem = self.stem(sparse_input)
            encoder1 = self.encoder1(self.down1(stem))
            encoder2 = self.encoder2(self.down2(encoder1))
            encoder3 = self.encoder3(self.down3(encoder2))
            encoder4 = self.encoder4(self.down4(encoder3))
            return {
                "stem": stem, "encoder1": encoder1, "encoder2": encoder2,
                "encoder3": encoder3, "encoder4": encoder4,
            }

        def decode_features(self, features: dict[str, Any]) -> Any:
            stem = features["stem"]
            encoder1 = features["encoder1"]
            encoder2 = features["encoder2"]
            encoder3 = features["encoder3"]
            encoder4 = features["encoder4"]
            decoder4 = self.up4(encoder4, coordinates=encoder3.coordinate_map_key)
            decoder4 = self.decoder4(ME.cat(decoder4, encoder3))
            decoder3 = self.up3(decoder4, coordinates=encoder2.coordinate_map_key)
            decoder3 = self.decoder3(ME.cat(decoder3, encoder2))
            decoder2 = self.up2(decoder3, coordinates=encoder1.coordinate_map_key)
            decoder2 = self.decoder2(ME.cat(decoder2, encoder1))
            decoder1 = self.up1(decoder2, coordinates=stem.coordinate_map_key)
            decoder1 = self.decoder1(ME.cat(decoder1, stem))
            return self.classifier(decoder1)

else:

    class MinkUNet34C:  # type: ignore[no-redef]
        """Dependency guard for environments without MinkowskiEngine."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("MinkUNet34C requires PyTorch and MinkowskiEngine")
