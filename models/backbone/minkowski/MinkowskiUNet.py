"""Minimal MinkowskiEngine U-Net baseline without geometry enhancements."""

from __future__ import annotations

from typing import Any

try:
    import torch.nn as nn
    import MinkowskiEngine as ME
except ImportError:
    nn = None  # type: ignore[assignment]
    ME = None  # type: ignore[assignment]


if nn is not None and ME is not None:

    class MinkowskiUNet(nn.Module):
        """Small encoder-decoder producing per-voxel semantic logits."""

        def __init__(
            self,
            in_channels: int,
            num_classes: int,
            base_channels: int = 32,
            dimension: int = 3,
        ) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels, base_channels, kernel_size=3, stride=1, dimension=dimension
                ),
                ME.MinkowskiBatchNorm(base_channels),
                ME.MinkowskiReLU(inplace=True),
            )
            self.encoder = nn.Sequential(
                ME.MinkowskiConvolution(
                    base_channels, base_channels * 2, kernel_size=2, stride=2, dimension=dimension
                ),
                ME.MinkowskiBatchNorm(base_channels * 2),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(
                    base_channels * 2, base_channels * 2, kernel_size=3, dimension=dimension
                ),
                ME.MinkowskiBatchNorm(base_channels * 2),
                ME.MinkowskiReLU(inplace=True),
            )
            self.decoder = ME.MinkowskiGenerativeConvolutionTranspose(
                base_channels * 2,
                base_channels,
                kernel_size=2,
                stride=2,
                dimension=dimension,
            )
            self.decoder_post = nn.Sequential(
                ME.MinkowskiBatchNorm(base_channels),
                ME.MinkowskiReLU(inplace=True),
            )
            self.fuse = nn.Sequential(
                ME.MinkowskiConvolution(
                    base_channels * 2, base_channels, kernel_size=3, dimension=dimension
                ),
                ME.MinkowskiBatchNorm(base_channels),
                ME.MinkowskiReLU(inplace=True),
            )
            self.classifier = ME.MinkowskiConvolution(
                base_channels, num_classes, kernel_size=1, bias=True, dimension=dimension
            )

        def forward(self, sparse_input: Any) -> Any:
            skip = self.stem(sparse_input)
            encoded = self.encoder(skip)
            decoded = self.decoder_post(
                self.decoder(encoded, coordinates=skip.coordinate_map_key)
            )
            fused = self.fuse(ME.cat(decoded, skip))
            return self.classifier(fused)

else:

    class MinkowskiUNet:  # type: ignore[no-redef]
        """Dependency guard used until PyTorch and MinkowskiEngine are installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError(
                "MinkowskiUNet requires both PyTorch and MinkowskiEngine. "
                "Neither dependency is installed automatically by this project."
            )
