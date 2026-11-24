# GeoMorph-Seg

Research code for **GeoMorph-Seg: Gated Residual Geometry Fusion for Low-Altitude Aerial LiDAR Semantic Segmentation**.

GeoMorph-Seg combines a sparse MinkUNet backbone with multiscale local geometry descriptors and gated residual fusion. The experiments compare a semantic baseline, raw and encoded geometry concatenation, and single- and dual-level fusion. On the ECLAIR test split, GeoMorph-SF achieved 52.62% mIoU, compared with 44.64% for the semantic baseline.

## Dataset

Experiments use the public [ECLAIR aerial LiDAR dataset](https://github.com/SharperShape/eclair-dataset). Download the point clouds and labels from the official source under its dataset license. See the [data protocol](docs/dataset_protocol.md) for the experimental split and directory layout.

## Code

- `models/` and `geometry/` contain the network and geometry descriptors.
- `datasets_loader/`, `engine/`, and `tools/` contain data loading, preprocessing, training, and evaluation code.
- `configs/` contains the experiment settings.

See [the command index](docs/command_index.md) for the historical experiment entry points. Set your dataset paths in the configuration and provide the dataset manifest before preprocessing. The environment files record the CUDA 12.1 and MinkowskiEngine setup. Exp1 uses a historical resume workflow and requires its prior checkpoint.
