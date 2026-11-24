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

## Experimental protocol

The paper uses 413 training tiles, 59 validation tiles, and 106 held-out test tiles from ECLAIR. Preprocessing builds 50 m blocks and multiscale geometry descriptors. Geometry normalization uses the training split. The validation split selects the checkpoint before held-out evaluation.

| Configuration | Geometry integration |
| --- | --- |
| Exp0 · Baseline | Semantic input features |
| Exp1 · Raw Concat | Raw geometry descriptors concatenated at the input |
| Exp2 · Encoded Concat | Learned geometry embedding concatenated at the input |
| Exp3 · GeoMorph-SF | Gated residual fusion at one level |
| Exp4 · GeoMorph-DF | Gated residual fusion at two levels |

## Running the code

From the repository root, review the CUDA 12.1 environment definition and the MinkowskiEngine setup scripts. Replace the dataset path placeholders in `configs/dataset/eclair_minkowski.yaml` and place the ECLAIR manifest at the path specified in that configuration. Follow the preprocessing and cache-building steps in the [command index](docs/command_index.md) before starting training.

After preparing the training and validation data, the GeoMorph-SF training entry point is

```bash
python tools/train_geomorph_experiment.py --experiment exp3 --output-dir results/exp3_single_fusion_final --epochs 100
```

The command index also documents the baseline and ablation entry points. The Figure 6 scripts in `results/paper_phase7_revision/` use validation data and require the corresponding experiment artifacts and local paths.
