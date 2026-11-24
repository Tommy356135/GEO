# ECLAIR Data Protocol

## Approved ground-truth partition

- 413 training tiles
- 59 validation tiles
- 106 held-out final-evaluation tiles
- 1,410 trainable 50 m merged block samples

The original ECLAIR distribution contains additional files on disk. The count
of 413 refers only to the approved ground-truth training partition used by this
paper; it is not the size of the entire dataset.

Valid raw labels `[2, 3, 4, 6, 7, 8, 9, 10, 11]` map to train labels
`[0, 1, 2, 3, 4, 5, 6, 7, 8]`. Raw labels 1 (Unassigned) and 5 (Noise) are
ignored with training ignore index `-100`.

## Directory template

```text
<DATASET_ROOT>/
├── pointclouds/
└── labels.json

<PROJECT_ROOT>/datasets/
├── processed/{train,val,test}/
├── block_indices_50m_merged/{train,val,test}/
└── geometry_features/{train,val,test}/
```

## Acquisition and license

Raw ECLAIR data are not redistributed in this package. Users should obtain the
dataset from its official source under the original license. The dataset is
identified in the manuscript bibliography as Melekhov et al., “ECLAIR: A
High-Fidelity Aerial LiDAR Dataset for Semantic Segmentation,” CVPR Workshops,
2024, DOI 10.1109/CVPRW63382.2024.00758. This package does not make a new
redistribution-rights claim.

## Integrity

The copied metadata manifest, split filename lists, mapping file, block merge
integrity report, and train/validation/held-out geometry-cache audit reports
provide the available integrity trail. No raw point-cloud file is included or
read during package assembly.
