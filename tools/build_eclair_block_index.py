#!/usr/bin/env python3
"""Build compact, lossless point-index files for fixed ECLAIR XY blocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.geometry_pipeline import atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size-m", type=int, choices=(50, 75), default=75)
    parser.add_argument("--processed-root", type=Path, default=Path("datasets/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/block_indices"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    block_voxels = int(round(args.block_size_m / 0.1))
    summaries = {}
    for split, expected_tiles in (("train", 413), ("val", 59), ("test", 106)):
        paths = sorted((args.processed_root / split).resolve().glob("*.npz"))
        if len(paths) != expected_tiles:
            raise ValueError(f"{split}: expected {expected_tiles} tiles, found {len(paths)}")
        split_output = (args.output_root / split).resolve()
        split_output.mkdir(parents=True, exist_ok=True)
        if any(split_output.glob("*.npz")):
            raise FileExistsError(f"refusing to overwrite existing indices: {split_output}")
        source_voxels = indexed_voxels = 0
        index_files = trainable_blocks = all_ignore_blocks = 0
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                coordinates = np.asarray(archive["coordinates"], dtype=np.int32)
                labels = np.asarray(archive["labels"], dtype=np.int64)
                metadata = json.loads(str(archive["metadata"].item()))
            tile_name = str(metadata["tile_name"])
            x_bins = np.floor_divide(coordinates[:, 0], block_voxels)
            y_bins = np.floor_divide(coordinates[:, 1], block_voxels)
            pairs = np.column_stack((x_bins, y_bins))
            unique_pairs = np.unique(pairs, axis=0)
            tile_seen = np.zeros(len(coordinates), dtype=np.uint8)
            for x_bin, y_bin in unique_pairs:
                point_indices = np.flatnonzero(
                    (x_bins == x_bin) & (y_bins == y_bin)
                ).astype(np.int64)
                if not len(point_indices):
                    raise RuntimeError("empty block must not be indexed")
                if np.any(tile_seen[point_indices]):
                    raise RuntimeError(f"duplicate point assignment in {tile_name}")
                tile_seen[point_indices] = 1
                block_labels = labels[point_indices]
                valid_count = int(np.count_nonzero(block_labels != -100))
                all_ignore = valid_count == 0
                all_ignore_blocks += int(all_ignore)
                block_id = f"x{int(x_bin)}_y{int(y_bin)}"
                block_origin = np.asarray(
                    [
                        int(x_bin) * block_voxels,
                        int(y_bin) * block_voxels,
                        int(coordinates[point_indices, 2].min()),
                    ],
                    dtype=np.int32,
                )
                target = split_output / f"{path.stem}__{block_id}.npz"
                np.savez_compressed(
                    target,
                    tile_name=np.asarray(tile_name),
                    tile_stem=np.asarray(path.stem),
                    block_id=np.asarray(block_id),
                    point_indices=point_indices,
                    block_origin=block_origin,
                    block_size=np.asarray(args.block_size_m, dtype=np.float32),
                    split=np.asarray(split),
                    voxel_count=np.asarray(len(point_indices), dtype=np.int64),
                    valid_label_count=np.asarray(valid_count, dtype=np.int64),
                )
                index_files += 1
                trainable_blocks += int(valid_count > 0)
                indexed_voxels += len(point_indices)
            if not np.all(tile_seen == 1):
                raise RuntimeError(f"incomplete block mapping for {tile_name}")
            source_voxels += len(coordinates)
        if indexed_voxels != source_voxels:
            raise RuntimeError(f"{split} index coverage is not 100%")
        summaries[split] = {
            "tiles": len(paths),
            "index_files": index_files,
            "trainable_blocks": trainable_blocks,
            "all_ignore_blocks": all_ignore_blocks,
            "source_voxels": source_voxels,
            "indexed_voxels": indexed_voxels,
            "mapping_complete": indexed_voxels == source_voxels,
        }
    report = {
        "block_size_m": args.block_size_m,
        "voxel_size_m": 0.1,
        "no_revoxelization": True,
        "no_point_dropping_within_indexed_blocks": True,
        "splits": summaries,
    }
    report_path = (
        Path("results/block50_memory/index_integrity.json")
        if args.block_size_m == 50
        else Path("results/block_partition_analysis/block_index_summary.json")
    )
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
