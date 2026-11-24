#!/usr/bin/env python3
"""Build a lossless Block50 train partition with deterministic small-block merges."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.geometry_pipeline import atomic_write_csv, atomic_write_json  # noqa: E402

SOURCE_INDEX = Path("datasets/block_indices_50m")
OUTPUT_INDEX = Path("datasets/block_indices_50m_merged")
TILE_ROOT = Path("datasets/processed/train")
OCCUPANCY = Path("results/block50_smallblock/downsample_occupancy.csv")
RESULT_ROOT = Path("results/block50_smallblock")


def parse_block_id(block_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"x(-?\d+)_y(-?\d+)", block_id)
    if not match:
        raise ValueError(f"invalid block id: {block_id}")
    return tuple(map(int, match.groups()))  # type: ignore[return-value]


def main() -> int:
    if OUTPUT_INDEX.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_INDEX}")
    occupancy = {}
    with OCCUPANCY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            occupancy[(row["tile_name"], row["block_id"])] = row
    source_paths = sorted((SOURCE_INDEX / "train").glob("*.npz"))
    nodes_by_tile: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in source_paths:
        with np.load(path, allow_pickle=False) as block:
            tile_name = str(block["tile_name"].item())
            block_id = str(block["block_id"].item())
            nodes_by_tile[tile_name][block_id] = {
                "path": path,
                "block_id": block_id,
                "xy": parse_block_id(block_id),
                "voxel_count": int(block["voxel_count"].item()),
                "valid_label_count": int(block["valid_label_count"].item()),
                "level4": int(occupancy[(tile_name, block_id)]["level4_active_sites"]),
            }
    groups_by_tile: dict[str, list[set[str]]] = {
        tile: [{block_id} for block_id in sorted(nodes)]
        for tile, nodes in nodes_by_tile.items()
    }
    affected_small = {
        (tile, block_id)
        for tile, nodes in nodes_by_tile.items()
        for block_id, node in nodes.items()
        if node["valid_label_count"] > 0 and node["level4"] < 8
    }
    if len(affected_small) != 15:
        raise RuntimeError(f"expected 15 small train blocks, found {len(affected_small)}")

    def group_for(groups: list[set[str]], block_id: str) -> set[str]:
        return next(group for group in groups if block_id in group)

    def choose_neighbor(
        tile: str, source_group: set[str], groups: list[set[str]]
    ) -> set[str]:
        nodes = nodes_by_tile[tile]
        choices = []
        for target in groups:
            if target is source_group:
                continue
            if sum(nodes[name]["valid_label_count"] for name in target) == 0:
                continue
            pair_scores = []
            for source_id in source_group:
                sx, sy = nodes[source_id]["xy"]
                for target_id in target:
                    tx, ty = nodes[target_id]["xy"]
                    dx, dy = abs(sx - tx), abs(sy - ty)
                    if max(dx, dy) != 1:
                        continue
                    shared = 50.0 if dx + dy == 1 else 0.0
                    distance = math.hypot(dx, dy) * 50.0
                    pair_scores.append((-shared, distance))
            if not pair_scores:
                continue
            boundary_distance = min(pair_scores)
            combined = sum(
                nodes[name]["voxel_count"] for name in source_group | target
            )
            choices.append(
                (
                    boundary_distance[0],
                    boundary_distance[1],
                    combined,
                    ";".join(sorted(target)),
                    target,
                )
            )
        if not choices:
            raise RuntimeError(f"no same-tile neighbor for {tile}:{sorted(source_group)}")
        return min(choices, key=lambda item: item[:4])[4]

    # Every originally unsafe block deterministically absorbs/is absorbed by one neighbor.
    for tile, block_id in sorted(affected_small):
        groups = groups_by_tile[tile]
        source = group_for(groups, block_id)
        if len(source) > 1:
            continue
        target = choose_neighbor(tile, source, groups)
        source.update(target)
        groups.remove(target)

    tile_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    index_cache: dict[tuple[str, str], np.ndarray] = {}

    def load_tile(tile_name: str) -> tuple[np.ndarray, np.ndarray]:
        if tile_name not in tile_cache:
            stem = Path(tile_name).stem
            with np.load(TILE_ROOT / f"{stem}.npz", allow_pickle=False) as tile:
                tile_cache[tile_name] = (
                    np.asarray(tile["coordinates"], dtype=np.int32),
                    np.asarray(tile["labels"], dtype=np.int64),
                )
        return tile_cache[tile_name]

    def indices_for(tile: str, block_id: str) -> np.ndarray:
        key = (tile, block_id)
        if key not in index_cache:
            with np.load(nodes_by_tile[tile][block_id]["path"], allow_pickle=False) as block:
                index_cache[key] = np.asarray(block["point_indices"], dtype=np.int64)
        return index_cache[key]

    def merged_stats(tile: str, group: set[str]) -> dict:
        coordinates, labels = load_tile(tile)
        concatenated = np.concatenate([indices_for(tile, name) for name in sorted(group)])
        unique = np.unique(concatenated)
        if len(unique) != len(concatenated):
            raise RuntimeError(f"duplicate source indices in {tile}:{sorted(group)}")
        selected = coordinates[unique]
        origin = selected.min(axis=0).astype(np.int32)
        level4 = len(np.unique(np.floor_divide(selected - origin, 16), axis=0))
        block_labels = labels[unique]
        return {
            "point_indices": unique,
            "origin": origin,
            "level4": level4,
            "valid": int(np.count_nonzero(block_labels != -100)),
            "present": np.unique(block_labels[block_labels != -100]).tolist(),
        }

    # Defensive closure: if a group containing an unsafe source is still <8, merge again.
    while True:
        unresolved = None
        for tile, groups in groups_by_tile.items():
            for group in groups:
                if any((tile, name) in affected_small for name in group):
                    stats = merged_stats(tile, group)
                    if stats["valid"] > 0 and stats["level4"] < 8:
                        unresolved = (tile, group)
                        break
            if unresolved:
                break
        if unresolved is None:
            break
        tile, source = unresolved
        groups = groups_by_tile[tile]
        target = choose_neighbor(tile, source, groups)
        source.update(target)
        groups.remove(target)

    train_output = OUTPUT_INDEX / "train"
    train_output.mkdir(parents=True)
    merge_rows = []
    source_total = merged_total = 0
    unique_by_tile: dict[str, list[np.ndarray]] = defaultdict(list)
    trainable_level4 = []
    max_voxels = 0
    sample_count = 0
    for tile_name in sorted(groups_by_tile):
        tile_stem = Path(tile_name).stem
        for sample_number, group in enumerate(groups_by_tile[tile_name]):
            stats = merged_stats(tile_name, group)
            source_ids = sorted(group)
            sample_id = (
                source_ids[0]
                if len(source_ids) == 1
                else f"merged_{sample_number:03d}"
            )
            source_counts = [
                nodes_by_tile[tile_name][name]["voxel_count"] for name in source_ids
            ]
            target = train_output / f"{tile_stem}__{sample_id}.npz"
            np.savez_compressed(
                target,
                tile_name=np.asarray(tile_name),
                tile_stem=np.asarray(tile_stem),
                block_id=np.asarray(sample_id),
                merged_sample_id=np.asarray(sample_id),
                source_block_ids=np.asarray(source_ids),
                point_indices=stats["point_indices"],
                block_origin=stats["origin"],
                block_size=np.asarray(50.0, dtype=np.float32),
                split=np.asarray("train"),
                voxel_count=np.asarray(len(stats["point_indices"]), dtype=np.int64),
                valid_label_count=np.asarray(stats["valid"], dtype=np.int64),
                level4_active_sites=np.asarray(stats["level4"], dtype=np.int64),
            )
            sample_count += 1
            merged_total += len(stats["point_indices"])
            unique_by_tile[tile_name].append(stats["point_indices"])
            max_voxels = max(max_voxels, len(stats["point_indices"]))
            if stats["valid"] > 0:
                trainable_level4.append(stats["level4"])
            if len(source_ids) > 1:
                merge_rows.append(
                    {
                        "merged_sample_id": sample_id,
                        "tile_name": tile_name,
                        "source_block_ids": ";".join(source_ids),
                        "source_block_count": len(source_ids),
                        "source_voxel_counts": ";".join(map(str, source_counts)),
                        "merged_voxel_count": len(stats["point_indices"]),
                        "level4_before_min": min(
                            nodes_by_tile[tile_name][name]["level4"] for name in source_ids
                        ),
                        "level4_after": stats["level4"],
                        "present_classes": ";".join(map(str, stats["present"])),
                        "merge_reason": "source block level4_active_sites < 8",
                        "selected_neighbor_rule": "shared_boundary>center_distance>merged_voxels>block_id",
                    }
                )
    duplicate_voxels = missing_voxels = unique_total = 0
    for tile_name, parts in unique_by_tile.items():
        coordinates, _ = load_tile(tile_name)
        concatenated = np.concatenate(parts)
        unique = np.unique(concatenated)
        duplicate_voxels += len(concatenated) - len(unique)
        missing_voxels += len(coordinates) - len(unique)
        unique_total += len(unique)
        source_total += len(coordinates)
    if source_total != 126172603:
        raise RuntimeError(f"unexpected train voxel total: {source_total}")
    if duplicate_voxels or missing_voxels or merged_total != source_total:
        raise RuntimeError("merged train partition is not a lossless exact cover")
    if min(trainable_level4) < 8:
        raise RuntimeError("one or more trainable merged samples remain below level4=8")
    for split in ("val", "test"):
        destination = OUTPUT_INDEX / split
        destination.mkdir(parents=True)
        for source in sorted((SOURCE_INDEX / split).glob("*.npz")):
            shutil.copy2(source, destination / source.name)
    original_trainable = sum(
        node["valid_label_count"] > 0
        for nodes in nodes_by_tile.values()
        for node in nodes.values()
    )
    merged_trainable = sum(level >= 8 for level in trainable_level4)
    integrity = {
        "original_train_samples": original_trainable,
        "merged_train_samples": merged_trainable,
        "audit_index_samples_including_all_ignore": sample_count,
        "affected_small_blocks": len(affected_small),
        "affected_tiles": len({tile for tile, _ in affected_small}),
        "source_voxel_total": source_total,
        "merged_index_voxel_total": merged_total,
        "unique_voxel_total": unique_total,
        "duplicate_voxels": duplicate_voxels,
        "missing_voxels": missing_voxels,
        "min_level4_active_sites": min(trainable_level4),
        "all_samples_batchnorm_robust": min(trainable_level4) >= 8,
        "max_merged_voxels": max_voxels,
        "all_ignore_audit_samples": sum(
            node["valid_label_count"] == 0
            for nodes in nodes_by_tile.values()
            for node in nodes.values()
        ),
        "val_index_files": len(list((OUTPUT_INDEX / "val").glob("*.npz"))),
        "test_index_files": len(list((OUTPUT_INDEX / "test").glob("*.npz"))),
    }
    atomic_write_csv(RESULT_ROOT / "merge_manifest.csv", tuple(merge_rows[0]), merge_rows)
    atomic_write_json(RESULT_ROOT / "merge_integrity.json", integrity)
    print(json.dumps(integrity, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
