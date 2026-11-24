"""Lossless block views over formal ECLAIR tile NPZ files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class EclairBlockDataset(Dataset):
    """Extract indexed voxels without copying or re-voxelizing source tiles."""

    def __init__(
        self,
        processed_root: Path | str = "datasets/processed",
        index_root: Path | str = "datasets/block_indices",
        split: str = "train",
        geometry_mode: str = "none",
        geometry_cache_root: Path | str | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.processed_root = Path(processed_root).resolve()
        self.index_root = Path(index_root).resolve()
        self.split = split
        if geometry_mode not in {"none", "raw_multiscale", "normalized_multiscale"}:
            raise ValueError("invalid geometry_mode")
        self.geometry_mode = geometry_mode
        self.geometry_cache_root = (
            Path(geometry_cache_root).resolve()
            if geometry_cache_root is not None else None
        )
        if geometry_mode != "none" and self.geometry_cache_root is None:
            raise ValueError("geometry_cache_root is required when geometry is enabled")
        self._source_hashes: dict[Path, str] = {}
        self.index_files = sorted((self.index_root / split).glob("*.npz"))
        if split == "train":
            trainable = []
            for path in self.index_files:
                with np.load(path, allow_pickle=False) as block:
                    if int(block["valid_label_count"].item()) > 0:
                        trainable.append(path)
            self.index_files = trainable
        self.tile_paths = {
            path.stem: path for path in (self.processed_root / split).glob("*.npz")
        }
        if not self.index_files:
            raise FileNotFoundError(f"no block indices for split={split}")

    def _sha256(self, path: Path) -> str:
        if path not in self._source_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._source_hashes[path] = digest.hexdigest()
        return self._source_hashes[path]

    def _load_geometry(
        self, tile_stem: str, source_path: Path, point_indices: np.ndarray
    ) -> dict[str, torch.Tensor]:
        assert self.geometry_cache_root is not None
        nested = self.geometry_cache_root / self.split / f"{tile_stem}_geometry.npz"
        direct = self.geometry_cache_root / f"{tile_stem}_geometry.npz"
        cache_path = nested if nested.exists() else direct
        if not cache_path.exists():
            raise FileNotFoundError(f"geometry cache is missing: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_indices = np.asarray(cache["point_indices"], dtype=np.int64)
            metadata = json.loads(str(cache["metadata"].item()))
            if metadata["source_npz_sha256"] != self._sha256(source_path):
                raise ValueError(f"geometry source hash mismatch: {cache_path}")
            if not np.array_equal(
                cached_indices, np.arange(len(cached_indices), dtype=np.int64)
            ):
                raise ValueError(f"geometry cache row mapping is not identity: {cache_path}")
            if np.any(point_indices >= len(cached_indices)):
                raise ValueError(f"geometry point index is out of range: {cache_path}")
            if self.geometry_mode == "raw_multiscale":
                small = np.asarray(cache["geometry_raw_small"], dtype=np.float32)[point_indices]
                large = np.asarray(cache["geometry_raw_large"], dtype=np.float32)[point_indices]
                return {
                    "geometry_small": torch.from_numpy(small),
                    "geometry_large": torch.from_numpy(large),
                }
            geometry = np.asarray(
                cache["geometry_normalized"], dtype=np.float32
            )[point_indices]
            return {"geometry": torch.from_numpy(geometry)}

    def __len__(self) -> int:
        return len(self.index_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index_path = self.index_files[index]
        with np.load(index_path, allow_pickle=False) as block:
            tile_name = str(block["tile_name"].item())
            tile_stem = str(block["tile_stem"].item())
            block_id = str(block["block_id"].item())
            merged_sample_id = (
                str(block["merged_sample_id"].item())
                if "merged_sample_id" in block.files
                else block_id
            )
            source_block_ids = (
                [str(value) for value in block["source_block_ids"].tolist()]
                if "source_block_ids" in block.files
                else [block_id]
            )
            point_indices = np.asarray(block["point_indices"], dtype=np.int64)
            origin = np.asarray(block["block_origin"], dtype=np.int32)
            stored_split = str(block["split"].item())
            voxel_count = int(block["voxel_count"].item())
            valid_label_count = int(block["valid_label_count"].item())
            block_size = float(block["block_size"].item())
            level4_active_sites = (
                int(block["level4_active_sites"].item())
                if "level4_active_sites" in block.files
                else None
            )
        if stored_split != self.split:
            raise ValueError(f"index split mismatch: {index_path}")
        source_path = self.tile_paths[tile_stem]
        with np.load(source_path, allow_pickle=False) as tile:
            coordinates = np.asarray(tile["coordinates"], dtype=np.int32)
            features = np.asarray(tile["features"], dtype=np.float32)
            labels = np.asarray(tile["labels"], dtype=np.int64)
            raw_labels = np.asarray(tile["raw_labels"], dtype=np.int64)
        if len(point_indices) != voxel_count or np.any(point_indices >= len(coordinates)):
            raise ValueError(f"invalid point indices: {index_path}")
        local_coordinates = coordinates[point_indices].copy()
        local_coordinates -= origin
        if local_coordinates.dtype != np.int32:
            raise TypeError("block coordinates must remain int32")
        block_features = features[point_indices]
        block_labels = labels[point_indices]
        if block_features.shape != (voxel_count, 6):
            raise ValueError("baseline block features must be [N,6]")
        if int(np.count_nonzero(block_labels != -100)) != valid_label_count:
            raise ValueError(f"block valid-label count mismatch: {index_path}")
        result = {
            "coordinates": torch.from_numpy(local_coordinates),
            "features": torch.from_numpy(block_features),
            "labels": torch.from_numpy(block_labels),
            "train_labels": torch.from_numpy(block_labels),
            "raw_labels": torch.from_numpy(raw_labels[point_indices]),
            "point_indices": torch.from_numpy(point_indices),
            "tile_name": tile_name,
            "block_id": block_id,
            "merged_sample_id": merged_sample_id,
            "source_block_ids": source_block_ids,
            "block_origin": torch.from_numpy(origin),
            "block_size": block_size,
            "present_classes": [
                int(value)
                for value in np.unique(block_labels[block_labels != -100]).tolist()
            ],
            "level4_active_sites": level4_active_sites,
        }
        if self.geometry_mode != "none":
            result.update(self._load_geometry(tile_stem, source_path, point_indices))
        return result


def build_block_sparse_collate_fn() -> Any:
    """Collate lossless blocks while retaining their audit metadata."""
    import MinkowskiEngine as ME

    sparse_collation = ME.utils.SparseCollation()

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        coordinates, features, labels = sparse_collation(
            [
                (sample["coordinates"], sample["features"], sample["labels"])
                for sample in samples
            ]
        )
        result = {
            "coordinates": coordinates,
            "features": features,
            "labels": labels,
            "tile_name": [sample["tile_name"] for sample in samples],
            "block_id": [sample["block_id"] for sample in samples],
            "merged_sample_id": [sample["merged_sample_id"] for sample in samples],
            "source_block_ids": [sample["source_block_ids"] for sample in samples],
            "level4_active_sites": [sample["level4_active_sites"] for sample in samples],
            "present_classes": [sample["present_classes"] for sample in samples],
            "point_indices": [sample["point_indices"] for sample in samples],
        }
        if "geometry" in samples[0]:
            result["geometry"] = torch.cat(
                [sample["geometry"] for sample in samples], dim=0
            )
        if "geometry_small" in samples[0]:
            result["geometry_small"] = torch.cat(
                [sample["geometry_small"] for sample in samples], dim=0
            )
            result["geometry_large"] = torch.cat(
                [sample["geometry_large"] for sample in samples], dim=0
            )
        return result

    return collate
