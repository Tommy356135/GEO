"""PyTorch Dataset for prepared ECLAIR MinkowskiEngine NPZ tiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Keep diagnostics/imports explicit before PyTorch is installed.
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """Temporary base used only to provide a clear dependency error."""


class EclairDataset(Dataset):
    """Load one already-voxelized ECLAIR tile per item.

    Coordinates have no batch column. Batch coordinates are produced exclusively
    by MinkowskiEngine's SparseCollation.
    """

    def __init__(
        self,
        processed_root: Path | str,
        split: str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError(
                "EclairDataset requires PyTorch. PyTorch is not installed in the "
                "current environment; follow the reviewed environment plan first."
            )
        if split is not None and split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test, or None")
        self.processed_root = Path(processed_root).expanduser().resolve()
        self.split = split
        self.transform = transform
        self.split_root = self.processed_root if split is None else self.processed_root / split
        self.files = sorted(self.split_root.glob("*.npz"))
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"processed split directory does not exist: {self.split_root}")
        if not self.files:
            raise FileNotFoundError(f"no NPZ tiles found in: {self.split_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates"], dtype=np.int32)
            features = np.asarray(archive["features"], dtype=np.float32)
            raw_labels = np.asarray(archive["raw_labels"], dtype=np.int64)
            if "labels" in archive.files:
                train_labels = np.asarray(archive["labels"], dtype=np.int64)
                metadata = json.loads(str(archive["metadata"].item()))
                ignore_index = -100
                raw_class_ids = np.arange(1, 12, dtype=np.int64)
                effective_class_ids = np.arange(9, dtype=np.int64)
                tile_name = str(metadata["tile_name"])
                stored_split = str(metadata.get("split", "train"))
            else:
                train_labels = np.asarray(archive["train_labels"], dtype=np.int64)
                ignore_index = int(archive["ignore_index"].item())
                raw_class_ids = np.asarray(archive["raw_class_ids"], dtype=np.int64)
                effective_class_ids = np.asarray(
                    archive["effective_class_ids"], dtype=np.int64
                )
                tile_name = str(archive["tile_name"].item())
                stored_split = str(archive["split"].item())
        if self.split is not None and stored_split != self.split:
            raise ValueError(
                f"processed tile split mismatch for {path}: {stored_split!r} != {self.split!r}"
            )
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(f"invalid coordinate shape in {path}: {coordinates.shape}")
        if features.ndim != 2 or len(features) != len(coordinates):
            raise ValueError(f"invalid feature shape in {path}: {features.shape}")
        if raw_labels.ndim != 1 or len(raw_labels) != len(coordinates):
            raise ValueError(f"invalid raw label shape in {path}: {raw_labels.shape}")
        if train_labels.ndim != 1 or len(train_labels) != len(coordinates):
            raise ValueError(f"invalid train label shape in {path}: {train_labels.shape}")
        if not np.all(np.isin(raw_labels, raw_class_ids)):
            raise ValueError(f"raw_labels contain values outside stored raw_class_ids: {path}")
        valid_train_ids = np.arange(len(effective_class_ids), dtype=np.int64)
        if not np.all(np.isin(train_labels, np.append(valid_train_ids, ignore_index))):
            raise ValueError(
                f"train_labels contain values outside configured IDs and ignore_index: {path}"
            )
        sample: dict[str, Any] = {
            "coordinates": torch.from_numpy(coordinates),
            "features": torch.from_numpy(features),
            "raw_labels": torch.from_numpy(raw_labels),
            "train_labels": torch.from_numpy(train_labels),
            # Training code consumes `labels`; retain explicit arrays for auditing.
            "labels": torch.from_numpy(train_labels),
            "ignore_index": ignore_index,
            "tile_name": tile_name,
        }
        return self.transform(sample) if self.transform is not None else sample


def build_sparse_collate_fn() -> Any:
    """Return MinkowskiEngine SparseCollation without implementing custom batching."""
    try:
        import MinkowskiEngine as ME
    except ImportError as error:
        raise ImportError(
            "MinkowskiEngine is required for sparse batching but is not installed. "
            "Do not substitute a custom batch coordinate implementation."
        ) from error
    sparse_collation = ME.utils.SparseCollation()

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Adapt sample dictionaries while delegating batching entirely to ME."""
        coordinates, features, labels = sparse_collation(
            [
                (sample["coordinates"], sample["features"], sample["labels"])
                for sample in samples
            ]
        )
        return {
            "coordinates": coordinates,
            "features": features,
            "labels": labels,
            "tile_name": [sample["tile_name"] for sample in samples],
        }

    return collate


def load_processing_summary(processed_root: Path | str) -> dict[str, Any]:
    """Read conversion provenance for experiment checks."""
    path = Path(processed_root) / "processing_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))
