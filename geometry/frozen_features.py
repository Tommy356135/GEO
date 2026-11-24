"""Frozen GeoMorph geometry ordering, fallback, and normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


FEATURE_NAMES = (
    "linearity",
    "planarity",
    "curvature",
    "verticality",
    "direction_consistency",
    "log_density",
)


def finalize_geometry_features(
    feature_map: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply v1 zero fallbacks and return the non-model audit validity mask."""
    matrix = np.column_stack([feature_map[name] for name in FEATURE_NAMES]).astype(
        np.float64, copy=False
    )
    geometry_valid = np.isfinite(matrix).all(axis=1)
    matrix = matrix.copy()
    matrix[~np.isfinite(matrix)] = 0.0
    if not np.isfinite(matrix).all():
        raise ValueError("geometry fallback failed to produce finite values")
    return matrix, geometry_valid


def load_frozen_normalization(
    path: Path | str = "configs/geometry/geomorph_normalization.json",
) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_geometry(
    raw_features: np.ndarray,
    normalization: Mapping,
) -> np.ndarray:
    """Apply frozen train parameters; this interface deliberately cannot refit."""
    raw = np.asarray(raw_features, dtype=np.float64)
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    std = np.asarray(normalization["std"], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(mean) or len(std) != len(mean):
        raise ValueError("geometry and normalization dimensions differ")
    if np.any(std < 1.0e-8) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("invalid frozen normalization parameters")
    result = (raw - mean) / (std + float(normalization["epsilon"]))
    result = np.clip(
        result,
        float(normalization["clip_min"]),
        float(normalization["clip_max"]),
    )
    if not np.isfinite(result).all():
        raise ValueError("normalized geometry is non-finite")
    return result
