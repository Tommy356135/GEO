"""Same-tile KNN neighborhood construction without semantic-label filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class KNNResult:
    """Batched KNN output; invalid padded neighbors use index -1 and distance inf."""

    indices: np.ndarray
    distances: np.ndarray
    neighbor_count: np.ndarray
    knn_radius: np.ndarray


def build_tree(points: np.ndarray) -> cKDTree:
    """Build one tree from all finite XYZ points in a single tile."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if len(points) == 0:
        raise ValueError("cannot build a KNN tree for an empty tile")
    if not np.isfinite(points).all():
        raise ValueError("tile coordinates contain non-finite values")
    return cKDTree(points)


def query_knn(
    tree: cKDTree,
    points: np.ndarray,
    query_indices: np.ndarray,
    k: int,
    *,
    exclude_query_point: bool = False,
    batch_size: int = 4096,
) -> KNNResult:
    """Query KNN from the same tile in bounded batches.

    Semantic labels are deliberately absent from this interface. Query indices and
    neighbors both refer to ``points`` belonging to one tile.
    """
    points = np.asarray(points, dtype=np.float64)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if query_indices.ndim != 1:
        raise ValueError("query_indices must be one-dimensional")
    if k <= 0 or batch_size <= 0:
        raise ValueError("k and batch_size must be positive")
    if np.any(query_indices < 0) or np.any(query_indices >= len(points)):
        raise IndexError("query index is outside the current tile")

    target_k = min(k, max(0, len(points) - (1 if exclude_query_point else 0)))
    if target_k == 0:
        shape = (len(query_indices), 0)
        return KNNResult(
            indices=np.empty(shape, dtype=np.int64),
            distances=np.empty(shape, dtype=np.float64),
            neighbor_count=np.zeros(len(query_indices), dtype=np.int32),
            knn_radius=np.full(len(query_indices), np.nan, dtype=np.float64),
        )

    requested_k = min(len(points), target_k + (1 if exclude_query_point else 0))
    all_indices = np.full((len(query_indices), target_k), -1, dtype=np.int64)
    all_distances = np.full((len(query_indices), target_k), np.inf, dtype=np.float64)

    for start in range(0, len(query_indices), batch_size):
        stop = min(start + batch_size, len(query_indices))
        batch_query_indices = query_indices[start:stop]
        distances, indices = tree.query(
            points[batch_query_indices], k=requested_k, workers=1
        )
        if requested_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        for local_row, query_index in enumerate(batch_query_indices):
            valid = np.isfinite(distances[local_row]) & (indices[local_row] < len(points))
            row_indices = indices[local_row][valid]
            row_distances = distances[local_row][valid]
            if exclude_query_point:
                keep = row_indices != query_index
                row_indices = row_indices[keep]
                row_distances = row_distances[keep]
            count = min(target_k, len(row_indices))
            all_indices[start + local_row, :count] = row_indices[:count]
            all_distances[start + local_row, :count] = row_distances[:count]

    valid = (all_indices >= 0) & np.isfinite(all_distances)
    neighbor_count = valid.sum(axis=1).astype(np.int32)
    knn_radius = np.full(len(query_indices), np.nan, dtype=np.float64)
    for row_index, count in enumerate(neighbor_count):
        if count:
            knn_radius[row_index] = all_distances[row_index, count - 1]
    return KNNResult(all_indices, all_distances, neighbor_count, knn_radius)
