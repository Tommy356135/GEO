"""Principal-direction features with sign-invariant local comparisons."""

from __future__ import annotations

import numpy as np


def verticality_horizontality(
    principal_directions: np.ndarray,
    degenerate: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    directions = np.asarray(principal_directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("principal_directions must have shape (n, 3)")
    valid = np.isfinite(directions).all(axis=1)
    if degenerate is not None:
        valid &= ~np.asarray(degenerate, dtype=bool)
    verticality = np.full(len(directions), np.nan, dtype=np.float64)
    horizontality = np.full(len(directions), np.nan, dtype=np.float64)
    verticality[valid] = np.clip(np.abs(directions[valid, 2]), 0.0, 1.0)
    horizontality[valid] = np.sqrt(
        np.maximum(0.0, 1.0 - np.square(verticality[valid]))
    )
    return verticality, horizontality


def sign_invariant_consistency(
    direction: np.ndarray, reference_directions: np.ndarray
) -> float:
    """Mean |v_i dot v_j|, invariant to eigenvector sign flips."""
    direction = np.asarray(direction, dtype=np.float64)
    references = np.asarray(reference_directions, dtype=np.float64)
    if direction.shape != (3,) or references.ndim != 2 or references.shape[1] != 3:
        raise ValueError("direction shapes must be (3,) and (n, 3)")
    valid = np.isfinite(references).all(axis=1)
    if not np.isfinite(direction).all() or not valid.any():
        return float("nan")
    dots = np.abs(references[valid] @ direction)
    return float(np.mean(np.clip(dots, 0.0, 1.0)))


def direction_consistency_from_references(
    query_directions: np.ndarray,
    query_neighbor_indices: np.ndarray,
    reference_point_indices: np.ndarray,
    reference_directions: np.ndarray,
    *,
    minimum_valid_directions: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare query directions with local sub-neighborhood directions.

    Reference directions are indexed by point index only; no semantic labels enter
    this computation.
    """
    query_directions = np.asarray(query_directions, dtype=np.float64)
    query_neighbor_indices = np.asarray(query_neighbor_indices, dtype=np.int64)
    reference_point_indices = np.asarray(reference_point_indices, dtype=np.int64)
    reference_directions = np.asarray(reference_directions, dtype=np.float64)
    if len(reference_point_indices) != len(reference_directions):
        raise ValueError("reference point indices and directions differ in length")
    lookup = {int(point_index): row for row, point_index in enumerate(reference_point_indices)}
    values = np.full(len(query_directions), np.nan, dtype=np.float64)
    insufficient = np.ones(len(query_directions), dtype=bool)
    for row_index, direction in enumerate(query_directions):
        if not np.isfinite(direction).all():
            continue
        reference_rows = [
            lookup[int(point_index)]
            for point_index in query_neighbor_indices[row_index]
            if int(point_index) in lookup
        ]
        if not reference_rows:
            continue
        local_directions = reference_directions[reference_rows]
        valid = np.isfinite(local_directions).all(axis=1)
        if int(valid.sum()) < minimum_valid_directions:
            continue
        values[row_index] = sign_invariant_consistency(direction, local_directions[valid])
        insufficient[row_index] = False
    return values, insufficient
