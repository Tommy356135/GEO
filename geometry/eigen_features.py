"""Covariance eigenfeatures for local point-cloud neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "lambda_1",
    "lambda_2",
    "lambda_3",
    "linearity",
    "planarity",
    "scattering",
    "anisotropy",
    "omnivariance",
    "eigenentropy",
    "curvature",
)


@dataclass(slots=True)
class EigenFeatureResult:
    values: dict[str, np.ndarray]
    principal_directions: np.ndarray
    neighbor_count: np.ndarray
    degenerate_neighborhood: np.ndarray


def compute_eigen_features(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_count: np.ndarray,
    *,
    minimum_valid_neighbors: int = 8,
    epsilon: float = 1.0e-12,
) -> EigenFeatureResult:
    """Compute population-covariance eigenfeatures.

    Degenerate or non-finite neighborhoods remain NaN and are explicitly marked;
    they are never silently replaced with zero.
    """
    points = np.asarray(points, dtype=np.float64)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    neighbor_count = np.asarray(neighbor_count, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if neighbor_indices.ndim != 2 or len(neighbor_indices) != len(neighbor_count):
        raise ValueError("neighbor arrays have incompatible shapes")
    if minimum_valid_neighbors < 1 or epsilon <= 0:
        raise ValueError("minimum_valid_neighbors and epsilon must be positive")

    size = len(neighbor_indices)
    values = {name: np.full(size, np.nan, dtype=np.float64) for name in FEATURE_NAMES}
    directions = np.full((size, 3), np.nan, dtype=np.float64)
    degenerate = np.ones(size, dtype=bool)

    for row_index in range(size):
        count = int(neighbor_count[row_index])
        if count < minimum_valid_neighbors:
            continue
        indices = neighbor_indices[row_index, :count]
        if np.any(indices < 0) or np.any(indices >= len(points)):
            continue
        neighborhood = points[indices]
        if not np.isfinite(neighborhood).all():
            continue
        centered = neighborhood - neighborhood.mean(axis=0)
        covariance = centered.T @ centered / float(count)
        if not np.isfinite(covariance).all():
            continue
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        eigenvectors = eigenvectors[:, order]
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= epsilon:
            continue

        lambda_1, lambda_2, lambda_3 = (float(value) for value in eigenvalues)
        eigen_sum = lambda_1 + lambda_2 + lambda_3
        normalized = eigenvalues / (eigen_sum + epsilon)
        feature_row = {
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "lambda_3": lambda_3,
            "linearity": (lambda_1 - lambda_2) / (lambda_1 + epsilon),
            "planarity": (lambda_2 - lambda_3) / (lambda_1 + epsilon),
            "scattering": lambda_3 / (lambda_1 + epsilon),
            "anisotropy": (lambda_1 - lambda_3) / (lambda_1 + epsilon),
            "omnivariance": float(np.cbrt(lambda_1 * lambda_2 * lambda_3)),
            "eigenentropy": float(-np.sum(normalized * np.log(normalized + epsilon))),
            "curvature": lambda_3 / (eigen_sum + epsilon),
        }
        if not all(np.isfinite(value) for value in feature_row.values()):
            continue
        for name, value in feature_row.items():
            values[name][row_index] = value
        directions[row_index] = eigenvectors[:, 0]
        degenerate[row_index] = False

    return EigenFeatureResult(values, directions, neighbor_count.copy(), degenerate)
