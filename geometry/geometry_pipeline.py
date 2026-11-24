"""Configuration, deterministic sampling, and multi-scale geometry orchestration."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

def _parse_scalar(text: str) -> Any:
    text = text.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return ast.literal_eval(text)
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Read the project's mapping/list YAML subset without adding PyYAML."""
    raw_lines: list[tuple[int, str, int]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise ValueError(f"tabs are not allowed in YAML indentation at line {line_number}")
        indent = len(raw) - len(raw.lstrip(" "))
        raw_lines.append((indent, raw.strip(), line_number))
    if not raw_lines:
        return {}

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        is_list = raw_lines[index][1].startswith("- ")
        container: Any = [] if is_list else {}
        while index < len(raw_lines):
            current_indent, content, line_number = raw_lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"unexpected indentation at line {line_number}")
            if is_list:
                if not content.startswith("- "):
                    break
                item = content[2:].strip()
                if not item:
                    raise ValueError(f"empty list item at line {line_number}")
                container.append(_parse_scalar(item))
                index += 1
                continue
            if content.startswith("- ") or ":" not in content:
                break
            key, raw_value = content.split(":", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"empty mapping key at line {line_number}")
            raw_value = raw_value.strip()
            index += 1
            if raw_value:
                container[key] = _parse_scalar(raw_value)
            else:
                if index >= len(raw_lines) or raw_lines[index][0] <= indent:
                    container[key] = {}
                else:
                    container[key], index = parse_block(index, raw_lines[index][0])
        return container, index

    parsed, final_index = parse_block(0, raw_lines[0][0])
    if final_index != len(raw_lines) or not isinstance(parsed, dict):
        raise ValueError(f"unsupported YAML structure in {path}")
    return parsed


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    def json_default(item: Any) -> Any:
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
    )


def atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def stable_seed(base_seed: int, class_id: int, tile_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{class_id}:{tile_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def sample_query_indices(
    classifications: np.ndarray,
    class_id: int,
    limit: int,
    seed: int,
) -> np.ndarray:
    """Select only query points of one class; this never constructs neighborhoods."""
    classifications = np.asarray(classifications)
    candidates = np.flatnonzero(classifications == class_id)
    if limit < 0:
        raise ValueError("sampling limit cannot be negative")
    if len(candidates) <= limit:
        return candidates.astype(np.int64, copy=False)
    generator = np.random.default_rng(seed)
    selected = generator.choice(candidates, size=limit, replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def compute_multiscale_geometry(
    points: np.ndarray,
    query_indices: np.ndarray,
    k_values: Sequence[int],
    *,
    minimum_valid_neighbors: int,
    epsilon: float,
    exclude_query_point: bool,
    batch_size: int,
    direction_consistency_scale: int = 32,
    direction_consistency_scales: Sequence[int] | None = None,
    tree: Any | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute all features using a tree built from the complete same-tile cloud."""
    from geometry.direction_features import (
        direction_consistency_from_references,
        verticality_horizontality,
    )
    from geometry.eigen_features import compute_eigen_features
    from geometry.neighborhood import build_tree, query_knn

    points = np.asarray(points, dtype=np.float64)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    if tree is None:
        tree = build_tree(points)
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for k in k_values:
        knn = query_knn(
            tree,
            points,
            query_indices,
            int(k),
            exclude_query_point=exclude_query_point,
            batch_size=batch_size,
        )
        eigen = compute_eigen_features(
            points,
            knn.indices,
            knn.neighbor_count,
            minimum_valid_neighbors=minimum_valid_neighbors,
            epsilon=epsilon,
        )
        verticality, horizontality = verticality_horizontality(
            eigen.principal_directions, eigen.degenerate_neighborhood
        )
        radius_invalid = (~np.isfinite(knn.knn_radius)) | (knn.knn_radius <= epsilon)
        density = np.full(len(query_indices), np.nan, dtype=np.float64)
        density_valid = ~radius_invalid & (knn.neighbor_count > 0)
        density[density_valid] = knn.neighbor_count[density_valid] / (
            (4.0 / 3.0) * math.pi * np.power(knn.knn_radius[density_valid], 3) + epsilon
        )
        log_density = np.full(len(query_indices), np.nan, dtype=np.float64)
        log_density[density_valid] = np.log1p(density[density_valid])
        degenerate = eigen.degenerate_neighborhood | radius_invalid

        direction_consistency = np.full(len(query_indices), np.nan, dtype=np.float64)
        insufficient_direction = np.ones(len(query_indices), dtype=bool)
        enabled_direction_scales = (
            {int(value) for value in direction_consistency_scales}
            if direction_consistency_scales is not None
            else {int(direction_consistency_scale)}
        )
        if int(k) in enabled_direction_scales and len(query_indices):
            reference_indices = np.unique(knn.indices[knn.indices >= 0])
            reference_knn = query_knn(
                tree,
                points,
                reference_indices,
                int(k),
                exclude_query_point=exclude_query_point,
                batch_size=batch_size,
            )
            reference_eigen = compute_eigen_features(
                points,
                reference_knn.indices,
                reference_knn.neighbor_count,
                minimum_valid_neighbors=minimum_valid_neighbors,
                epsilon=epsilon,
            )
            direction_consistency, insufficient_direction = (
                direction_consistency_from_references(
                    eigen.principal_directions,
                    knn.indices,
                    reference_indices,
                    reference_eigen.principal_directions,
                    minimum_valid_directions=2,
                )
            )

        feature_map = {name: values.copy() for name, values in eigen.values.items()}
        feature_map.update(
            {
                "neighbor_count": knn.neighbor_count.copy(),
                "knn_radius": knn.knn_radius.copy(),
                "verticality": verticality,
                "horizontality": horizontality,
                "direction_consistency": direction_consistency,
                "direction_consistency_insufficient": insufficient_direction,
                "density": density,
                "log_density": log_density,
                "density_degenerate": radius_invalid,
                "degenerate_neighborhood": degenerate,
            }
        )
        outputs[int(k)] = feature_map
    return outputs
