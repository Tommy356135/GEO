#!/usr/bin/env python3
"""Read-only integrity audit for the frozen Phase 4C validation geometry cache."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_pipeline import atomic_write_csv, atomic_write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_for(source: Path, cache: Path, expected: dict[str, str]) -> dict[str, Any]:
    errors: list[str] = []
    result: dict[str, Any] = {
        "tile_name": source.name,
        "source_path": str(source),
        "cache_path": str(cache),
        "status": "invalid",
        "source_voxels": "",
        "geometry_rows": "",
        "source_hash_match": False,
        "coordinates_hash_match": False,
        "point_index_ok": False,
        "raw_small_shape_ok": False,
        "raw_large_shape_ok": False,
        "normalized_shape_ok": False,
        "valid_mask_ok": False,
        "finite_ok": False,
        "normalized_range_ok": False,
        "geometry_version_ok": False,
        "normalization_hash_ok": False,
        "errors": "",
    }
    if not cache.exists():
        result["status"] = "missing"
        result["errors"] = "missing_cache"
        return result
    try:
        with np.load(source, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates"], dtype=np.int32)
            source_voxels = len(coordinates)
        with np.load(cache, allow_pickle=False) as archive:
            required = {
                "point_indices", "geometry_raw_small", "geometry_raw_large",
                "geometry_normalized", "geometry_valid_small",
                "geometry_valid_large", "metadata",
            }
            absent = sorted(required.difference(archive.files))
            if absent:
                raise KeyError(f"missing_keys:{','.join(absent)}")
            metadata = json.loads(str(archive["metadata"].item()))
            point_indices = np.asarray(archive["point_indices"])
            raw_small = np.asarray(archive["geometry_raw_small"])
            raw_large = np.asarray(archive["geometry_raw_large"])
            normalized = np.asarray(archive["geometry_normalized"])
            valid_small = np.asarray(archive["geometry_valid_small"])
            valid_large = np.asarray(archive["geometry_valid_large"])

        geometry_rows = int(normalized.shape[0]) if normalized.ndim else 0
        result["source_voxels"] = source_voxels
        result["geometry_rows"] = geometry_rows
        checks = {
            "source_hash_match": metadata.get("source_npz_sha256") == file_sha256(source),
            "coordinates_hash_match": metadata.get("source_coordinates_sha256")
            == hashlib.sha256(coordinates.tobytes()).hexdigest(),
            "point_index_ok": point_indices.dtype == np.int64
            and point_indices.shape == (source_voxels,)
            and np.array_equal(point_indices, np.arange(source_voxels, dtype=np.int64)),
            "raw_small_shape_ok": raw_small.shape == (source_voxels, 6),
            "raw_large_shape_ok": raw_large.shape == (source_voxels, 6),
            "normalized_shape_ok": normalized.shape == (source_voxels, 12),
            "valid_mask_ok": valid_small.shape == (source_voxels,)
            and valid_large.shape == (source_voxels,)
            and valid_small.dtype == np.bool_
            and valid_large.dtype == np.bool_,
            "finite_ok": all(
                np.isfinite(value).all()
                for value in (raw_small, raw_large, normalized)
            ),
            "normalized_range_ok": bool(
                normalized.size
                and np.nanmin(normalized) >= -5.0
                and np.nanmax(normalized) <= 5.0
            ),
            "geometry_version_ok": metadata.get("geometry_definition_version")
            == expected["geometry_version"],
            "normalization_hash_ok": metadata.get("normalization_sha256")
            == expected["normalization_hash"],
        }
        result.update(checks)
        if int(metadata.get("point_count", -1)) != source_voxels:
            errors.append("metadata_point_count")
        if geometry_rows != source_voxels:
            errors.append("geometry_rows")
        errors.extend(name for name, passed in checks.items() if not passed)
        result["status"] = "valid" if not errors else "invalid"
    except Exception as error:
        errors.append(f"{type(error).__name__}:{error}")
    result["errors"] = ";".join(errors)
    return result


def main() -> int:
    source_root = ROOT / "datasets/processed/val"
    cache_root = ROOT / "datasets/geometry_features/val"
    output_root = ROOT / "results/geomorph_phase4c"
    sources = sorted(source_root.glob("*.npz"))
    caches = sorted(cache_root.glob("*_geometry.npz"))
    normalization = ROOT / "configs/geometry/geomorph_normalization.json"
    expected = {
        "geometry_version": "geomorph_geometry_v2",
        "normalization_hash": file_sha256(normalization),
    }
    rows = [
        row_for(source, cache_root / f"{source.stem}_geometry.npz", expected)
        for source in sources
    ]
    source_stems = {path.stem for path in sources}
    unexpected = [
        path.name for path in caches
        if path.name.removesuffix("_geometry.npz") not in source_stems
    ]
    tmp_files = sorted(
        str(path.relative_to(cache_root))
        for path in cache_root.rglob("*")
        if path.is_file() and (path.suffix == ".tmp" or ".tmp." in path.name)
    )
    valid = sum(row["status"] == "valid" for row in rows)
    missing = sum(row["status"] == "missing" for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    summary = {
        "status": "passed"
        if len(sources) == 59 and len(caches) == 59 and valid == 59
        and not unexpected and not tmp_files
        else "failed",
        "source_tiles": len(sources),
        "cache_tiles": len(caches),
        "valid_cache_tiles": valid,
        "missing_tiles": missing,
        "invalid_tiles": invalid,
        "hash_mismatches": sum(
            not row["source_hash_match"] or not row["coordinates_hash_match"]
            for row in rows if row["status"] != "missing"
        ),
        "nonfinite_values": sum(
            not row["finite_ok"] for row in rows if row["status"] != "missing"
        ),
        "shape_errors": sum(
            not (
                row["raw_small_shape_ok"]
                and row["raw_large_shape_ok"]
                and row["normalized_shape_ok"]
                and row["valid_mask_ok"]
            )
            for row in rows if row["status"] != "missing"
        ),
        "point_index_errors": sum(
            not row["point_index_ok"] for row in rows if row["status"] != "missing"
        ),
        "geometry_rows": sum(
            int(row["geometry_rows"]) for row in rows if row["geometry_rows"] != ""
        ),
        "unexpected_cache_files": unexpected,
        "tmp_files": tmp_files,
        "geometry_definition_version": expected["geometry_version"],
        "normalization_sha256": expected["normalization_hash"],
        "test_accessed": False,
    }
    fields = list(rows[0]) if rows else ["tile_name", "status", "errors"]
    atomic_write_csv(output_root / "val_cache_integrity.csv", fields, rows)
    atomic_write_json(output_root / "val_cache_integrity_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
