#!/usr/bin/env python3
"""Audit all frozen test geometry caches against their source tiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_pipeline import atomic_write_json
from tools.audit_val_geometry_cache import file_sha256, row_for

SOURCE_ROOT = ROOT / "datasets/processed/test"
CACHE_ROOT = ROOT / "datasets/geometry_features/test"
REPORT = (
    ROOT / "results/exp1_geometry_concat_final_v2_resume_fp32val"
    / "test_cache_integrity_report.json"
)


def main() -> int:
    sources = sorted(SOURCE_ROOT.glob("*.npz"))
    caches = sorted(CACHE_ROOT.glob("*_geometry.npz"))
    expected = {
        "geometry_version": "geomorph_geometry_v2",
        "normalization_hash": file_sha256(
            ROOT / "configs/geometry/geomorph_normalization.json"
        ),
    }
    rows = [
        row_for(
            source, CACHE_ROOT / f"{source.stem}_geometry.npz", expected
        )
        for source in sources
    ]
    source_stems = {source.stem for source in sources}
    unexpected = [
        path.name for path in caches
        if path.name.removesuffix("_geometry.npz") not in source_stems
    ]
    tmp_files = sorted(
        str(path.relative_to(CACHE_ROOT))
        for path in CACHE_ROOT.rglob("*")
        if path.is_file() and (path.suffix == ".tmp" or ".tmp." in path.name)
    )
    valid = sum(row["status"] == "valid" for row in rows)
    missing = sum(row["status"] == "missing" for row in rows)
    invalid = sum(row["status"] == "invalid" for row in rows)
    summary = {
        "status": "passed"
        if len(sources) == len(caches) == valid == 106
        and not unexpected and not tmp_files else "failed",
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
            not row["finite_ok"] for row in rows
            if row["status"] != "missing"
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
            not row["point_index_ok"] for row in rows
            if row["status"] != "missing"
        ),
        "geometry_rows": sum(
            int(row["geometry_rows"])
            for row in rows if row["geometry_rows"] != ""
        ),
        "unexpected_cache_files": unexpected,
        "tmp_files": tmp_files,
        "geometry_definition_version": expected["geometry_version"],
        "normalization_sha256": expected["normalization_hash"],
        "small_k": 16,
        "large_k": 64,
        "test_labels_used_for_cache": False,
    }
    atomic_write_json(REPORT, {"summary": summary, "tiles": rows})
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
