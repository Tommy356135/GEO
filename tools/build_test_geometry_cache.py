#!/usr/bin/env python3
"""Build the frozen geometry cache for all test tiles without reading labels."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_pipeline import atomic_write_json
from tools.prepare_eclair_geometry_features import build_tile

SOURCE_ROOT = ROOT / "datasets/processed/test"
CACHE_ROOT = ROOT / "datasets/geometry_features/test"
RESULT_ROOT = (
    ROOT / "results/exp1_geometry_concat_final_v2_resume_fp32val"
)


def main() -> int:
    sources = sorted(SOURCE_ROOT.glob("*.npz"))
    if len(sources) != 106:
        raise RuntimeError(f"expected 106 test sources, found {len(sources)}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    progress = RESULT_ROOT / "test_cache_progress.csv"
    existing = {}
    if progress.exists():
        with progress.open(newline="", encoding="utf-8") as handle:
            existing = {row["tile_name"]: row for row in csv.DictReader(handle)}
    fields = (
        "tile_index", "tile_name", "point_count", "runtime_seconds",
        "cache_size_bytes", "status",
    )
    rows = []
    for index, source in enumerate(sources, 1):
        out = CACHE_ROOT / f"{source.stem}_geometry.npz"
        result = build_tile(
            source, out, "test", chunk_size=8192,
            overwrite=False, resume=True,
        )
        row = {
            "tile_index": index,
            "tile_name": source.name,
            "point_count": result.get(
                "point_count", existing.get(source.name, {}).get("point_count", "")
            ),
            "runtime_seconds": result.get("runtime_seconds", ""),
            "cache_size_bytes": result.get("cache_size_bytes", out.stat().st_size),
            "status": result["status"],
        }
        rows.append(row)
        with progress.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"test_cache={index}/106 tile={source.name} status={row['status']}",
            flush=True,
        )
    atomic_write_json(
        RESULT_ROOT / "test_cache_build_summary.json",
        {
            "status": "complete",
            "source_split": "test",
            "source_tiles": len(sources),
            "cache_tiles": len(list(CACHE_ROOT.glob("*_geometry.npz"))),
            "geometry_definition_version": "geomorph_geometry_v2",
            "small_k": 16,
            "large_k": 64,
            "normalization_fit_split": "train",
            "test_labels_read": False,
            "tiles": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
