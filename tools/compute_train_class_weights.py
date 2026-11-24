#!/usr/bin/env python3
"""Compute preregistered balanced-CE weights from the train split only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.geometry_pipeline import atomic_write_csv, atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("datasets/processed/train"))
    parser.add_argument(
        "--class-mapping", type=Path, default=Path("configs/dataset/class_mapping.json")
    )
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--clip-min", type=float, default=0.5)
    parser.add_argument("--clip-max", type=float, default=5.0)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/baseline_diagnostics")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import numpy as np

    paths = sorted(args.train_root.resolve().glob("*.npz"))
    if len(paths) != 413:
        raise ValueError(f"expected 413 train NPZ files, found {len(paths)}")
    counts = np.zeros(9, dtype=np.int64)
    ignore_count = 0
    tile_counts = np.zeros(9, dtype=np.int64)
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            labels = np.asarray(archive["labels"], dtype=np.int64)
        ignore_count += int((labels == -100).sum())
        valid = labels[labels != -100]
        current = np.bincount(valid, minlength=9)
        counts += current
        tile_counts += current > 0
    mapping = json.loads(args.class_mapping.read_text(encoding="utf-8"))
    names = [mapping["train_to_name"][str(index)] for index in range(9)]
    frequencies = counts / counts.sum()
    raw = 1.0 / np.sqrt(frequencies + args.epsilon)
    mean_normalized = raw / raw.mean()
    clipped = np.clip(mean_normalized, args.clip_min, args.clip_max)
    rows = [
        {
            "class_id": class_id,
            "class_name": names[class_id],
            "train_points": int(counts[class_id]),
            "train_tiles": int(tile_counts[class_id]),
            "class_frequency": float(frequencies[class_id]),
            "raw_weight": float(raw[class_id]),
            "mean_normalized_weight": float(mean_normalized[class_id]),
            "final_clipped_weight": float(clipped[class_id]),
        }
        for class_id in range(9)
    ]
    report = {
        "source_split": "train_only",
        "train_npz_count": len(paths),
        "valid_point_count": int(counts.sum()),
        "ignore_point_count": ignore_count,
        "ignore_index": -100,
        "epsilon": args.epsilon,
        "formula": "raw_weight_c = 1 / sqrt(class_frequency_c + epsilon)",
        "mean_normalization": "divide raw weights by their arithmetic mean",
        "clip_range": [args.clip_min, args.clip_max],
        "class_ids": list(range(9)),
        "raw_weights": raw.tolist(),
        "mean_normalized_weights": mean_normalized.tolist(),
        "final_weights": clipped.tolist(),
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_root / "train_class_statistics.csv", tuple(rows[0]), rows)
    atomic_write_json(output_root / "train_class_weights.json", report)
    print(json.dumps({"statistics": rows, "weights": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
