#!/usr/bin/env python3
"""Prepare immutable ECLAIR LAZ tiles as voxelized MinkowskiEngine NPZ inputs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.geometry_pipeline import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    canonical_hash,
    load_simple_yaml,
)


CONVERSION_VERSION = 2
FAILED_FIELDS = (
    "tile_name", "file_path", "split", "status", "backend_attempted",
    "error_type", "error_message",
)


class LazReadError(RuntimeError):
    """Preserve both backend attempts at the single-file boundary."""

    def __init__(self, attempted: list[str], messages: list[str]) -> None:
        super().__init__(" | ".join(messages))
        self.attempted = attempted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/dataset/eclair_minkowski.yaml"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--splits", default=None,
        help="Comma-separated official splits to prepare (default: config train,val,test).",
    )
    parser.add_argument("--max-files", type=int, help="Development-only file limit.")
    parser.add_argument(
        "--include-tiles",
        help="Optional comma-separated tile names; no unrequested tile is added.",
    )
    parser.add_argument(
        "--feature-mode",
        help="Registered feature mode from config (default: default_feature_mode).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print the exact selection without reading or writing NPZ files.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace incompatible processed outputs; compatible outputs are always resumed.",
    )
    args = parser.parse_args()
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive")
    if args.splits:
        args.splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if args.include_tiles:
        args.include_tiles = [
            value.strip() for value in args.include_tiles.split(",") if value.strip()
        ]
    return args


def is_true(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def require_dependencies() -> tuple[Any, Any] | None:
    try:
        import laspy  # type: ignore
        import lazrs  # noqa: F401  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as error:
        print(
            "Error: conversion requires only numpy, laspy, and lazrs. "
            f"Missing dependency: {error}",
            file=sys.stderr,
        )
        return None
    return laspy, np


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("labels.json must contain a list")
    output: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("tile_name"):
            raise ValueError(f"invalid labels.json record at index {index}")
        tile_name = str(item["tile_name"])
        if tile_name in output:
            raise ValueError(f"duplicate labels.json tile: {tile_name}")
        output[tile_name] = item
    return output


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tile_name", "file_path", "file_exists", "in_labels_json", "split", "usable"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    counts = Counter(row["tile_name"] for row in rows)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"manifest has duplicate tile names: {duplicates[:10]}")
    return rows


def select_rows(
    rows: Sequence[dict[str, str]],
    labels: dict[str, dict[str, Any]],
    splits: Sequence[str],
    include_tiles: Sequence[str] | None,
    max_files: int | None,
) -> list[dict[str, str]]:
    official = {"train", "val", "test"}
    if not splits or any(split not in official for split in splits):
        raise ValueError("splits must be a non-empty subset of train,val,test")
    selected = [
        row for row in rows
        if is_true(row["usable"])
        and is_true(row["file_exists"])
        and is_true(row["in_labels_json"])
        and row["split"] in splits
    ]
    for row in selected:
        metadata = labels.get(row["tile_name"])
        if metadata is None:
            raise ValueError(f"usable manifest tile absent from labels.json: {row['tile_name']}")
        if str(metadata.get("split", "")) != row["split"]:
            raise ValueError(
                f"official split mismatch for {row['tile_name']}: "
                f"manifest={row['split']} labels.json={metadata.get('split')}"
            )
    if include_tiles is not None:
        requested = list(dict.fromkeys(include_tiles))
        by_name = {row["tile_name"]: row for row in selected}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"requested tiles are absent or unusable: {missing}")
        selected = [by_name[name] for name in requested]
    else:
        selected.sort(key=lambda row: (splits.index(row["split"]), row["tile_name"]))
    if include_tiles is not None:
        return selected
    return selected[:max_files] if max_files is not None else selected


def read_laz(path: Path, laspy: Any) -> tuple[Any, list[str], str, list[str]]:
    failures: list[str] = []
    attempted: list[str] = []
    for backend_name, backend in (
        ("LazrsParallel", laspy.LazBackend.LazrsParallel),
        ("Lazrs", laspy.LazBackend.Lazrs),
    ):
        attempted.append(backend_name)
        try:
            with laspy.open(path, mode="r", laz_backend=backend) as reader:
                points = reader.read()
                if len(points) != int(reader.header.point_count):
                    raise ValueError(
                        f"read {len(points)} points but header declares {reader.header.point_count}"
                    )
                dimensions = [str(name) for name in reader.header.point_format.dimension_names]
            return points, dimensions, backend_name, attempted
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            failures.append(f"{backend_name}: {type(error).__name__}: {error}")
    raise LazReadError(attempted, failures)


def extract_arrays(
    points: Any,
    dimensions: Sequence[str],
    feature_names: Sequence[str],
    audit_names: Sequence[str],
    raw_class_ids: Sequence[int],
    np: Any,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    by_lower = {name.lower(): name for name in dimensions}
    laz_names = {
        "local_x": "x", "local_y": "y", "local_z": "z",
        "intensity": "intensity", "red": "red", "green": "green", "blue": "blue",
        "Reflectance": "reflectance", "Deviation": "deviation", "Amplitude": "amplitude",
    }
    unknown = sorted((set(feature_names) | set(audit_names)) - set(laz_names))
    if unknown:
        raise ValueError(f"unsupported configured feature or audit fields: {unknown}")
    required = {"x", "y", "z", "classification"} | {
        laz_names[name] for name in set(feature_names) | set(audit_names)
        if name not in {"local_x", "local_y", "local_z"}
    }
    missing = sorted(required - set(by_lower))
    if missing:
        raise ValueError(f"required LAZ dimensions are absent: {missing}")
    xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64, copy=False)
    if not np.all(np.isfinite(xyz)):
        raise ValueError("XYZ contains non-finite values")
    origin = np.min(xyz, axis=0)
    local_xyz = xyz - origin
    values: dict[str, Any] = {
        "local_x": local_xyz[:, 0], "local_y": local_xyz[:, 1], "local_z": local_xyz[:, 2]
    }
    for name in set(feature_names) | set(audit_names):
        if name in values:
            continue
        value = np.asarray(points[by_lower[laz_names[name]]], dtype=np.float64)
        if name in {"intensity", "red", "green", "blue"}:
            value = value / 65535.0
        values[name] = value
    features = np.column_stack([values[name] for name in feature_names])
    if not np.all(np.isfinite(features)):
        raise ValueError("input features contain non-finite values")
    source_labels = np.asarray(points[by_lower["classification"]], dtype=np.int16)
    allowed = np.asarray(list(raw_class_ids), dtype=np.int16)
    invalid = np.unique(source_labels[~np.isin(source_labels, allowed)])
    if len(invalid):
        raise ValueError(f"classification IDs outside configured raw_class_ids: {invalid.tolist()}")
    ranges = {
        name: {"min": float(np.min(values[name])), "max": float(np.max(values[name]))}
        for name in list(feature_names) + list(audit_names)
    }
    return local_xyz, features.astype(np.float32), source_labels, origin, ranges


def voxelize(
    local_xyz: Any,
    features: Any,
    source_labels: Any,
    voxel_size: float,
    raw_class_ids: Sequence[int],
    raw_to_train: dict[int, int],
    ignore_index: int,
    np: Any,
) -> tuple[Any, Any, Any, Any]:
    """Mean features and majority labels per deterministic local voxel."""
    if voxel_size <= 0:
        raise ValueError("voxel size must be positive")
    if len(local_xyz) == 0:
        raise ValueError("cannot voxelize an empty point cloud")
    coordinates = np.floor(local_xyz / voxel_size).astype(np.int32)
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    sorted_coordinates = coordinates[order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.any(sorted_coordinates[1:] != sorted_coordinates[:-1], axis=1)) + 1,
        )
    )
    counts = np.diff(np.append(starts, len(order))).astype(np.float64)
    voxel_coordinates = sorted_coordinates[starts]
    voxel_features = np.add.reduceat(features[order].astype(np.float64), starts, axis=0)
    voxel_features = (voxel_features / counts[:, None]).astype(np.float32)
    sorted_labels = source_labels[order]
    raw_ids = np.asarray(sorted(raw_class_ids), dtype=np.int64)
    class_counts = np.column_stack(
        [np.add.reduceat(sorted_labels == class_id, starts) for class_id in raw_ids]
    )
    # Keep a deterministic raw label for audit-only voxels containing only ignored
    # points, but exclude ignored classes from the supervised majority vote.
    raw_labels = raw_ids[np.argmax(class_counts, axis=1)]
    effective_raw_ids = np.asarray(sorted(raw_to_train), dtype=np.int64)
    effective_counts = np.column_stack(
        [np.add.reduceat(sorted_labels == class_id, starts) for class_id in effective_raw_ids]
    )
    has_effective_label = np.any(effective_counts > 0, axis=1)
    raw_labels[has_effective_label] = effective_raw_ids[
        np.argmax(effective_counts[has_effective_label], axis=1)
    ]
    train_labels = np.full(len(raw_labels), ignore_index, dtype=np.int64)
    for raw_id, train_id in raw_to_train.items():
        selected = has_effective_label & (raw_labels == raw_id)
        train_labels[selected] = train_id
    return voxel_coordinates, voxel_features, raw_labels, train_labels


def load_class_mapping(
    path: Path,
) -> tuple[list[int], dict[int, int], int, dict[str, Any]]:
    mapping_config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping_config, dict):
        raise ValueError("class mapping must be a JSON object")
    raw_to_train_data = mapping_config["raw_to_train"]
    train_to_name_data = mapping_config["train_to_name"]
    ignore_classes_data = mapping_config["ignore_classes"]
    if not all(
        isinstance(value, dict)
        for value in (raw_to_train_data, train_to_name_data, ignore_classes_data)
    ):
        raise ValueError("class mapping tables must be JSON objects")
    ignore_index = int(mapping_config["ignore_index"])
    complete_mapping = {
        int(raw_id): int(train_id) for raw_id, train_id in raw_to_train_data.items()
    }
    train_to_name = {
        int(train_id): str(name) for train_id, name in train_to_name_data.items()
    }
    ignore_classes = {
        int(raw_id): str(name) for raw_id, name in ignore_classes_data.items()
    }
    if not complete_mapping:
        raise ValueError("raw_to_train cannot be empty")
    mapped_ignore_ids = {
        raw_id for raw_id, train_id in complete_mapping.items() if train_id == ignore_index
    }
    if mapped_ignore_ids != set(ignore_classes):
        raise ValueError("ignore_classes must exactly match raw IDs mapped to ignore_index")
    effective_mapping = {
        raw_id: train_id
        for raw_id, train_id in complete_mapping.items()
        if train_id != ignore_index
    }
    if sorted(effective_mapping.values()) != list(range(len(effective_mapping))):
        raise ValueError("effective train IDs must be unique and contiguous from zero")
    if set(train_to_name) != set(effective_mapping.values()):
        raise ValueError("train_to_name must name every effective train ID")
    return sorted(complete_mapping), effective_mapping, ignore_index, mapping_config


def atomic_save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            import numpy as np

            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def compatible_output(
    path: Path,
    config_hash: str,
    source_size: int,
    source_mtime_ns: int,
    np: Any,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                int(archive["conversion_version"].item()) == CONVERSION_VERSION
                and str(archive["config_hash"].item()) == config_hash
                and int(archive["source_size_bytes"].item()) == source_size
                and int(archive["source_mtime_ns"].item()) == source_mtime_ns
            )
    except (OSError, KeyError, ValueError):
        return False


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.expanduser().resolve()
        config = load_simple_yaml(config_path)
        manifest_path = (args.manifest or Path(str(config["manifest"]))).expanduser().resolve()
        labels_path = (args.labels_json or Path(str(config["labels_file"]))).expanduser().resolve()
        output_root = (args.output_root or Path(str(config["processed_root"]))).expanduser().resolve()
        class_mapping_ref = Path(str(config["class_mapping_file"])).expanduser()
        class_mapping_path = (
            class_mapping_ref.resolve()
            if class_mapping_ref.is_absolute()
            else (PROJECT_ROOT / class_mapping_ref).resolve()
        )
        config_splits = [str(value) for value in config.get("splits", ["train", "val", "test"])]
        splits = args.splits or config_splits
        voxel_size = float(config["voxel"]["size"])
        feature_mode = args.feature_mode or str(config["default_feature_mode"])
        feature_modes = config["feature_modes"]
        if feature_mode not in feature_modes:
            raise ValueError(
                f"unknown feature mode {feature_mode!r}; available: {sorted(feature_modes)}"
            )
        feature_names = [str(value) for value in feature_modes[feature_mode]["features"]]
        audit_names = [str(value) for value in config.get("auxiliary_audit_fields", [])]
        normalization = config["feature_normalization"]
        missing_normalization = sorted(
            name for name in feature_names + audit_names if name not in normalization
        )
        if missing_normalization:
            raise ValueError(
                f"features without a recorded normalization strategy: {missing_normalization}"
            )
        raw_class_ids, raw_to_train, ignore_index, class_mapping = load_class_mapping(
            class_mapping_path
        )
        known_failed = {str(value) for value in config.get("known_failed_tiles", [])}
        labels = load_labels(labels_path)
        rows = select_rows(
            load_manifest(manifest_path), labels, splits, args.include_tiles, args.max_files
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: invalid conversion inputs: {error}", file=sys.stderr)
        return 3

    # Create this audit artifact for every valid invocation, including dry runs and
    # invocations where a conversion dependency is missing.
    try:
        atomic_write_csv(output_root / "failed_files.csv", FAILED_FIELDS, [])
    except OSError as error:
        print(f"Error: could not initialize failed_files.csv: {error}", file=sys.stderr)
        return 5

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "feature_mode": feature_mode,
                    "features": feature_names,
                    "output_root": str(output_root),
                    "selected_tiles": [row["tile_name"] for row in rows],
                    "failed_files_csv": str(output_root / "failed_files.csv"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    dependencies = require_dependencies()
    if dependencies is None:
        return 2
    laspy, np = dependencies

    conversion_spec = {
        "conversion_version": CONVERSION_VERSION,
        "voxel": config["voxel"],
        "feature_mode": feature_mode,
        "features": feature_names,
        "feature_normalization": config["feature_normalization"],
        "class_mapping": class_mapping,
    }
    config_hash = canonical_hash(conversion_spec)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    started = time.monotonic()

    for index, row in enumerate(rows, start=1):
        tile_name = row["tile_name"]
        split = row["split"]
        source = Path(row["file_path"]).expanduser().resolve()
        target = output_root / split / f"{source.stem}.npz"
        print(f"[{index}/{len(rows)}] {split}/{tile_name}", flush=True)
        record: dict[str, Any] = {
            "tile_name": tile_name, "file_path": str(source), "split": split,
            "output_path": str(target), "status": "", "backend_used": "",
            "source_point_count": 0, "voxel_count": 0, "elapsed_seconds": 0.0,
            "voxel_compression_ratio": 0.0, "npz_size_bytes": 0,
            "raw_label_counts": {}, "train_label_counts": {}, "feature_ranges": {},
            "source_raw_label_counts": {},
        }
        points = local_xyz = features = source_labels = None
        coordinates = voxel_features = raw_labels = train_labels = None
        tile_started = time.monotonic()
        if tile_name in known_failed:
            failure = {
                "tile_name": tile_name, "file_path": str(source), "split": split,
                "status": "skipped_known_failure", "backend_attempted": "",
                "error_type": "KnownDecompressionFailure",
                "error_message": "Excluded without retry: known complete-decompression failure.",
            }
            failures.append(failure)
            record.update(failure)
            records.append(record)
            split_counts[split]["failed"] += 1
            continue
        try:
            stat = source.stat()
            if compatible_output(
                target, config_hash, stat.st_size, stat.st_mtime_ns, np
            ):
                record["status"] = "resumed"
                records.append(record)
                split_counts[split]["resumed"] += 1
                continue
            if target.exists() and not args.overwrite:
                raise FileExistsError(
                    f"incompatible processed output exists; use --overwrite after review: {target}"
                )
            points, dimensions, backend_used, attempted = read_laz(source, laspy)
            record["backend_attempted"] = attempted
            local_xyz, features, source_labels, origin, input_ranges = extract_arrays(
                points, dimensions, feature_names, audit_names, raw_class_ids, np
            )
            coordinates, voxel_features, raw_labels, train_labels = voxelize(
                local_xyz, features, source_labels, voxel_size,
                raw_class_ids, raw_to_train, ignore_index, np
            )
            atomic_save_npz(
                target,
                {
                    "coordinates": coordinates,
                    "features": voxel_features,
                    "raw_labels": raw_labels,
                    "train_labels": train_labels,
                    "tile_name": np.asarray(tile_name),
                    "split": np.asarray(split),
                    "coordinate_origin": origin.astype(np.float64),
                    "voxel_size": np.asarray(voxel_size, dtype=np.float64),
                    "feature_mode": np.asarray(feature_mode),
                    "feature_layout": np.asarray(feature_names),
                    "ignore_index": np.asarray(ignore_index, dtype=np.int64),
                    "raw_class_ids": np.asarray(raw_class_ids, dtype=np.int16),
                    "effective_class_ids": np.asarray(list(raw_to_train), dtype=np.int16),
                    "conversion_version": np.asarray(CONVERSION_VERSION, dtype=np.int16),
                    "config_hash": np.asarray(config_hash),
                    "source_size_bytes": np.asarray(stat.st_size, dtype=np.int64),
                    "source_mtime_ns": np.asarray(stat.st_mtime_ns, dtype=np.int64),
                },
            )
            record.update(
                {
                    "status": "success", "backend_used": backend_used,
                    "backend_attempted": attempted, "source_point_count": len(points),
                    "voxel_count": len(coordinates),
                    "voxel_compression_ratio": len(coordinates) / len(points),
                    "npz_size_bytes": target.stat().st_size,
                    "source_raw_label_counts": {
                        str(int(key)): int(value)
                        for key, value in zip(*np.unique(source_labels, return_counts=True))
                    },
                    "raw_label_counts": {
                        str(int(key)): int(value)
                        for key, value in zip(*np.unique(raw_labels, return_counts=True))
                    },
                    "train_label_counts": {
                        str(int(key)): int(value)
                        for key, value in zip(*np.unique(train_labels, return_counts=True))
                    },
                    "feature_ranges": {
                        "input_and_audit": input_ranges,
                        "voxelized": {
                            name: {
                                "min": float(np.min(voxel_features[:, column])),
                                "max": float(np.max(voxel_features[:, column])),
                            }
                            for column, name in enumerate(feature_names)
                        },
                    },
                }
            )
            print(
                f"  points={len(points)} voxels={len(coordinates)} "
                f"compression={record['voxel_compression_ratio']:.6f} "
                f"npz_bytes={record['npz_size_bytes']}"
            )
            print(f"  raw_label_counts={record['raw_label_counts']}")
            print(f"  train_label_counts={record['train_label_counts']}")
            print(f"  feature_ranges={record['feature_ranges']}")
            split_counts[split]["success"] += 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            attempted = error.attempted if isinstance(error, LazReadError) else record.get(
                "backend_attempted", ""
            )
            failure = {
                "tile_name": tile_name, "file_path": str(source), "split": split,
                "status": "failed", "backend_attempted": attempted,
                "error_type": type(error).__name__, "error_message": str(error),
            }
            failures.append(failure)
            record.update(failure)
            split_counts[split]["failed"] += 1
            print(f"  failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        finally:
            record["elapsed_seconds"] = time.monotonic() - tile_started
            if record not in records:
                records.append(record)
            points = local_xyz = features = source_labels = None
            coordinates = voxel_features = raw_labels = train_labels = None
            gc.collect()

    summary = {
        "processing_version": CONVERSION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_hash": config_hash,
        "manifest": str(manifest_path), "labels_json": str(labels_path),
        "class_mapping": str(class_mapping_path),
        "output_root": str(output_root), "official_splits": list(splits),
        "selected_files": len(rows),
        "split_counts": {split: dict(split_counts[split]) for split in splits},
        "success_count": sum(row["status"] == "success" for row in records),
        "resumed_count": sum(row["status"] == "resumed" for row in records),
        "failed_count": sum(row["status"] == "failed" for row in records),
        "known_skipped_count": sum(
            row["status"] == "skipped_known_failure" for row in records
        ),
        "failed_or_skipped_count": len(failures),
        "known_failed_tiles": sorted(known_failed),
        "voxel_size": voxel_size,
        "feature_mode": feature_mode,
        "input_features": feature_names,
        "feature_normalization": {
            name: normalization[name] for name in feature_names + audit_names
        },
        "auxiliary_audit_fields": audit_names,
        "geometry_features_included": False,
        "label_encoding": {
            "raw_labels": raw_class_ids,
            "train_mapping": raw_to_train,
            "ignore_index": ignore_index,
        },
        "batch_dimension_saved": False,
        "elapsed_seconds": time.monotonic() - started,
        "files": records,
    }
    try:
        atomic_write_json(output_root / "processing_summary.json", summary)
        atomic_write_csv(output_root / "failed_files.csv", FAILED_FIELDS, failures)
    except OSError as error:
        print(f"Error: could not write conversion summary: {error}", file=sys.stderr)
        return 5
    print(
        f"Prepared={summary['success_count']} resumed={summary['resumed_count']} "
        f"failed_or_skipped={summary['failed_or_skipped_count']}",
        flush=True,
    )
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
