#!/usr/bin/env python3
"""Exp1 epoch-6 resume with logged, finite-guarded sliding validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_pipeline import atomic_write_json
import tools.train_exp1_geometry_concat_dryrun as runtime
import tools.train_exp1_geometry_concat_formal as formal
import tools.train_exp1_geometry_concat_resume_final as resume

SOURCE = ROOT / "results/exp1_geometry_concat_final"
OUTPUT = ROOT / "results/exp1_geometry_concat_final_v2"
CHECKPOINT = SOURCE / "checkpoint_last.pth"
WINDOW_COLUMNS = (
    "phase", "epoch", "split", "tile_index", "tile_name", "window_id",
    "x0", "y0", "voxel_count", "coordinates_shape", "features_shape",
    "geometry_shape", "logits_shape", "logits_finite",
    "nonfinite_logit_count", "validation_dtype", "logits_dtype",
    "output_order_matches_input",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_finite_audit(epoch: str | int, model: Any) -> dict[str, Any]:
    rows = []
    for name, value in model.state_dict().items():
        finite = bool(value.isfinite().all())
        rows.append(
            {
                "epoch": epoch,
                "tensor_name": name,
                "shape": json.dumps(list(value.shape)),
                "finite": finite,
                "nonfinite_count": int((~value.isfinite()).sum()),
            }
        )
    path = OUTPUT / "pre_validation_model_finite.csv"
    create = not path.exists()
    with path.open("x" if create else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        if create:
            writer.writeheader()
        writer.writerows(rows)
    failures = [row["tensor_name"] for row in rows if not row["finite"]]
    result = {
        "epoch": epoch,
        "tensor_count": len(rows),
        "all_finite": not failures,
        "nonfinite_tensors": failures,
    }
    if failures:
        raise RuntimeError(
            f"pre-validation model finite check failed: {failures[:5]}"
        )
    return result


def evaluate_logged(model: Any, paths: list[Path], split: str, device: Any,
                    weights: Any, *, epoch: str | int,
                    phase: str) -> dict[str, Any]:
    import MinkowskiEngine as ME
    import torch
    import torch.nn.functional as F

    from engine.sliding_window_inference import _window_starts
    from tools.train_baseline_strong_block import metrics_from_confusion

    expected = 59 if split == "val" else 106
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} {split} tiles, found {len(paths)}")
    log_path = OUTPUT / "inference_window_log.csv"
    create_log = not log_path.exists()
    confusion = torch.zeros((9, 9), dtype=torch.int64)
    loss_sum = 0.0
    valid_total = 0
    minimum_vote = None
    total_windows = 0
    invalid_logits = 0
    with log_path.open(
        "x" if create_log else "a", newline="", encoding="utf-8"
    ) as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=WINDOW_COLUMNS)
        if create_log:
            writer.writeheader()
        for tile_index, source in enumerate(paths, 1):
            cache_path = (
                ROOT / "datasets/geometry_features" / split
                / f"{source.stem}_geometry.npz"
            )
            with np.load(source, allow_pickle=False) as archive:
                coordinates = np.asarray(
                    archive["coordinates"], dtype=np.int32
                )
                features = np.asarray(archive["features"], dtype=np.float32)
                labels = np.asarray(archive["labels"], dtype=np.int64)
            with np.load(cache_path, allow_pickle=False) as archive:
                geometry = np.asarray(
                    archive["geometry_normalized"], dtype=np.float32
                )
                point_indices = np.asarray(
                    archive["point_indices"], dtype=np.int64
                )
                metadata = json.loads(str(archive["metadata"].item()))
            if (
                features.shape != (len(coordinates), 6)
                or geometry.shape != (len(coordinates), 12)
                or not np.array_equal(
                    point_indices,
                    np.arange(len(coordinates), dtype=np.int64),
                )
                or metadata["source_npz_sha256"] != sha256(source)
            ):
                raise RuntimeError(
                    f"geometry/input alignment error: {split}/{source.name}"
                )
            logits_sum = torch.zeros(
                (len(coordinates), 9), dtype=torch.float32
            )
            votes = torch.zeros(len(coordinates), dtype=torch.int32)
            x_starts = _window_starts(
                int(coordinates[:, 0].min()),
                int(coordinates[:, 0].max()), 500, 250,
            )
            y_starts = _window_starts(
                int(coordinates[:, 1].min()),
                int(coordinates[:, 1].max()), 500, 250,
            )
            model.eval()
            window_id = 0
            with torch.no_grad():
                for x0 in x_starts:
                    for y0 in y_starts:
                        indices = np.flatnonzero(
                            (coordinates[:, 0] >= x0)
                            & (coordinates[:, 0] < x0 + 500)
                            & (coordinates[:, 1] >= y0)
                            & (coordinates[:, 1] < y0 + 500)
                        )
                        if not len(indices):
                            continue
                        window_id += 1
                        local = coordinates[indices].copy()
                        local -= np.asarray(
                            [x0, y0, int(local[:, 2].min())],
                            dtype=np.int32,
                        )
                        batched = np.column_stack(
                            (np.zeros(len(local), dtype=np.int32), local)
                        )
                        sparse = ME.SparseTensor(
                            features=torch.from_numpy(
                                features[indices]
                            ).to(device),
                            coordinates=torch.from_numpy(batched),
                            device=device,
                        )
                        geometry_input = torch.from_numpy(
                            geometry[indices]
                        ).to(device)
                        # Validation/inference is FP32; training keeps its
                        # separate mixed-precision GradScaler path.
                        with torch.autocast("cuda", enabled=False):
                            fused = ME.SparseTensor(
                                features=torch.cat(
                                    (sparse.F, geometry_input), 1
                                ),
                                coordinate_map_key=sparse.coordinate_map_key,
                                coordinate_manager=sparse.coordinate_manager,
                            )
                            output = model(fused)
                        logits = output.F.detach().float().cpu()
                        finite = torch.isfinite(logits)
                        order_ok = torch.equal(
                            output.C.cpu(), torch.from_numpy(batched)
                        )
                        row = {
                            "phase": phase,
                            "epoch": epoch,
                            "split": split,
                            "tile_index": tile_index,
                            "tile_name": source.name,
                            "window_id": window_id,
                            "x0": x0,
                            "y0": y0,
                            "voxel_count": len(indices),
                            "coordinates_shape": json.dumps(
                                list(batched.shape)
                            ),
                            "features_shape": json.dumps(
                                list(sparse.F.shape)
                            ),
                            "geometry_shape": json.dumps(
                                list(geometry_input.shape)
                            ),
                            "logits_shape": json.dumps(list(logits.shape)),
                            "logits_finite": bool(finite.all()),
                            "nonfinite_logit_count": int((~finite).sum()),
                            "validation_dtype": "float32",
                            "logits_dtype": str(output.F.dtype),
                            "output_order_matches_input": order_ok,
                        }
                        writer.writerow(row)
                        log_handle.flush()
                        total_windows += 1
                        invalid_logits += int(not row["logits_finite"])
                        if logits.shape != (len(indices), 9):
                            raise RuntimeError(
                                f"invalid logits shape: {split}/"
                                f"{source.name} window={window_id} "
                                f"shape={list(logits.shape)}"
                            )
                        if not row["logits_finite"]:
                            raise RuntimeError(
                                f"non-finite logits: {split}/"
                                f"{source.name} window={window_id} "
                                f"x0={x0} y0={y0} count="
                                f"{row['nonfinite_logit_count']}"
                            )
                        if not order_ok:
                            raise RuntimeError(
                                f"output order mismatch: {split}/"
                                f"{source.name} window={window_id}"
                            )
                        index_tensor = torch.from_numpy(indices)
                        logits_sum[index_tensor] += logits
                        votes[index_tensor] += 1
                        del sparse, geometry_input, fused, output, logits
            if torch.any(votes == 0):
                raise RuntimeError(
                    f"sliding windows missed voxels: {split}/{source.name}"
                )
            averaged = logits_sum / votes.float().unsqueeze(1)
            if averaged.shape != (len(coordinates), 9) or not bool(
                torch.isfinite(averaged).all()
            ):
                raise RuntimeError(
                    f"invalid aggregated logits: {split}/{source.name}"
                )
            label_tensor = torch.from_numpy(labels)
            valid = label_tensor != -100
            encoded = label_tensor[valid] * 9 + averaged.argmax(1)[valid]
            confusion += torch.bincount(
                encoded, minlength=81
            ).reshape(9, 9)
            tile_loss = F.cross_entropy(
                averaged, label_tensor, weight=weights,
                ignore_index=-100, reduction="sum",
            )
            if not bool(torch.isfinite(tile_loss)):
                raise RuntimeError(
                    f"non-finite evaluation loss: {split}/{source.name}"
                )
            loss_sum += float(tile_loss)
            valid_total += int(valid.sum())
            tile_minimum = int(votes.min())
            minimum_vote = (
                tile_minimum if minimum_vote is None
                else min(minimum_vote, tile_minimum)
            )
            print(
                f"{phase} {split}_tile={tile_index}/{expected} "
                f"name={source.stem} windows={window_id} "
                f"min_vote={tile_minimum}",
                flush=True,
            )
            del logits_sum, votes, averaged, label_tensor, valid, encoded
    if int(confusion.sum()) != valid_total:
        raise RuntimeError(f"{split} confusion total mismatch")
    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {
            "split": split,
            "loss": loss_sum / valid_total,
            "minimum_vote_count": minimum_vote,
            "tiles": len(paths),
            "window_count": total_windows,
            "validation_dtype": "float32",
            "logits_dtype": "torch.float32",
            "invalid_logits_count": invalid_logits,
            "window_log": str(log_path),
        }
    )
    return metrics


def copy_history() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with (SOURCE / "training_log.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, 7)):
        raise RuntimeError("source training history is not epochs 1..6")
    with (OUTPUT / "training_log.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=resume.LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {
                key: row.get(key, "") for key in resume.LOG_COLUMNS
            }
            for row in rows
        )
    amp = json.loads((SOURCE / "amp_summary.json").read_text())
    epochs = amp["epochs"]
    if [int(item["epoch"]) for item in epochs] != list(range(1, 7)):
        raise RuntimeError("source AMP history is not epochs 1..6")
    return rows, epochs


def append_training_row(row: dict[str, Any]) -> None:
    with (OUTPUT / "training_log.csv").open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        csv.DictWriter(
            handle, fieldnames=resume.LOG_COLUMNS
        ).writerow({key: row[key] for key in resume.LOG_COLUMNS})


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    source_manifest = json.loads(
        (SOURCE / "run_manifest.json").read_text()
    )
    reproduction = json.loads(
        (
            SOURCE
            / "epoch7_failure_reproduction/reproduction_report.json"
        ).read_text()
    )
    if (
        source_manifest["test_accessed"]
        or source_manifest["test_evaluations"] != 0
        or reproduction["formal_state"]["test_accessed"]
        or not reproduction["formal_state"]["checkpoint_last_unchanged"]
    ):
        raise RuntimeError("source/test/reproduction gate failed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CHECKPOINT, OUTPUT / "checkpoint_last.pth")
    shutil.copy2(
        SOURCE / "checkpoint_best.pth",
        OUTPUT / "checkpoint_best.pth",
    )
    _history, amp_epochs = copy_history()
    (
        _experiment, train_config, dataset, loader, generator, model,
        weights, optimizer, scheduler, scaler,
    ) = runtime.build_runtime()
    if len(dataset) != 1410:
        raise RuntimeError("train merged sample count changed")
    val_paths = sorted((ROOT / "datasets/processed/val").glob("*.npz"))
    if len(val_paths) != 59:
        raise RuntimeError("val tile count changed")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(state["completed_epoch"]) != 6:
        raise RuntimeError("source checkpoint is not epoch6")
    device = torch.device("cuda:0")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    runtime.restore_rng(torch, state["rng_state"])
    generator.set_state(state["train_generator_state"])
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights.to(device), ignore_index=-100
    )
    best_epoch = int(state["best_epoch"])
    best_miou = float(state["best_val_miou"])
    manifest = {
        "status": "training_resumed_v2",
        "experiment_id": "Exp1",
        "model_name": "Geometry-Concat",
        "output_dir": str(OUTPUT.relative_to(ROOT)),
        "resume_source": str(CHECKPOINT.relative_to(ROOT)),
        "resume_source_sha256": sha256(CHECKPOINT),
        "resume_completed_epoch": 6,
        "completed_epochs": 6,
        "target_epochs": 100,
        "train_merged_samples": 1410,
        "val_tiles": 59,
        "test_isolated_until_final_best": True,
        "test_accessed": False,
        "test_evaluations": 0,
        "frozen_settings": {
            "model": "Geometry-Concat",
            "geometry_input_dim": 12,
            "block_index": "datasets/block_indices_50m_merged",
            "voxel_size": 0.1,
            "loss": "balanced_cross_entropy",
            "class_weight_file": train_config["loss"]["class_weight_file"],
            "class_weights_fit_split": "train",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": 100,
            "gradient_accumulation_steps": 2,
            "physical_batch_size": 1,
            "seed": 42,
            "gradient_clipping": False,
        },
        "amp_policy": "standard_gradscaler_overflow_skip",
        "validation": {
            "pre_validation_model_finite_check": True,
            "per_window_shape_and_finite_log": True,
            "block_size_m": 50.0,
            "stride_m": 25.0,
            "overlap": 0.5,
        },
        "resume_state_restored": {
            "model": True, "optimizer": True, "scheduler": True,
            "scaler": True, "python_rng": True, "numpy_rng": True,
            "torch_rng": True, "cuda_rng": True,
            "dataloader_generator": True,
        },
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "started_at_utc": runtime.utc_now(),
        "git": runtime.git_state(),
    }
    atomic_write_json(OUTPUT / "run_manifest.json", manifest)
    current_context: dict[str, Any] = {}
    try:
        for epoch in range(7, 101):
            epoch_started = time.monotonic()
            current_context = {"phase": "train", "epoch": epoch}
            train_started = time.monotonic()
            train_metrics = resume.train_epoch(
                torch, model, loader, optimizer, scaler,
                criterion, device, epoch,
            )
            train_seconds = time.monotonic() - train_started
            current_context = {
                "phase": "pre_validation_finite", "epoch": epoch
            }
            finite_audit = append_finite_audit(epoch, model)
            current_context = {"phase": "val", "epoch": epoch}
            val_started = time.monotonic()
            val_metrics = evaluate_logged(
                model, val_paths, "val", device, weights,
                epoch=epoch, phase="epoch_validation",
            )
            val_seconds = time.monotonic() - val_started
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_OA": val_metrics["oa"],
                "val_mIoU": val_metrics["miou"],
                "val_mAcc": val_metrics["macc"],
                "train_batches": train_metrics["batches"],
                "optimizer_step_attempts": train_metrics[
                    "optimizer_step_attempts"
                ],
                "amp_skipped_steps": train_metrics["amp_skipped_steps"],
                "amp_skip_rate": train_metrics["amp_skip_rate"],
                "starting_grad_scale": train_metrics[
                    "starting_grad_scale"
                ],
                "minimum_grad_scale": train_metrics["minimum_grad_scale"],
                "ending_grad_scale": train_metrics["ending_grad_scale"],
                "scale_change_count": train_metrics["scale_change_count"],
                "amp_skip_samples": json.dumps(
                    train_metrics["amp_skip_samples"],
                    ensure_ascii=False, separators=(",", ":"),
                ),
                "epoch_train_time_seconds": train_seconds,
                "epoch_val_time_seconds": val_seconds,
            }
            append_training_row(row)
            amp_epochs.append(
                {
                    "epoch": epoch,
                    "optimizer_step_attempts": train_metrics[
                        "optimizer_step_attempts"
                    ],
                    "skipped_steps": train_metrics["amp_skipped_steps"],
                    "skip_rate": train_metrics["amp_skip_rate"],
                    "starting_scale": train_metrics[
                        "starting_grad_scale"
                    ],
                    "minimum_scale": train_metrics["minimum_grad_scale"],
                    "ending_scale": train_metrics["ending_grad_scale"],
                    "scale_change_count": train_metrics[
                        "scale_change_count"
                    ],
                    "skip_samples": train_metrics["amp_skip_samples"],
                }
            )
            atomic_write_json(
                OUTPUT / "amp_summary.json",
                {
                    "status": "training",
                    "policy": "standard_gradscaler_overflow_skip",
                    "epochs": amp_epochs,
                    "total_optimizer_step_attempts": sum(
                        item["optimizer_step_attempts"]
                        for item in amp_epochs
                    ),
                    "total_skipped_steps": sum(
                        item["skipped_steps"] for item in amp_epochs
                    ),
                    "test_accessed": False,
                },
            )
            if val_metrics["miou"] > best_miou:
                best_miou, best_epoch = float(val_metrics["miou"]), epoch
                resume.save_checkpoint(
                    OUTPUT / "checkpoint_best.pth", torch, model,
                    optimizer, scheduler, scaler, generator,
                    epoch=epoch, best_epoch=best_epoch,
                    best_miou=best_miou, metrics=val_metrics,
                )
            scheduler.step()
            resume.save_checkpoint(
                OUTPUT / "checkpoint_last.pth", torch, model,
                optimizer, scheduler, scaler, generator,
                epoch=epoch, best_epoch=best_epoch,
                best_miou=best_miou, metrics=val_metrics,
            )
            manifest.update(
                {
                    "completed_epochs": epoch,
                    "best_epoch": best_epoch,
                    "best_val_miou": best_miou,
                    "last_pre_validation_finite": finite_audit,
                    "last_epoch_seconds": time.monotonic() - epoch_started,
                    "checkpoint_best_sha256": sha256(
                        OUTPUT / "checkpoint_best.pth"
                    ),
                    "checkpoint_last_sha256": sha256(
                        OUTPUT / "checkpoint_last.pth"
                    ),
                    "updated_at_utc": runtime.utc_now(),
                }
            )
            atomic_write_json(OUTPUT / "run_manifest.json", manifest)
            print(
                f"epoch_complete={epoch}/100 train_loss="
                f"{row['train_loss']:.8f} val_miou="
                f"{row['val_mIoU']:.8f} skips="
                f"{row['amp_skipped_steps']} best_epoch={best_epoch} "
                f"best_miou={best_miou:.8f}",
                flush=True,
            )
        current_context = {"phase": "best_pre_validation_finite"}
        best = torch.load(
            OUTPUT / "checkpoint_best.pth",
            map_location="cpu", weights_only=False,
        )
        model.load_state_dict(best["model_state_dict"])
        model.to(device)
        append_finite_audit("best_revalidation", model)
        current_context = {"phase": "best_val_revalidation"}
        replay_val = evaluate_logged(
            model, val_paths, "val", device, weights,
            epoch="best", phase="best_revalidation",
        )
        with (OUTPUT / "training_log.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 100:
            raise RuntimeError(
                f"training log rows={len(rows)}, expected 100"
            )
        best_row = max(rows, key=lambda item: float(item["val_mIoU"]))
        difference = abs(
            replay_val["miou"] - float(best_row["val_mIoU"])
        )
        if difference > 1e-12:
            raise RuntimeError(f"best val replay mismatch: {difference}")
        atomic_write_json(
            OUTPUT / "metrics.json",
            {
                "checkpoint_epoch": int(best["completed_epoch"]),
                "training_log_best_mIoU": float(
                    best_row["val_mIoU"]
                ),
                "revalidation_difference": difference,
                "matches_training_log": True,
                **replay_val,
            },
        )
        # First test access: only after epoch100, fixed best, and val replay.
        current_context = {"phase": "final_test"}
        manifest["test_accessed"] = True
        manifest["test_access_reason"] = (
            "epoch100 complete; best fixed; 59-tile val replay passed"
        )
        atomic_write_json(OUTPUT / "run_manifest.json", manifest)
        test_paths = sorted(
            (ROOT / "datasets/processed/test").glob("*.npz")
        )
        append_finite_audit("final_test", model)
        test_metrics = evaluate_logged(
            model, test_paths, "test", device, weights,
            epoch="best", phase="final_test",
        )
        atomic_write_json(OUTPUT / "test_metrics.json", test_metrics)
        resume.write_class_rows(
            OUTPUT / "per_class_metrics.csv",
            resume.combined_class_rows(replay_val, test_metrics),
        )
        amp_summary = json.loads(
            (OUTPUT / "amp_summary.json").read_text()
        )
        amp_summary.update(
            {
                "status": "complete",
                "total_skip_rate": amp_summary["total_skipped_steps"]
                / amp_summary["total_optimizer_step_attempts"],
                "test_accessed_after_training": True,
            }
        )
        atomic_write_json(OUTPUT / "amp_summary.json", amp_summary)
        manifest.update(
            {
                "status": "complete",
                "ended_at_utc": runtime.utc_now(),
                "completed_epochs": 100,
                "best_epoch": best_epoch,
                "best_val_miou": best_miou,
                "best_val_revalidation_matches": True,
                "best_val_revalidation_difference": difference,
                "test_evaluations": 1,
                "test_checkpoint_epoch": int(best["completed_epoch"]),
            }
        )
        atomic_write_json(OUTPUT / "run_manifest.json", manifest)
        return 0
    except BaseException as error:
        atomic_write_json(
            OUTPUT / "failure_report.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "context": current_context,
                "timestamp_utc": runtime.utc_now(),
                "automatic_parameter_changes": False,
            },
        )
        manifest.update(
            {
                "status": "failed",
                "failure_report": "failure_report.json",
                "updated_at_utc": runtime.utc_now(),
            }
        )
        atomic_write_json(OUTPUT / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
