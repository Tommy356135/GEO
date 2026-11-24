#!/usr/bin/env python3
"""Authorized Exp1 epoch-5 resume using standard GradScaler overflow skips."""

from __future__ import annotations

import csv
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

SOURCE = ROOT / "results/exp1_geometry_concat"
OUTPUT = ROOT / "results/exp1_geometry_concat_final"
DIAGNOSTIC = SOURCE / "amp_diagnostics_v2/diagnostic_summary.json"
LOG_COLUMNS = (
    "epoch", "learning_rate", "train_loss", "val_loss", "val_OA", "val_mIoU",
    "val_mAcc", "train_batches", "optimizer_step_attempts",
    "amp_skipped_steps", "amp_skip_rate", "starting_grad_scale",
    "minimum_grad_scale", "ending_grad_scale", "scale_change_count",
    "amp_skip_samples", "epoch_train_time_seconds", "epoch_val_time_seconds",
)


def save_checkpoint(path: Path, torch: Any, model: Any, optimizer: Any,
                    scheduler: Any, scaler: Any, generator: Any, *, epoch: int,
                    best_epoch: int, best_miou: float,
                    metrics: dict[str, Any]) -> None:
    temporary = path.with_suffix(".pth.tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_miou": best_miou,
            "val_metrics": metrics,
            "rng_state": runtime.capture_rng(torch),
            "train_generator_state": generator.get_state(),
            "physical_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "experiment_config_sha256": runtime.sha256(
                runtime.EXPERIMENT_CONFIG
            ),
            "training_config_sha256": runtime.sha256(runtime.TRAIN_CONFIG),
            "amp_policy": "standard_gradscaler_overflow_skip",
            "test_accessed": False,
        },
        temporary,
    )
    os.replace(temporary, path)


def write_initial_log() -> None:
    source_path = SOURCE / "training_log.csv"
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in source_rows] != list(range(1, 6)):
        raise RuntimeError("source training log is not exactly epochs 1..5")
    with (OUTPUT / "training_log.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for row in source_rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in LOG_COLUMNS},
                    "minimum_grad_scale": row.get("starting_grad_scale", ""),
                    "scale_change_count": 0,
                    "amp_skip_samples": "[]",
                }
            )


def append_log(row: dict[str, Any]) -> None:
    with (OUTPUT / "training_log.csv").open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        csv.DictWriter(handle, fieldnames=LOG_COLUMNS).writerow(
            {key: row[key] for key in LOG_COLUMNS}
        )


def train_epoch(torch: Any, model: Any, loader: Any, optimizer: Any,
                scaler: Any, criterion: Any, device: Any,
                epoch: int) -> dict[str, Any]:
    import MinkowskiEngine as ME

    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    valid_total = attempts = skips = 0
    starting_scale = float(scaler.get_scale())
    minimum_scale = starting_scale
    scale_changes = 0
    skip_samples: list[dict[str, Any]] = []
    for micro_index, batch in enumerate(loader, 1):
        sparse = ME.SparseTensor(
            features=batch["features"].to(device),
            coordinates=batch["coordinates"], device=device,
        )
        geometry = batch["geometry"].to(device)
        labels = batch["labels"].to(device)
        if len(sparse.F) != len(geometry) or len(labels) != len(geometry):
            raise RuntimeError("geometry alignment error")
        if not bool(torch.isfinite(batch["features"]).all()):
            raise RuntimeError("non-finite input features")
        if not bool(torch.isfinite(geometry).all()):
            raise RuntimeError("non-finite geometry")
        valid_points = int((labels != -100).sum())
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            fused = ME.SparseTensor(
                features=torch.cat((sparse.F, geometry), 1),
                coordinate_map_key=sparse.coordinate_map_key,
                coordinate_manager=sparse.coordinate_manager,
            )
            logits = model(fused)
            raw_loss = criterion(logits.F, labels)
        if logits.F.shape != (len(labels), 9) or not bool(
            torch.isfinite(raw_loss)
        ):
            raise RuntimeError("NaN/non-finite training loss or invalid logits")
        scaler.scale(raw_loss / 2).backward()
        should_step = micro_index % 2 == 0 or micro_index == len(loader)
        if should_step:
            attempts += 1
            scaler.unscale_(optimizer)
            gradients_finite = all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            scale_before = float(scaler.get_scale())
            # Authorized standard GradScaler policy: an overflow suppresses the
            # optimizer step; update backs off the scale. No clipping is used.
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            skipped = scale_after < scale_before
            if scale_after != scale_before:
                scale_changes += 1
            minimum_scale = min(minimum_scale, scale_after)
            if skipped:
                skips += 1
                skip_samples.append(
                    {
                        "micro_batch_index": micro_index,
                        "tile_name": batch["tile_name"][0],
                        "block_name": batch["block_id"][0],
                        "sample_name": batch["merged_sample_id"][0],
                        "voxel_count": int(batch["features"].shape[0]),
                        "loss": float(raw_loss.detach().cpu()),
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                        "gradient_finite": gradients_finite,
                    }
                )
                print(
                    f"amp_skip epoch={epoch} micro={micro_index} "
                    f"tile={batch['tile_name'][0]} block={batch['block_id'][0]} "
                    f"scale={scale_before}->{scale_after}",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
        weighted_loss += float(raw_loss.detach()) * valid_points
        valid_total += valid_points
        if micro_index == 1 or micro_index % 100 == 0 or micro_index == len(loader):
            print(
                f"epoch={epoch}/100 batch={micro_index}/{len(loader)} "
                f"loss={float(raw_loss.detach()):.6f} "
                f"scale={float(scaler.get_scale()):.1f} skips={skips}",
                flush=True,
            )
        del sparse, fused, geometry, labels, logits, raw_loss, batch
    unhealthy = [
        name for name, value in model.state_dict().items()
        if not bool(torch.isfinite(value).all())
    ]
    if unhealthy:
        raise RuntimeError(f"non-finite model/BatchNorm state: {unhealthy[:5]}")
    return {
        "loss": weighted_loss / valid_total,
        "valid_points": valid_total,
        "batches": len(loader),
        "optimizer_step_attempts": attempts,
        "amp_skipped_steps": skips,
        "amp_skip_rate": skips / attempts,
        "starting_grad_scale": starting_scale,
        "minimum_grad_scale": minimum_scale,
        "ending_grad_scale": float(scaler.get_scale()),
        "scale_change_count": scale_changes,
        "amp_skip_samples": skip_samples,
    }


def combined_class_rows(val_metrics: dict[str, Any],
                        test_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, metrics in (("val", val_metrics), ("test", test_metrics)):
        for row in formal.class_rows(metrics):
            rows.append({"split": split, **row})
    return rows


def write_class_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    diagnosis = json.loads(DIAGNOSTIC.read_text())
    if not (
        diagnosis["status"] == "diagnosed"
        and diagnosis["root_cause"] == "AMP_overflow"
        and diagnosis["resume_allowed"]
        and diagnosis["formal_state"]["checkpoint_last_unchanged"]
    ):
        raise RuntimeError("AMP v2 diagnosis does not authorize resume")
    source_manifest = json.loads((SOURCE / "run_manifest.json").read_text())
    if source_manifest["test_accessed"] or source_manifest["test_evaluations"]:
        raise RuntimeError("test isolation was already violated")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_checkpoint = SOURCE / "checkpoint_last.pth"
    source_best = SOURCE / "checkpoint_best.pth"
    shutil.copy2(source_checkpoint, OUTPUT / "checkpoint_last.pth")
    shutil.copy2(source_best, OUTPUT / "checkpoint_best.pth")
    write_initial_log()
    (
        exp_config, train_config, dataset, loader, generator, model,
        weights, optimizer, scheduler, scaler,
    ) = runtime.build_runtime()
    if len(dataset) != 1410:
        raise RuntimeError("train merged sample count changed")
    val_paths = sorted((ROOT / "datasets/processed/val").glob("*.npz"))
    if len(val_paths) != 59:
        raise RuntimeError("val tile count changed")
    device = torch.device("cuda:0")
    state = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if int(state["completed_epoch"]) != 5:
        raise RuntimeError("resume checkpoint is not epoch 5")
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
        "status": "training_resumed",
        "experiment_id": "Exp1",
        "model_name": "Geometry-Concat",
        "formal_training": True,
        "output_dir": str(OUTPUT.relative_to(ROOT)),
        "resume_source": str(source_checkpoint.relative_to(ROOT)),
        "resume_source_sha256": runtime.sha256(source_checkpoint),
        "resume_completed_epoch": 5,
        "completed_epochs": 5,
        "target_epochs": 100,
        "train_merged_samples": 1410,
        "val_tiles": 59,
        "test_isolated_until_final_best": True,
        "test_accessed": False,
        "test_evaluations": 0,
        "frozen_settings": {
            "model": "Geometry-Concat",
            "geometry_input_dim": 12,
            "voxel_size": 0.1,
            "block_index": "datasets/block_indices_50m_merged",
            "loss": "balanced_cross_entropy",
            "class_weight_file": train_config["loss"]["class_weight_file"],
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": 100,
            "seed": 42,
            "physical_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "gradient_clipping": False,
        },
        "amp_policy": "standard_gradscaler_overflow_skip",
        "amp_diagnostic_basis": str(DIAGNOSTIC.relative_to(ROOT)),
        "resume_state_restored": {
            "model": True, "optimizer": True, "scheduler": True,
            "scaler": True, "rng": True, "cuda_rng": True,
            "dataloader_generator": True,
        },
        "best_epoch": best_epoch,
        "best_val_miou": best_miou,
        "started_at_utc": runtime.utc_now(),
        "git": runtime.git_state(),
    }
    atomic_write_json(OUTPUT / "run_manifest.json", manifest)
    current_context: dict[str, Any] = {}
    with (OUTPUT / "training_log.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        historical_rows = list(csv.DictReader(handle))
    amp_epochs: list[dict[str, Any]] = [
        {
            "epoch": int(row["epoch"]),
            "optimizer_step_attempts": int(row["optimizer_step_attempts"]),
            "skipped_steps": int(row["amp_skipped_steps"]),
            "skip_rate": float(row["amp_skip_rate"]),
            "starting_scale": float(row["starting_grad_scale"]),
            "minimum_scale": float(
                row["minimum_grad_scale"] or row["starting_grad_scale"]
            ),
            "ending_scale": float(row["ending_grad_scale"]),
            "scale_change_count": int(row["scale_change_count"] or 0),
            "skip_samples": json.loads(row["amp_skip_samples"] or "[]"),
            "source": "pre_resume_history",
        }
        for row in historical_rows
    ]
    try:
        for epoch in range(6, 101):
            epoch_started = time.monotonic()
            current_context = {"phase": "train", "epoch": epoch}
            train_started = time.monotonic()
            train_metrics = train_epoch(
                torch, model, loader, optimizer, scaler, criterion, device, epoch
            )
            train_seconds = time.monotonic() - train_started
            current_context = {"phase": "val", "epoch": epoch}
            val_started = time.monotonic()
            val_metrics = formal.evaluate_split(
                model, val_paths, "val", device, weights
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
                "starting_grad_scale": train_metrics["starting_grad_scale"],
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
            append_log(row)
            amp_epochs.append(
                {
                    "epoch": epoch,
                    "optimizer_step_attempts": train_metrics[
                        "optimizer_step_attempts"
                    ],
                    "skipped_steps": train_metrics["amp_skipped_steps"],
                    "skip_rate": train_metrics["amp_skip_rate"],
                    "starting_scale": train_metrics["starting_grad_scale"],
                    "minimum_scale": train_metrics["minimum_grad_scale"],
                    "ending_scale": train_metrics["ending_grad_scale"],
                    "scale_change_count": train_metrics["scale_change_count"],
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
                        item["optimizer_step_attempts"] for item in amp_epochs
                    ),
                    "total_skipped_steps": sum(
                        item["skipped_steps"] for item in amp_epochs
                    ),
                    "test_accessed": False,
                },
            )
            if val_metrics["miou"] > best_miou:
                best_miou, best_epoch = float(val_metrics["miou"]), epoch
                save_checkpoint(
                    OUTPUT / "checkpoint_best.pth", torch, model, optimizer,
                    scheduler, scaler, generator, epoch=epoch,
                    best_epoch=best_epoch, best_miou=best_miou,
                    metrics=val_metrics,
                )
            scheduler.step()
            save_checkpoint(
                OUTPUT / "checkpoint_last.pth", torch, model, optimizer,
                scheduler, scaler, generator, epoch=epoch,
                best_epoch=best_epoch, best_miou=best_miou,
                metrics=val_metrics,
            )
            manifest.update(
                {
                    "completed_epochs": epoch,
                    "best_epoch": best_epoch,
                    "best_val_miou": best_miou,
                    "last_epoch_seconds": time.monotonic() - epoch_started,
                    "checkpoint_best_sha256": runtime.sha256(
                        OUTPUT / "checkpoint_best.pth"
                    ),
                    "checkpoint_last_sha256": runtime.sha256(
                        OUTPUT / "checkpoint_last.pth"
                    ),
                    "updated_at_utc": runtime.utc_now(),
                }
            )
            atomic_write_json(OUTPUT / "run_manifest.json", manifest)
            print(
                f"epoch_complete={epoch}/100 train_loss={row['train_loss']:.8f} "
                f"val_miou={row['val_mIoU']:.8f} skips="
                f"{row['amp_skipped_steps']} best_epoch={best_epoch} "
                f"best_miou={best_miou:.8f}",
                flush=True,
            )
        current_context = {"phase": "best_val_revalidation"}
        best = torch.load(
            OUTPUT / "checkpoint_best.pth", map_location="cpu", weights_only=False
        )
        model.load_state_dict(best["model_state_dict"])
        model.to(device)
        replay_val = formal.evaluate_split(
            model, val_paths, "val", device, weights
        )
        with (OUTPUT / "training_log.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 100:
            raise RuntimeError(f"training log has {len(rows)} rows, expected 100")
        best_row = max(rows, key=lambda item: float(item["val_mIoU"]))
        difference = abs(
            replay_val["miou"] - float(best_row["val_mIoU"])
        )
        if difference > 1e-12:
            raise RuntimeError(f"best val replay mismatch: {difference}")
        atomic_write_json(
            OUTPUT / "metrics.json",
            {
                "split": "val",
                "checkpoint_epoch": int(best["completed_epoch"]),
                "training_log_best_mIoU": float(best_row["val_mIoU"]),
                "revalidation_difference": difference,
                "matches_training_log": True,
                **replay_val,
            },
        )
        # First and only test access, after epoch100 and best-val replay.
        current_context = {"phase": "final_test"}
        manifest["test_accessed"] = True
        manifest["test_access_reason"] = (
            "epoch100 complete; best checkpoint fixed; val replay passed"
        )
        atomic_write_json(OUTPUT / "run_manifest.json", manifest)
        test_paths = sorted((ROOT / "datasets/processed/test").glob("*.npz"))
        test_metrics = formal.evaluate_split(
            model, test_paths, "test", device, weights
        )
        atomic_write_json(OUTPUT / "test_metrics.json", test_metrics)
        write_class_rows(
            OUTPUT / "per_class_metrics.csv",
            combined_class_rows(replay_val, test_metrics),
        )
        amp_summary = json.loads((OUTPUT / "amp_summary.json").read_text())
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
                "updated_at_utc": runtime.utc_now(),
                "failure_report": "failure_report.json",
            }
        )
        atomic_write_json(OUTPUT / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
