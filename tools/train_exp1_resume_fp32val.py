#!/usr/bin/env python3
"""Resume frozen Exp1 from epoch 7 through 100 with FP32 validation."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_pipeline import atomic_write_json
import tools.train_exp1_geometry_concat_dryrun as runtime
import tools.train_exp1_geometry_concat_final_v2 as final_v2
import tools.train_exp1_geometry_concat_resume_final as resume

SOURCE = ROOT / "results/exp1_geometry_concat_final_v2"
OUTPUT = ROOT / "results/exp1_geometry_concat_final_v2_resume_fp32val"
CHECKPOINT = SOURCE / "checkpoint_last.pth"


def copy_history() -> list[dict]:
    with (SOURCE / "training_log.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(1, 8)):
        raise RuntimeError("source training history is not epochs 1..7")
    with (OUTPUT / "training_log.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=resume.LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in resume.LOG_COLUMNS}
            for row in rows
        )
    amp = json.loads((SOURCE / "amp_summary.json").read_text())
    if [int(item["epoch"]) for item in amp["epochs"]] != list(range(1, 8)):
        raise RuntimeError("source AMP history is not epochs 1..7")
    return amp["epochs"]


def append_training_row(row: dict) -> None:
    with (OUTPUT / "training_log.csv").open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        csv.DictWriter(handle, fieldnames=resume.LOG_COLUMNS).writerow(
            {key: row[key] for key in resume.LOG_COLUMNS}
        )


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    source_manifest = json.loads(
        (SOURCE / "run_manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest["test_accessed"]
        or int(source_manifest["test_evaluations"]) != 0
        or int(source_manifest["completed_epochs"]) != 7
    ):
        raise RuntimeError("source resume/test-isolation gate failed")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CHECKPOINT, OUTPUT / "checkpoint_last.pth")
    shutil.copy2(
        SOURCE / "checkpoint_best.pth", OUTPUT / "checkpoint_best.pth"
    )
    amp_epochs = copy_history()
    final_v2.OUTPUT = OUTPUT
    (
        _experiment, train_config, dataset, loader, generator, model,
        weights, optimizer, scheduler, scaler,
    ) = runtime.build_runtime()
    if (
        len(dataset) != 1410
        or int(train_config["training"]["gradient_accumulation_steps"]) != 2
    ):
        raise RuntimeError("frozen train data/accumulation gate failed")
    val_paths = sorted((ROOT / "datasets/processed/val").glob("*.npz"))
    if len(val_paths) != 59:
        raise RuntimeError("val tile count changed")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(state["completed_epoch"]) != 7:
        raise RuntimeError("resume checkpoint is not epoch7")

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
        "status": "training",
        "experiment_id": "Exp1",
        "model_name": "Geometry-Concat",
        "output_dir": str(OUTPUT.relative_to(ROOT)),
        "resume_source": str(CHECKPOINT.relative_to(ROOT)),
        "resume_source_sha256": runtime.sha256(CHECKPOINT),
        "resume_completed_epoch": 7,
        "completed_epochs": 7,
        "target_epochs": 100,
        "train_merged_samples": 1410,
        "val_tiles": 59,
        "test_isolated_until_final_best": True,
        "test_accessed": False,
        "test_evaluations": 0,
        "frozen_settings": source_manifest["frozen_settings"],
        "amp_policy": "standard_gradscaler_overflow_skip",
        "validation": {
            **source_manifest["validation"],
            "dtype": "float32",
            "autocast_enabled": False,
            "invalid_logits_fatal": True,
            "skip_invalid_tiles": False,
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
    context = {}
    try:
        for epoch in range(8, 101):
            epoch_started = time.monotonic()
            context = {"phase": "train", "epoch": epoch}
            train_started = time.monotonic()
            train_metrics = resume.train_epoch(
                torch, model, loader, optimizer, scaler,
                criterion, device, epoch,
            )
            train_seconds = time.monotonic() - train_started
            context = {"phase": "pre_validation_finite", "epoch": epoch}
            finite_audit = final_v2.append_finite_audit(epoch, model)
            context = {"phase": "val", "epoch": epoch}
            val_started = time.monotonic()
            val_metrics = final_v2.evaluate_logged(
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
                    "starting_scale": train_metrics["starting_grad_scale"],
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
                    "last_validation_dtype": "float32",
                    "last_validation_window_count": val_metrics[
                        "window_count"
                    ] if "window_count" in val_metrics else None,
                    "last_validation_invalid_logits": 0,
                    "last_pre_validation_finite": finite_audit,
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
                f"epoch_complete={epoch}/100 train_loss="
                f"{row['train_loss']:.8f} val_miou="
                f"{row['val_mIoU']:.8f} skips="
                f"{row['amp_skipped_steps']} best_epoch={best_epoch} "
                f"best_miou={best_miou:.8f}",
                flush=True,
            )

        context = {"phase": "best_val_revalidation"}
        best = torch.load(
            OUTPUT / "checkpoint_best.pth",
            map_location="cpu", weights_only=False,
        )
        model.load_state_dict(best["model_state_dict"])
        model.to(device)
        final_v2.append_finite_audit("best_revalidation", model)
        replay_val = final_v2.evaluate_logged(
            model, val_paths, "val", device, weights,
            epoch="best", phase="best_revalidation",
        )
        with (OUTPUT / "training_log.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 100:
            raise RuntimeError(f"training log rows={len(rows)}, expected 100")
        best_row = max(rows, key=lambda item: float(item["val_mIoU"]))
        difference = abs(
            replay_val["miou"] - float(best_row["val_mIoU"])
        )
        if difference > 1e-12:
            raise RuntimeError(f"best val replay mismatch: {difference}")

        # First and only test access after epoch100 and best-val replay.
        context = {"phase": "final_test"}
        manifest["test_accessed"] = True
        manifest["test_access_reason"] = (
            "epoch100 complete; best fixed; FP32 59-tile val replay passed"
        )
        atomic_write_json(OUTPUT / "run_manifest.json", manifest)
        test_paths = sorted(
            (ROOT / "datasets/processed/test").glob("*.npz")
        )
        final_v2.append_finite_audit("final_test", model)
        test_metrics = final_v2.evaluate_logged(
            model, test_paths, "test", device, weights,
            epoch="best", phase="final_test",
        )
        metrics = {
            "checkpoint_epoch": int(best["completed_epoch"]),
            "training_log_best_mIoU": float(best_row["val_mIoU"]),
            "best_val_revalidation": replay_val,
            "best_val_revalidation_difference": difference,
            "best_val_revalidation_matches": True,
            "test": test_metrics,
            "validation_dtype": "float32",
            "test_dtype": "float32",
        }
        atomic_write_json(OUTPUT / "metrics.json", metrics)
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
        atomic_write_json(
            OUTPUT / "failure_report.json",
            {
                "status": "success",
                "failure": None,
                "completed_epochs": 100,
                "test_evaluations": 1,
                "timestamp_utc": runtime.utc_now(),
            },
        )
        return 0
    except BaseException as error:
        atomic_write_json(
            OUTPUT / "failure_report.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "context": context,
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
