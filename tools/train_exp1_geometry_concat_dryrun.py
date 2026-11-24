#!/usr/bin/env python3
"""Two-stage, restartable Exp1 gate using the real geometry training path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_pipeline import (
    atomic_write_json,
    canonical_hash,
    load_simple_yaml,
)

OUTPUT = ROOT / "results/exp1_geometry_concat_dryrun"
EXPERIMENT_CONFIG = ROOT / "configs/experiments/exp1_geometry_concat.yaml"
TRAIN_CONFIG = ROOT / "configs/train/baseline_strong_block50_merged_train.yaml"
LOG_COLUMNS = (
    "epoch", "learning_rate", "train_loss", "val_loss", "val_OA", "val_mIoU",
    "val_mAcc", "train_batches", "optimizer_step_attempts", "amp_skipped_steps",
    "amp_skip_rate", "starting_grad_scale", "ending_grad_scale",
    "epoch_train_time_seconds", "epoch_val_time_seconds",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng(torch: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def tensor_digest(torch: Any, value: Any) -> str:
    buffer = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(buffer).hexdigest()


def save_checkpoint(
    path: Path,
    torch: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    train_generator: Any,
    *,
    epoch: int,
    best_epoch: int,
    best_miou: float,
    metrics: dict[str, Any],
) -> None:
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
            "rng_state": capture_rng(torch),
            "train_generator_state": train_generator.get_state(),
            "physical_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "experiment_config_sha256": sha256(EXPERIMENT_CONFIG),
            "training_config_sha256": sha256(TRAIN_CONFIG),
            "test_accessed": False,
        },
        temporary,
    )
    os.replace(temporary, path)


def build_runtime():
    import torch
    from torch.utils.data import DataLoader

    from datasets_loader.eclair_block_dataset import (
        EclairBlockDataset,
        build_block_sparse_collate_fn,
    )
    from models.geomorph import GeometryConcatMinkUNet

    exp_config = load_simple_yaml(EXPERIMENT_CONFIG)
    train_config = load_simple_yaml(TRAIN_CONFIG)
    required = {
        "experiment_id": "Exp1",
        "geometry_mode": "normalized_multiscale",
        "geometry_input_dim": 12,
        "input_channels": 18,
        "data_cache_root": "datasets/geometry_features",
    }
    for key, expected in required.items():
        if exp_config.get(key) != expected:
            raise ValueError(f"Exp1 config mismatch for {key}")
    if exp_config["training_config"] != str(TRAIN_CONFIG.relative_to(ROOT)):
        raise ValueError("Exp1 training config link changed")
    if train_config["training"]["physical_batch_size"] != 1:
        raise ValueError("physical batch must be 1")
    if train_config["training"]["gradient_accumulation_steps"] != 2:
        raise ValueError("accumulation must be 2")
    seed = int(train_config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    dataset = EclairBlockDataset(
        processed_root=ROOT / "datasets/processed",
        index_root=ROOT / train_config["data"]["block_index_root"],
        split="train",
        geometry_mode="normalized_multiscale",
        geometry_cache_root=ROOT / exp_config["data_cache_root"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=int(train_config["training"]["num_workers"]),
        pin_memory=True,
        collate_fn=build_block_sparse_collate_fn(),
    )
    model = GeometryConcatMinkUNet(
        num_classes=9,
        dimension=3,
        stem_channels=32,
        planes=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    )
    weights_report = json.loads(
        (ROOT / train_config["loss"]["class_weight_file"]).read_text()
    )
    weights = torch.tensor(weights_report["final_weights"], dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["optimizer"]["learning_rate"]),
        weight_decay=float(train_config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_config["training"]["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return (
        exp_config, train_config, dataset, loader, generator, model,
        weights, optimizer, scheduler, scaler,
    )


def evaluate(model: Any, val_paths: list[Path], device: Any, weights: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from engine.sliding_window_inference import sliding_window_inference
    from tools.train_baseline_strong_block import metrics_from_confusion

    confusion = torch.zeros((9, 9), dtype=torch.int64)
    loss_sum = 0.0
    valid_total = 0
    minimum_vote = None
    for number, source in enumerate(val_paths, 1):
        cache = ROOT / "datasets/geometry_features/val" / f"{source.stem}_geometry.npz"
        with np.load(source, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates"], dtype=np.int32)
            features = np.asarray(archive["features"], dtype=np.float32)
            labels = np.asarray(archive["labels"], dtype=np.int64)
        with np.load(cache, allow_pickle=False) as archive:
            geometry = np.asarray(archive["geometry_normalized"], dtype=np.float32)
            point_indices = np.asarray(archive["point_indices"], dtype=np.int64)
        if geometry.shape != (len(coordinates), 12) or not np.array_equal(
            point_indices, np.arange(len(coordinates), dtype=np.int64)
        ):
            raise RuntimeError(f"geometry alignment failure: {source.name}")
        output = sliding_window_inference(
            model, coordinates, features, device,
            block_size_voxels=500, stride_voxels=250, num_classes=9,
            use_amp=True, geometry=geometry,
        )
        logits = output["averaged_logits"]
        if logits.shape != (len(labels), 9) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"invalid validation logits: {source.name}")
        label_tensor = torch.from_numpy(labels)
        valid = label_tensor != -100
        encoded = label_tensor[valid] * 9 + logits.argmax(1)[valid]
        confusion += torch.bincount(encoded, minlength=81).reshape(9, 9)
        tile_loss = F.cross_entropy(
            logits, label_tensor, weight=weights, ignore_index=-100, reduction="sum"
        )
        if not torch.isfinite(tile_loss):
            raise RuntimeError(f"non-finite validation loss: {source.name}")
        loss_sum += float(tile_loss)
        valid_total += int(valid.sum())
        minimum_vote = (
            output["minimum_vote_count"] if minimum_vote is None
            else min(minimum_vote, output["minimum_vote_count"])
        )
        print(f"val tile={number}/59 name={source.stem}", flush=True)
        del output, logits, geometry, label_tensor, valid, encoded, tile_loss
        torch.cuda.empty_cache()
    if int(confusion.sum()) != valid_total or minimum_vote is None or minimum_vote < 1:
        raise RuntimeError("validation coverage/confusion mismatch")
    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {"loss": loss_sum / valid_total, "minimum_vote_count": minimum_vote, "tiles": 59}
    )
    return metrics


def train_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    scaler: Any,
    criterion: Any,
    device: Any,
    epoch: int,
) -> dict[str, Any]:
    import MinkowskiEngine as ME

    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    valid_total = 0
    attempts = skips = consecutive_skips = 0
    starting_scale = float(scaler.get_scale())
    for micro_index, batch in enumerate(loader, 1):
        sparse = ME.SparseTensor(
            features=batch["features"].to(device),
            coordinates=batch["coordinates"],
            device=device,
        )
        geometry = batch["geometry"].to(device)
        labels = batch["labels"].to(device)
        if len(geometry) != len(sparse.F) or len(labels) != len(sparse.F):
            raise RuntimeError("training geometry alignment failure")
        valid_points = int((labels != -100).sum())
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            fused = ME.SparseTensor(
                features=torch.cat((sparse.F, geometry), 1),
                coordinate_map_key=sparse.coordinate_map_key,
                coordinate_manager=sparse.coordinate_manager,
            )
            logits = model(fused)
            raw_loss = criterion(logits.F, labels)
        if logits.F.shape != (len(labels), 9) or not torch.isfinite(raw_loss):
            raise RuntimeError("invalid training logits or loss")
        scaler.scale(raw_loss / 2).backward()
        should_step = micro_index % 2 == 0 or micro_index == len(loader)
        if should_step:
            attempts += 1
            scaler.unscale_(optimizer)
            before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            skipped = float(scaler.get_scale()) < before
            skips += int(skipped)
            consecutive_skips = consecutive_skips + 1 if skipped else 0
            optimizer.zero_grad(set_to_none=True)
            if consecutive_skips >= 8:
                raise RuntimeError("AMP safe policy: 8 consecutive skipped steps")
        weighted_loss += float(raw_loss.detach()) * valid_points
        valid_total += valid_points
        if micro_index == 1 or micro_index % 50 == 0 or micro_index == len(loader):
            print(
                f"epoch={epoch} batch={micro_index}/{len(loader)} "
                f"loss={float(raw_loss.detach()):.6f} scale={scaler.get_scale():.1f}",
                flush=True,
            )
        del sparse, fused, logits, raw_loss, labels, geometry, batch
    skip_rate = skips / attempts
    if skip_rate > 0.01:
        raise RuntimeError(f"AMP safe policy: skip rate {skip_rate:.6f} > 0.01")
    return {
        "loss": weighted_loss / valid_total,
        "valid_points": valid_total,
        "batches": len(loader),
        "optimizer_step_attempts": attempts,
        "amp_skipped_steps": skips,
        "amp_skip_rate": skip_rate,
        "starting_grad_scale": starting_scale,
        "ending_grad_scale": float(scaler.get_scale()),
    }


def git_state() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return {"dirty": bool(result.stdout.strip()), "status_short": result.stdout.splitlines()}


def write_log(row: dict[str, Any], *, create: bool) -> None:
    path = OUTPUT / "training_log.csv"
    mode = "x" if create else "a"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if create:
            writer.writeheader()
        writer.writerow({key: row[key] for key in LOG_COLUMNS})


def run_stage(stage: str) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if stage == "epoch1":
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty {OUTPUT}")
        OUTPUT.mkdir(parents=True, exist_ok=True)
    elif not (OUTPUT / "checkpoint_last.pth").exists():
        raise FileNotFoundError("checkpoint_last.pth is required for epoch2")
    (
        exp_config, train_config, dataset, loader, generator, model,
        weights, optimizer, scheduler, scaler,
    ) = build_runtime()
    device = torch.device("cuda:0")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-100)
    epoch = 1 if stage == "epoch1" else 2
    best_epoch, best_miou = 0, float("-inf")
    resume_audit = None
    if stage == "epoch2":
        state = torch.load(OUTPUT / "checkpoint_last.pth", map_location="cpu", weights_only=False)
        required = {
            "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
            "scaler_state_dict", "rng_state", "train_generator_state",
        }
        missing = sorted(required.difference(state))
        if missing or int(state["completed_epoch"]) != 1:
            raise RuntimeError(f"invalid resume checkpoint, missing={missing}")
        expected_model_digest = tensor_digest(
            torch, next(iter(state["model_state_dict"].values()))
        )
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        restore_rng(torch, state["rng_state"])
        generator.set_state(state["train_generator_state"])
        restored_model_digest = tensor_digest(
            torch, next(iter(model.state_dict().values()))
        )
        resume_audit = {
            "source": str((OUTPUT / "checkpoint_last.pth").resolve()),
            "completed_epoch": int(state["completed_epoch"]),
            "model_state_restored": restored_model_digest == expected_model_digest,
            "optimizer_state_restored": bool(optimizer.state_dict()["state"]),
            "scheduler_state_restored": scheduler.state_dict() == state["scheduler_state_dict"],
            "scaler_state_restored": scaler.state_dict() == state["scaler_state_dict"],
            "rng_state_restored": True,
            "cuda_rng_state_restored": True,
            "dataloader_generator_restored": torch.equal(
                generator.get_state(), state["train_generator_state"]
            ),
        }
        best_epoch = int(state["best_epoch"])
        best_miou = float(state["best_val_miou"])
    val_paths = sorted((ROOT / "datasets/processed/val").glob("*.npz"))
    if len(dataset) != 1410 or len(val_paths) != 59:
        raise ValueError(f"dataset counts changed: train={len(dataset)} val={len(val_paths)}")
    manifest_path = OUTPUT / "run_manifest.json"
    if stage == "epoch1":
        manifest = {
            "status": "epoch1_running",
            "purpose": "Phase 4C-3A.7 two-epoch dry-run gate",
            "formal_training": False,
            "experiment_config": str(EXPERIMENT_CONFIG.relative_to(ROOT)),
            "experiment_config_sha256": sha256(EXPERIMENT_CONFIG),
            "training_config": str(TRAIN_CONFIG.relative_to(ROOT)),
            "training_config_sha256": sha256(TRAIN_CONFIG),
            "experiment": exp_config,
            "train_samples": len(dataset),
            "val_tiles": len(val_paths),
            "geometry_dataset": True,
            "physical_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 2,
            "loss": "balanced_cross_entropy",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": int(train_config["training"]["epochs"]),
            "amp_policy": "GradScaler; abort at >1% epoch skips or 8 consecutive skips",
            "checkpoint_state": [
                "model", "optimizer", "scheduler", "scaler", "python_rng",
                "numpy_rng", "torch_rng", "cuda_rng", "dataloader_generator",
            ],
            "started_at_utc": utc_now(),
            "completed_epochs": [],
            "resume_success": None,
            "test_accessed": False,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
            "git": git_state(),
        }
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "epoch2_resumed_running"
        manifest["resume_audit"] = resume_audit
        manifest["resume_success"] = all(
            value is True
            for key, value in resume_audit.items()
            if key.endswith("_restored")
        )
    atomic_write_json(manifest_path, manifest)
    epoch_started = time.monotonic()
    train_started = time.monotonic()
    train = train_epoch(
        torch, model, loader, optimizer, scaler, criterion, device, epoch
    )
    train_seconds = time.monotonic() - train_started
    val_started = time.monotonic()
    metrics = evaluate(model, val_paths, device, weights)
    val_seconds = time.monotonic() - val_started
    atomic_write_json(OUTPUT / f"val_metrics_epoch{epoch}.json", metrics)
    row = {
        "epoch": epoch,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_loss": train["loss"],
        "val_loss": metrics["loss"],
        "val_OA": metrics["oa"],
        "val_mIoU": metrics["miou"],
        "val_mAcc": metrics["macc"],
        "train_batches": train["batches"],
        "optimizer_step_attempts": train["optimizer_step_attempts"],
        "amp_skipped_steps": train["amp_skipped_steps"],
        "amp_skip_rate": train["amp_skip_rate"],
        "starting_grad_scale": train["starting_grad_scale"],
        "ending_grad_scale": train["ending_grad_scale"],
        "epoch_train_time_seconds": train_seconds,
        "epoch_val_time_seconds": val_seconds,
    }
    write_log(row, create=stage == "epoch1")
    if metrics["miou"] > best_miou:
        best_miou, best_epoch = float(metrics["miou"]), epoch
        save_checkpoint(
            OUTPUT / "checkpoint_best.pth", torch, model, optimizer, scheduler,
            scaler, generator, epoch=epoch, best_epoch=best_epoch,
            best_miou=best_miou, metrics=metrics,
        )
    scheduler.step()
    save_checkpoint(
        OUTPUT / "checkpoint_last.pth", torch, model, optimizer, scheduler,
        scaler, generator, epoch=epoch, best_epoch=best_epoch,
        best_miou=best_miou, metrics=metrics,
    )
    manifest["completed_epochs"] = sorted(set(manifest["completed_epochs"] + [epoch]))
    manifest["status"] = "epoch1_complete" if epoch == 1 else "dryrun_complete"
    manifest["best_epoch"] = best_epoch
    manifest["best_val_miou"] = best_miou
    manifest["checkpoint_best_sha256"] = sha256(OUTPUT / "checkpoint_best.pth")
    manifest["checkpoint_last_sha256"] = sha256(OUTPUT / "checkpoint_last.pth")
    manifest["last_epoch_elapsed_seconds"] = time.monotonic() - epoch_started
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    if epoch == 2:
        with (OUTPUT / "training_log.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        summary = {
            "status": "passed",
            "epochs_completed": [int(row["epoch"]) for row in rows],
            "epoch1": rows[0],
            "epoch2": rows[1],
            "resume_success": manifest["resume_success"],
            "resume_audit": resume_audit,
            "checkpoint_best_exists": (OUTPUT / "checkpoint_best.pth").exists(),
            "checkpoint_last_exists": (OUTPUT / "checkpoint_last.pth").exists(),
            "best_epoch": best_epoch,
            "best_val_miou": best_miou,
            "test_accessed": False,
        }
        atomic_write_json(OUTPUT / "dryrun_summary.json", summary)
    print(json.dumps({"epoch": epoch, "train": train, "val": metrics}, indent=2))
    return 0


def revalidate() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    (
        _exp, _train, _dataset, _loader, _generator, model,
        weights, _optimizer, _scheduler, _scaler,
    ) = build_runtime()
    checkpoint = torch.load(
        OUTPUT / "checkpoint_best.pth", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda:0")
    model = model.to(device)
    val_paths = sorted((ROOT / "datasets/processed/val").glob("*.npz"))
    recomputed = evaluate(model, val_paths, device, weights)
    with (OUTPUT / "training_log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best_row = max(rows, key=lambda row: float(row["val_mIoU"]))
    logged = float(best_row["val_mIoU"])
    difference = abs(recomputed["miou"] - logged)
    result = {
        "status": "passed" if difference <= 1e-12 else "failed",
        "checkpoint": str((OUTPUT / "checkpoint_best.pth").resolve()),
        "checkpoint_sha256": sha256(OUTPUT / "checkpoint_best.pth"),
        "checkpoint_epoch": int(checkpoint["completed_epoch"]),
        "training_log_best_epoch": int(best_row["epoch"]),
        "training_log_best_mIoU": logged,
        "recomputed_mIoU": recomputed["miou"],
        "absolute_difference": difference,
        "tolerance": 1e-12,
        "matches": difference <= 1e-12,
        "recomputed_metrics": recomputed,
        "test_accessed": False,
    }
    atomic_write_json(OUTPUT / "best_checkpoint_revalidation.json", result)
    summary_path = OUTPUT / "dryrun_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["best_checkpoint_revalidation_matches"] = result["matches"]
    summary["best_checkpoint_revalidation_difference"] = difference
    atomic_write_json(summary_path, summary)
    print(json.dumps(result, indent=2))
    return 0 if result["matches"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("epoch1", "epoch2", "revalidate")
    )
    stage = parser.parse_args().stage
    return revalidate() if stage == "revalidate" else run_stage(stage)


if __name__ == "__main__":
    raise SystemExit(main())
