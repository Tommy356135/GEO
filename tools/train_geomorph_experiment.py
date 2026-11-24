#!/usr/bin/env python3
"""Unified frozen GeoMorph experiment trainer and Phase 4C framework gate."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_pipeline import atomic_write_json, load_simple_yaml

CONFIGS = {
    "Exp0": "exp0_baseline.yaml",
    "Exp1": "exp1_geometry_concat.yaml",
    "Exp2": "exp2_geometry_encoder.yaml",
    "Exp3": "exp3_single_fusion.yaml",
    "Exp4": "exp4_geomorph_seg.yaml",
}
MODEL_KWARGS = {
    "num_classes": 9,
    "dimension": 3,
    "stem_channels": 32,
    "planes": (32, 64, 128, 256, 256, 128, 96, 96),
    "layers": (2, 3, 4, 6, 2, 2, 2, 2),
}
LOG_COLUMNS = (
    "epoch", "learning_rate", "train_loss", "val_loss", "val_OA",
    "val_mIoU", "val_mAcc", "train_batches", "optimizer_step_attempts",
    "amp_skipped_steps", "amp_skip_rate", "starting_grad_scale",
    "minimum_grad_scale", "ending_grad_scale", "validation_dtype",
    "validation_windows", "invalid_logits_count",
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


def model_factory(experiment: str) -> Any:
    from models.backbone.minkowski.MinkUNet34C import MinkUNet34C
    from models.geomorph import (
        GeoMorphMinkUNet,
        GeometryConcatMinkUNet,
        GeometryEncodedInputMinkUNet,
        GeometrySingleFusionMinkUNet,
    )

    classes = {
        "Exp0": MinkUNet34C,
        "Exp1": GeometryConcatMinkUNet,
        "Exp2": GeometryEncodedInputMinkUNet,
        "Exp3": GeometrySingleFusionMinkUNet,
        "Exp4": GeoMorphMinkUNet,
    }
    if experiment == "Exp0":
        return classes[experiment](in_channels=6, **MODEL_KWARGS)
    return classes[experiment](**MODEL_KWARGS)


def forward_geometry_model(model: Any, sparse: Any, geometry: Any) -> Any:
    import MinkowskiEngine as ME
    import torch

    if geometry is None:
        return model(sparse)
    if model.__class__.__name__ == "GeometryConcatMinkUNet":
        fused = ME.SparseTensor(
            features=torch.cat((sparse.F, geometry), dim=1),
            coordinate_map_key=sparse.coordinate_map_key,
            coordinate_manager=sparse.coordinate_manager,
        )
        return model(fused)
    return model(sparse, geometry)


def build_runtime(experiment: str) -> tuple[Any, ...]:
    import torch
    from torch.utils.data import DataLoader

    from datasets_loader.eclair_block_dataset import (
        EclairBlockDataset,
        build_block_sparse_collate_fn,
    )

    experiment_path = ROOT / "configs/experiments" / CONFIGS[experiment]
    experiment_config = load_simple_yaml(experiment_path)
    train_path = ROOT / experiment_config["training_config"]
    train_config = load_simple_yaml(train_path)
    geometry_mode = experiment_config.get("geometry_mode", "none")
    if experiment != "Exp0":
        if geometry_mode != "normalized_multiscale":
            raise RuntimeError("geometry experiment must use normalized_multiscale")
        if int(experiment_config["geometry_input_dim"]) != 12:
            raise RuntimeError("geometry input dimension changed")
    frozen = train_config["training"]
    if (
        int(frozen["physical_batch_size"]) != 1
        or int(frozen["gradient_accumulation_steps"]) != 2
        or not bool(frozen["amp"])
        or int(frozen["seed"]) != 42
        or train_config["optimizer"]["name"] != "AdamW"
        or float(train_config["optimizer"]["learning_rate"]) != 1e-3
        or train_config["scheduler"]["name"] != "CosineAnnealingLR"
        or train_config["loss"]["mode"] != "ce_balanced"
        or bool(train_config["sampling"]["class_aware"])
        or float(train_config["data"]["voxel_size"]) != 0.1
    ):
        raise RuntimeError("frozen experiment setting changed")
    seed = int(frozen["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    dataset = EclairBlockDataset(
        processed_root=ROOT / "datasets/processed",
        index_root=ROOT / train_config["data"]["block_index_root"],
        split="train",
        geometry_mode=geometry_mode,
        geometry_cache_root=(
            None if geometry_mode == "none"
            else ROOT / experiment_config["data_cache_root"]
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=int(frozen["num_workers"]),
        pin_memory=True,
        collate_fn=build_block_sparse_collate_fn(),
    )
    model = model_factory(experiment)
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
        optimizer, T_max=int(frozen["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return (
        experiment_config, train_config, experiment_path, train_path,
        dataset, loader, generator, model, weights, optimizer, scheduler,
        scaler,
    )


def train_one_epoch(
    model: Any, loader: Any, optimizer: Any, criterion: Any,
    device: Any, scaler: Any, epoch: int, accumulation_steps: int = 2,
) -> dict[str, Any]:
    import MinkowskiEngine as ME
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    valid_total = attempts = skips = 0
    start_scale = float(scaler.get_scale())
    minimum_scale = start_scale
    skip_context = []
    for micro_index, batch in enumerate(loader, 1):
        sparse = ME.SparseTensor(
            features=batch["features"].to(device),
            coordinates=batch["coordinates"], device=device,
        )
        geometry = (
            batch["geometry"].to(device) if "geometry" in batch else None
        )
        labels = batch["labels"].to(device)
        if geometry is not None and len(geometry) != len(sparse.F):
            raise RuntimeError("geometry alignment failure")
        valid_points = int((labels != -100).sum())
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            logits = forward_geometry_model(model, sparse, geometry)
            raw_loss = criterion(logits.F, labels)
        if logits.F.shape != (len(labels), 9) or not bool(torch.isfinite(raw_loss)):
            raise RuntimeError("invalid training logits/loss")
        scaler.scale(raw_loss / accumulation_steps).backward()
        should_step = (
            micro_index % accumulation_steps == 0
            or micro_index == len(loader)
        )
        if should_step:
            attempts += 1
            before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            after = float(scaler.get_scale())
            minimum_scale = min(minimum_scale, after)
            if after < before:
                skips += 1
                skip_context.append(
                    {
                        "micro_batch_index": micro_index,
                        "tile_name": batch["tile_name"][0],
                        "block_name": batch["block_id"][0],
                        "scale_before": before,
                        "scale_after": after,
                    }
                )
            optimizer.zero_grad(set_to_none=True)
        weighted_loss += float(raw_loss.detach()) * valid_points
        valid_total += valid_points
        if micro_index == 1 or micro_index % 100 == 0 or micro_index == len(loader):
            print(
                f"epoch={epoch} batch={micro_index}/{len(loader)} "
                f"loss={float(raw_loss.detach()):.6f} "
                f"scale={float(scaler.get_scale()):.1f} skips={skips}",
                flush=True,
            )
        del sparse, geometry, labels, logits, raw_loss, batch
    return {
        "loss": weighted_loss / valid_total,
        "valid_points": valid_total,
        "batches": len(loader),
        "optimizer_step_attempts": attempts,
        "amp_skipped_steps": skips,
        "amp_skip_rate": skips / attempts,
        "starting_grad_scale": start_scale,
        "minimum_grad_scale": minimum_scale,
        "ending_grad_scale": float(scaler.get_scale()),
        "skip_context": skip_context,
    }


def validate(
    model: Any, tile_paths: list[Path], device: Any, class_weights: Any,
    geometry_cache_root: Path | None, split: str = "val",
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from engine.sliding_window_inference import sliding_window_inference
    from tools.train_baseline_strong_block import metrics_from_confusion

    expected = 59 if split == "val" else 106
    if split not in {"val", "test"} or len(tile_paths) != expected:
        raise RuntimeError(
            f"{split} evaluation requires {expected} tiles, got {len(tile_paths)}"
        )
    confusion = torch.zeros((9, 9), dtype=torch.int64)
    loss_sum = 0.0
    valid_total = windows = invalid_logits = 0
    for tile_index, path in enumerate(tile_paths, 1):
        with np.load(path, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates"], dtype=np.int32)
            features = np.asarray(archive["features"], dtype=np.float32)
            labels = np.asarray(archive["labels"], dtype=np.int64)
        geometry = None
        if geometry_cache_root is not None:
            cache_path = (
                geometry_cache_root / split / f"{path.stem}_geometry.npz"
            )
            with np.load(cache_path, allow_pickle=False) as archive:
                geometry = np.asarray(
                    archive["geometry_normalized"], dtype=np.float32
                )
                point_indices = np.asarray(
                    archive["point_indices"], dtype=np.int64
                )
            if not np.array_equal(
                point_indices, np.arange(len(coordinates), dtype=np.int64)
            ):
                raise RuntimeError(f"geometry mapping error: {path.name}")
        output = sliding_window_inference(
            model, coordinates, features, device,
            block_size_voxels=500, stride_voxels=250,
            num_classes=9, use_amp=False, geometry=geometry,
        )
        logits = output["averaged_logits"]
        windows += int(output["window_count"])
        finite = bool(torch.isfinite(logits).all())
        invalid_logits += int((~torch.isfinite(logits)).sum())
        if (
            logits.shape != (len(labels), 9)
            or output["logits_dtype"] != "torch.float32"
            or not finite
        ):
            raise RuntimeError(f"invalid FP32 validation logits: {path.name}")
        label_tensor = torch.from_numpy(labels)
        valid = label_tensor != -100
        encoded = label_tensor[valid] * 9 + logits.argmax(1)[valid]
        confusion += torch.bincount(encoded, minlength=81).reshape(9, 9)
        tile_loss = F.cross_entropy(
            logits, label_tensor, weight=class_weights,
            ignore_index=-100, reduction="sum",
        )
        loss_sum += float(tile_loss)
        valid_total += int(valid.sum())
        print(
            f"{split}_tile={tile_index}/{expected} name={path.stem} "
            f"windows={output['window_count']} dtype=torch.float32 finite={finite}",
            flush=True,
        )
    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {
            "loss": loss_sum / valid_total,
            "split": split,
            "tiles": len(tile_paths),
            "window_count": windows,
            "validation_dtype": "float32",
            "logits_dtype": "torch.float32",
            "invalid_logits_count": invalid_logits,
            "logits_finite": invalid_logits == 0,
        }
    )
    return metrics


def save_checkpoint(
    path: Path, model: Any, optimizer: Any, scheduler: Any, scaler: Any,
    epoch: int, best_epoch: int, best_miou: float, generator: Any,
    val_metrics: dict[str, Any], experiment_path: Path, train_path: Path,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_miou": best_miou,
                "rng_state": capture_rng(torch),
                "train_generator_state": generator.get_state(),
                "val_metrics": val_metrics,
                "experiment_config_sha256": sha256(experiment_path),
                "training_config_sha256": sha256(train_path),
                "test_accessed": False,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path, model: Any, optimizer: Any, scheduler: Any, scaler: Any,
    generator: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict", "rng_state", "train_generator_state",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(f"checkpoint missing states: {missing}")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    restore_rng(torch, state["rng_state"])
    generator.set_state(state["train_generator_state"])
    audit = {
        "source_checkpoint": str(path),
        "source_completed_epoch": int(state["completed_epoch"]),
        "model_state_restored": True,
        "optimizer_state_restored": bool(optimizer.state_dict()["state"]),
        "scheduler_state_restored": scheduler.state_dict()
        == state["scheduler_state_dict"],
        "scaler_state_restored": scaler.state_dict()
        == state["scaler_state_dict"],
        "python_rng_restored": True,
        "numpy_rng_restored": True,
        "torch_rng_restored": True,
        "cuda_rng_restored": True,
        "dataloader_generator_restored": torch.equal(
            generator.get_state(), state["train_generator_state"]
        ),
    }
    return state, audit


def largest_sample_smoke(experiment: str) -> dict[str, Any]:
    import MinkowskiEngine as ME
    import torch

    (
        _exp, _train, _ep, _tp, dataset, _loader, _generator, model,
        weights, optimizer, _scheduler, scaler,
    ) = build_runtime(experiment)
    target = next(
        index for index, path in enumerate(dataset.index_files)
        if path.name == "pointcloud_883__x1_y1.npz"
    )
    sample = dataset[target]
    coordinates, features, labels = ME.utils.SparseCollation()(
        [(sample["coordinates"], sample["features"], sample["labels"])]
    )
    geometry = sample.get("geometry")
    device = torch.device("cuda:0")
    model.to(device).train()
    sparse = ME.SparseTensor(
        features=features.float().to(device), coordinates=coordinates,
        device=device,
    )
    geometry_device = geometry.float().to(device) if geometry is not None else None
    labels_device = labels.long().to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights.to(device), ignore_index=-100
    )
    embedding: dict[str, Any] = {}
    hook = None
    if hasattr(model, "geometry_encoder"):
        def record_embedding(_module: Any, _inputs: Any, output: Any) -> None:
            embedding.update(
                {
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "finite": bool(torch.isfinite(output).all()),
                }
            )
        hook = model.geometry_encoder.register_forward_hook(record_embedding)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    error = None
    logits = loss = None
    try:
        with torch.autocast("cuda", dtype=torch.float16, enabled=True):
            logits = forward_geometry_model(model, sparse, geometry_device)
            loss = criterion(logits.F, labels_device)
        scaler.scale(loss / 2).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        gradient_finite = bool(gradients) and all(
            bool(torch.isfinite(value).all()) for value in gradients
        )
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        gradient_finite = False
    result = {
        "status": "passed" if error is None and gradient_finite else "failed",
        "experiment_id": experiment,
        "tile_name": sample["tile_name"],
        "block_name": sample["block_id"],
        "index_file": dataset.index_files[target].name,
        "voxel_count": len(features),
        "baseline_features_shape": list(features.shape),
        "baseline_features_finite": bool(torch.isfinite(features).all()),
        "geometry_shape": list(geometry.shape),
        "geometry_finite": bool(torch.isfinite(geometry).all()),
        "embedding": embedding,
        "logits_shape": list(logits.F.shape) if logits is not None else None,
        "logits_finite": bool(
            logits is not None and torch.isfinite(logits.F).all()
        ),
        "loss": float(loss.detach()) if loss is not None else None,
        "loss_finite": bool(loss is not None and torch.isfinite(loss)),
        "backward_completed": error is None,
        "gradient_finite": gradient_finite,
        "optimizer_step_executed": False,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "error": error,
    }
    if hook is not None:
        hook.remove()
    del model, sparse, geometry_device, labels_device, logits, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def append_log(path: Path, row: dict[str, Any], create: bool) -> None:
    with path.open("x" if create else "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if create:
            writer.writeheader()
        writer.writerow({key: row[key] for key in LOG_COLUMNS})


def run_training(
    experiment: str, output: Path, epochs: int, dry_run: bool,
) -> None:
    import torch

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    if dry_run:
        smoke = largest_sample_smoke(experiment)
        atomic_write_json(output / "smoke.json", smoke)
        atomic_write_json(
            output / "memory_gate.json",
            {
                "status": smoke["status"],
                "sample": smoke["index_file"],
                "voxel_count": smoke["voxel_count"],
                "peak_allocated_bytes": smoke["peak_allocated_bytes"],
                "peak_reserved_bytes": smoke["peak_reserved_bytes"],
                "device_total_bytes": smoke["device_total_bytes"],
                "oom": smoke["error"] is not None
                and "out of memory" in smoke["error"].lower(),
            },
        )
        if smoke["status"] != "passed":
            raise RuntimeError("maximum-sample memory smoke failed")
    best_epoch = 0
    best_miou = float("-inf")
    resume_audit = None
    amp_epochs: list[dict[str, Any]] = []
    val_paths = sorted((ROOT / "datasets/processed/val").glob("*.npz"))
    log_path = output / ("dryrun_log.csv" if dry_run else "training_log.csv")
    manifest = {
        "status": "dry_run" if dry_run else "training",
        "experiment_id": experiment,
        "formal_training": not dry_run,
        "target_epochs": epochs,
        "completed_epochs": 0,
        "train_samples": 1410,
        "val_tiles": 59,
        "test_accessed": False,
        "test_evaluations": 0,
        "physical_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "scheduler": "CosineAnnealingLR",
        "loss": "balanced_cross_entropy",
        "amp_policy": "standard_gradscaler_overflow_skip",
        "validation_dtype": "float32",
        "started_at_utc": utc_now(),
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    for epoch in range(1, epochs + 1):
        runtime = build_runtime(experiment)
        (
            experiment_config, _train_config, experiment_path, train_path,
            dataset, loader, generator, model, weights, optimizer, scheduler,
            scaler,
        ) = runtime
        if len(dataset) != 1410 or len(val_paths) != 59:
            raise RuntimeError("train/val split counts changed")
        device = torch.device("cuda:0")
        model.to(device)
        if epoch > 1:
            state, resume_audit = load_checkpoint(
                output / "checkpoint_last.pth", model, optimizer,
                scheduler, scaler, generator,
            )
            if int(state["completed_epoch"]) != epoch - 1:
                raise RuntimeError("resume epoch mismatch")
            best_epoch = int(state["best_epoch"])
            best_miou = float(state["best_val_miou"])
        criterion = torch.nn.CrossEntropyLoss(
            weight=weights.to(device), ignore_index=-100
        )
        train_metrics = train_one_epoch(
            model, loader, optimizer, criterion, device, scaler,
            epoch=epoch, accumulation_steps=2,
        )
        geometry_root = (
            None if experiment == "Exp0"
            else ROOT / experiment_config["data_cache_root"]
        )
        val_metrics = validate(
            model, val_paths, device, weights,
            geometry_cache_root=geometry_root, split="val",
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_OA": val_metrics["oa"],
            "val_mIoU": val_metrics["miou"],
            "val_mAcc": val_metrics["macc"],
            "train_batches": train_metrics["batches"],
            "optimizer_step_attempts": train_metrics["optimizer_step_attempts"],
            "amp_skipped_steps": train_metrics["amp_skipped_steps"],
            "amp_skip_rate": train_metrics["amp_skip_rate"],
            "starting_grad_scale": train_metrics["starting_grad_scale"],
            "minimum_grad_scale": train_metrics["minimum_grad_scale"],
            "ending_grad_scale": train_metrics["ending_grad_scale"],
            "validation_dtype": val_metrics["validation_dtype"],
            "validation_windows": val_metrics["window_count"],
            "invalid_logits_count": val_metrics["invalid_logits_count"],
        }
        append_log(log_path, row, create=epoch == 1)
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
                "skip_context": train_metrics["skip_context"],
            }
        )
        atomic_write_json(
            output / "amp_summary.json",
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
            best_epoch = epoch
            best_miou = float(val_metrics["miou"])
            save_checkpoint(
                output / "checkpoint_best.pth", model, optimizer,
                scheduler, scaler, epoch, best_epoch, best_miou,
                generator, val_metrics, experiment_path, train_path,
            )
        scheduler.step()
        save_checkpoint(
            output / "checkpoint_last.pth", model, optimizer,
            scheduler, scaler, epoch, best_epoch, best_miou,
            generator, val_metrics, experiment_path, train_path,
        )
        manifest.update(
            {
                "completed_epochs": epoch,
                "best_epoch": best_epoch,
                "best_val_mIoU": best_miou,
                "checkpoint_best_sha256": sha256(
                    output / "checkpoint_best.pth"
                ),
                "checkpoint_last_sha256": sha256(
                    output / "checkpoint_last.pth"
                ),
                "last_validation_windows": val_metrics["window_count"],
                "last_invalid_logits_count": val_metrics[
                    "invalid_logits_count"
                ],
                "updated_at_utc": utc_now(),
            }
        )
        atomic_write_json(output / "run_manifest.json", manifest)
        print(
            f"epoch_complete={epoch}/{epochs} train_loss={row['train_loss']:.8f} "
            f"val_miou={row['val_mIoU']:.8f} skips={row['amp_skipped_steps']}",
            flush=True,
        )
        del runtime, model, optimizer, scheduler, scaler, loader, dataset
        gc.collect()
        torch.cuda.empty_cache()
    atomic_write_json(
        output / "resume_check.json",
        {
            "status": "passed" if resume_audit and all(
                value is True for key, value in resume_audit.items()
                if key.endswith("_restored")
            ) else "failed",
            "formal_training_started": not dry_run,
            "dry_run_epochs": epochs if dry_run else 0,
            "test_accessed": False,
            "resume": resume_audit,
            "best_epoch": best_epoch,
            "best_val_mIoU": best_miou,
        },
    )
    if dry_run:
        return

    # Rebuild a clean runtime, load the fixed best checkpoint and revalidate.
    (
        experiment_config, _train_config, _experiment_path, _train_path,
        _dataset, _loader, _generator, model, weights, optimizer, scheduler,
        scaler,
    ) = build_runtime(experiment)
    best_state, _best_restore = load_checkpoint(
        output / "checkpoint_best.pth", model, optimizer, scheduler,
        scaler, _generator,
    )
    device = torch.device("cuda:0")
    model.to(device)
    geometry_root = ROOT / experiment_config["data_cache_root"]
    best_val = validate(
        model, val_paths, device, weights,
        geometry_cache_root=geometry_root, split="val",
    )
    if abs(best_val["miou"] - best_miou) > 1e-12:
        raise RuntimeError("best checkpoint validation replay mismatch")

    # Current cache-integrity gate; no test label is read here.
    from tools.audit_test_geometry_cache import file_sha256, row_for

    test_sources = sorted((ROOT / "datasets/processed/test").glob("*.npz"))
    test_cache_root = ROOT / "datasets/geometry_features/test"
    expected_cache = {
        "geometry_version": "geomorph_geometry_v2",
        "normalization_hash": file_sha256(
            ROOT / "configs/geometry/geomorph_normalization.json"
        ),
    }
    cache_rows = [
        row_for(
            source, test_cache_root / f"{source.stem}_geometry.npz",
            expected_cache,
        )
        for source in test_sources
    ]
    if len(test_sources) != 106 or any(
        row["status"] != "valid" for row in cache_rows
    ):
        raise RuntimeError("test geometry cache integrity gate failed")

    # First test-label access: epoch100 complete, best fixed and replayed.
    manifest.update(
        {
            "status": "final_test",
            "test_accessed": True,
            "test_access_reason": (
                "epoch100 complete; best checkpoint fixed; val replay passed"
            ),
        }
    )
    atomic_write_json(output / "run_manifest.json", manifest)
    test_metrics = validate(
        model, test_sources, device, weights,
        geometry_cache_root=geometry_root, split="test",
    )
    atomic_write_json(output / "metrics_val_best.json", best_val)
    atomic_write_json(output / "test_metrics.json", test_metrics)
    from tools.train_exp1_geometry_concat_formal import (
        class_rows, write_classes,
    )
    write_classes(output / "test_per_class_metrics.csv", class_rows(test_metrics))
    amp_summary = json.loads((output / "amp_summary.json").read_text())
    amp_summary.update(
        {
            "status": "complete",
            "total_skip_rate": amp_summary["total_skipped_steps"]
            / amp_summary["total_optimizer_step_attempts"],
            "test_accessed_after_best_selection": True,
        }
    )
    atomic_write_json(output / "amp_summary.json", amp_summary)
    manifest.update(
        {
            "status": "complete",
            "test_evaluations": 1,
            "test_checkpoint_epoch": int(best_state["completed_epoch"]),
            "test_checkpoint_sha256": sha256(
                output / "checkpoint_best.pth"
            ),
            "best_val_revalidation_matches": True,
            "test_cache_valid_tiles": 106,
            "ended_at_utc": utc_now(),
        }
    )
    atomic_write_json(output / "run_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=CONFIGS, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.smoke_only:
        print(json.dumps(largest_sample_smoke(args.experiment), indent=2))
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required for training/dry-run")
    if args.dry_run and args.epochs != 2:
        raise ValueError("Phase 4C dry-run must be exactly two epochs")
    if not args.dry_run and args.epochs != 100:
        raise ValueError("formal training must be exactly 100 epochs")
    run_training(args.experiment, args.output_dir, args.epochs, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
