"""Baseline MinkowskiEngine training loop and experiment artifact interfaces."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def set_random_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch, and every visible CUDA device."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _make_sparse_input(batch: Mapping[str, Any], device: Any) -> tuple[Any, Any]:
    import torch
    import MinkowskiEngine as ME

    coordinates = batch["coordinates"]
    features = batch["features"]
    labels = batch["labels"].to(device=device, dtype=torch.long, non_blocking=True)
    sparse_input = ME.SparseTensor(
        features=features.to(device=device, dtype=torch.float32, non_blocking=True),
        coordinates=coordinates,
        device=device,
    )
    return sparse_input, labels


def train_one_epoch(
    model: Any,
    data_loader: Iterable[Mapping[str, Any]],
    optimizer: Any,
    criterion: Any,
    device: Any,
    *,
    epoch: int,
    scaler: Any | None = None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    """Train the baseline for one pass over ``data_loader``."""
    import torch

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    model.train()
    total_loss = 0.0
    total_valid_points = 0
    batch_count = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in data_loader:
        sparse_input, labels = _make_sparse_input(batch, device)
        valid_points = int((labels != -100).sum().item())
        if valid_points == 0:
            raise ValueError("training batch contains only ignore_index=-100 labels")
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            logits = model(sparse_input)
            if logits.F.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"logit/label length mismatch: {logits.F.shape[0]} != {labels.shape[0]}"
                )
            raw_loss = criterion(logits.F, labels)
        if not torch.isfinite(raw_loss):
            raise RuntimeError(f"non-finite training loss at epoch {epoch}")
        scaled_loss = raw_loss / gradient_accumulation_steps
        if use_amp:
            if scaler is None:
                raise ValueError("AMP training requires a GradScaler")
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = (batch_count + 1) % gradient_accumulation_steps == 0
        total_loss += float(raw_loss.detach()) * valid_points
        total_valid_points += valid_points
        batch_count += 1
        if should_step:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        del sparse_input, logits, raw_loss, scaled_loss, labels, batch
    if batch_count and batch_count % gradient_accumulation_steps:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    if batch_count == 0 or total_valid_points == 0:
        raise ValueError("training data loader produced no batches")
    return {
        "epoch": float(epoch),
        "loss": total_loss / total_valid_points,
        "batches": float(batch_count),
        "optimizer_steps": float(optimizer_steps),
        "valid_points": float(total_valid_points),
    }


def prepare_experiment_directory(output_dir: Path | str) -> dict[str, Path]:
    """Create the Baseline artifact directory and return its fixed paths."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "checkpoint": root / "checkpoint_best.pth",
        "training_log": root / "training_log.csv",
        "metrics": root / "metrics.json",
        "confusion_matrix": root / "confusion_matrix.csv",
        "per_class_iou": root / "per_class_iou.csv",
        "test_confusion_matrix": root / "test_confusion_matrix.csv",
    }


def save_checkpoint(
    path: Path | str,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any | None,
    epoch: int,
    metrics: Mapping[str, Any],
) -> None:
    """Save the best restartable training state."""
    import torch

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "metrics": dict(metrics),
        },
        Path(path),
    )


def append_training_log(path: Path | str, row: Mapping[str, Any]) -> None:
    """Append one epoch to the fixed CSV training log."""
    columns = (
        "epoch", "train_loss", "val_loss", "OA", "mIoU",
        "mAcc", "learning_rate",
    )
    target = Path(path)
    write_header = not target.exists()
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row[column] for column in columns})


def save_evaluation_artifacts(
    metrics_path: Path | str,
    confusion_matrix_path: Path | str,
    metrics: Mapping[str, Any],
) -> None:
    """Write JSON metrics and a numeric confusion matrix CSV."""
    public_metrics = {
        key: value for key, value in metrics.items() if key != "confusion_matrix"
    }
    Path(metrics_path).write_text(
        json.dumps(public_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_confusion_matrix(confusion_matrix_path, metrics["confusion_matrix"])


def save_confusion_matrix(path: Path | str, matrix: Any) -> None:
    """Write one labeled confusion matrix CSV."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *range(len(matrix))])
        for class_id, row in enumerate(matrix):
            writer.writerow([class_id, *row])


def save_per_class_iou(
    path: Path | str,
    val_iou: list[float | None],
    test_iou: list[float | None],
) -> None:
    """Write best-checkpoint val/test IoU for every training class."""
    if len(val_iou) != len(test_iou):
        raise ValueError("val/test per-class IoU lengths differ")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "val_iou", "test_iou"])
        for class_id, (val_value, test_value) in enumerate(zip(val_iou, test_iou)):
            writer.writerow([class_id, val_value, test_value])
