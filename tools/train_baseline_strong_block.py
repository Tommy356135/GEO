#!/usr/bin/env python3
"""Train frozen Baseline-Strong-Block75 and evaluate full tiles by sliding windows."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geometry.geometry_pipeline import atomic_write_json, canonical_hash, load_simple_yaml  # noqa: E402


class Tee:
    def __init__(self, terminal: Any, handle: Any) -> None:
        self.terminal, self.handle = terminal, handle

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.handle.write(value)
        self.handle.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.npz")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "UNBORN_HEAD",
        "dirty": bool(status.stdout.strip()),
        "status_short": status.stdout.splitlines(),
    }


def validate_config(config: dict[str, Any]) -> None:
    is_gate = config["experiment_name"] == "Baseline-Strong-Block50-MergedSmall-AmpResumeGate"
    is_merged = config["experiment_name"] in {
        "Baseline-Strong-Block50-MergedSmall",
        "Baseline-Strong-Block50-MergedSmall-AmpResumeGate",
        "Baseline-Strong-Block50-MergedSmall-AmpSafe",
    }
    is_block50 = config["experiment_name"] in {
        "Baseline-Strong-Block50", "Baseline-Strong-Block50-MergedSmall",
        "Baseline-Strong-Block50-MergedSmall-AmpResumeGate",
        "Baseline-Strong-Block50-MergedSmall-AmpSafe",
    }
    expected_block = 50.0 if is_block50 else 75.0
    expected_stride = 25.0 if is_block50 else 37.5
    expected_output = (
        (
            (
                "results/baseline_strong_block50_merged_amp_resume_gate"
                if is_gate else (
                    "results/baseline_strong_block50_merged_amp_safe"
                    if config["experiment_name"]
                    == "Baseline-Strong-Block50-MergedSmall-AmpSafe"
                    else "results/baseline_strong_block50_merged"
                )
            )
            if is_merged
            else "results/baseline_strong_block50"
        )
        if is_block50 else "results/baseline_strong_block75"
    )
    checks = {
        "experiment": config["experiment_name"] in {
            "Baseline-Strong-Block50", "Baseline-Strong-Block50-MergedSmall",
            "Baseline-Strong-Block50-MergedSmall-AmpResumeGate",
            "Baseline-Strong-Block50-MergedSmall-AmpSafe",
            "Baseline-Strong-Block75"
        },
        "output": config["output_dir"] == expected_output,
        "model": config["model"]["name"] == "MinkUNet34C",
        "channels": config["model"]["in_channels"] == 6,
        "classes": config["model"]["num_classes"] == 9,
        "dimension": config["model"]["dimension"] == 3,
        "voxel": config["data"]["voxel_size"] == 0.1,
        "block": config["data"]["block_size_m"] == expected_block,
        "train_overlap": config["data"]["train_overlap"] == 0.0,
        "stride": config["data"]["inference_stride_m"] == expected_stride,
        "inference_overlap": config["data"]["inference_overlap"] == 0.5,
        "ignore": config["data"]["ignore_index"] == -100,
        "epochs": config["training"]["epochs"] == 100,
        "physical_batch": config["training"]["physical_batch_size"] == 1,
        "accumulation": config["training"]["gradient_accumulation_steps"] == 2,
        "effective_batch": config["training"]["effective_batch_size"] == 2,
        "amp": config["training"]["amp"] is True,
        "seed": config["training"]["seed"] == 42,
        "sampler": config["sampling"]["mode"] == "uniform_blocks",
        "class_aware": config["sampling"]["class_aware"] is False,
        "loss": config["loss"]["mode"] == "ce_balanced",
        "optimizer": config["optimizer"]["name"] == "AdamW",
        "lr": config["optimizer"]["learning_rate"] == 1e-3,
        "weight_decay": config["optimizer"]["weight_decay"] == 1e-4,
        "scheduler": config["scheduler"]["name"] == "CosineAnnealingLR",
        "scheduler_step": config["scheduler"]["step_mode"] == "epoch",
        "selection": config["selection"] == {"metric": "val_mIoU", "mode": "max"},
        "small_block_policy": (
            not is_merged
            or config["small_block_policy"]
            == {
                "mode": "deterministic_adjacent_merge",
                "minimum_deepest_active_sites": 8,
                "point_dropping": False,
                "voxel_duplication": False,
            }
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"frozen Block75 config mismatch: {failed}")


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
    val_metrics: dict[str, Any],
    hashes: dict[str, str],
    block_size_m: float,
    inference_stride_m: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_epoch": epoch,
            "best_val_miou": best_miou,
            "best_epoch": best_epoch,
            "rng_state": capture_rng(torch),
            "train_generator_state": train_generator.get_state(),
            "physical_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "block_size_m": block_size_m,
            "inference_stride_m": inference_stride_m,
            **hashes,
            "val_metrics": val_metrics,
        },
        temporary,
    )
    os.replace(temporary, path)


def metrics_from_confusion(confusion: Any) -> dict[str, Any]:
    import torch

    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    gt = confusion.sum(1)
    predicted = confusion.sum(0)
    union = gt + predicted - tp
    if torch.any(gt == 0):
        raise RuntimeError("one or more evaluation classes have empty ground truth")
    iou = tp / union
    accuracy = tp / gt
    return {
        "oa": float(tp.sum() / confusion.sum()),
        "miou": float(iou.mean()),
        "macc": float(accuracy.mean()),
        "per_class_iou": [float(value) for value in iou],
        "confusion_matrix": confusion.to(torch.int64).tolist(),
        "valid_points": int(confusion.sum()),
    }


def evaluate_full_tiles(
    model: Any,
    tile_paths: list[Path],
    device: Any,
    class_weights: Any,
    block_size_voxels: int,
    stride_voxels: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from engine.sliding_window_inference import sliding_window_inference

    confusion = torch.zeros((9, 9), dtype=torch.int64)
    loss_sum = 0.0
    valid_total = 0
    minimum_vote = None
    overlap_total = 0
    for tile_index, path in enumerate(tile_paths, start=1):
        with np.load(path, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates"], dtype=np.int32)
            features = np.asarray(archive["features"], dtype=np.float32)
            labels = np.asarray(archive["labels"], dtype=np.int64)
        output = sliding_window_inference(
            model, coordinates, features, device,
            block_size_voxels=block_size_voxels, stride_voxels=stride_voxels,
            num_classes=9, use_amp=True,
        )
        logits = output["averaged_logits"]
        label_tensor = torch.from_numpy(labels)
        valid = label_tensor != -100
        valid_labels = label_tensor[valid]
        predictions = logits.argmax(1)[valid]
        encoded = valid_labels * 9 + predictions
        confusion += torch.bincount(encoded, minlength=81).reshape(9, 9)
        tile_loss = F.cross_entropy(
            logits, label_tensor, weight=class_weights,
            ignore_index=-100, reduction="sum",
        )
        if not torch.isfinite(tile_loss):
            raise RuntimeError(f"non-finite sliding loss: {path.name}")
        loss_sum += float(tile_loss)
        valid_total += int(valid.sum())
        minimum_vote = (
            output["minimum_vote_count"] if minimum_vote is None
            else min(minimum_vote, output["minimum_vote_count"])
        )
        overlap_total += output["overlap_voxels"]
        print(
            f"eval_tile={tile_index}/{len(tile_paths)} tile={path.stem} "
            f"voxels={len(labels)} min_vote={output['minimum_vote_count']}",
            flush=True,
        )
        del output, logits, label_tensor, valid, valid_labels, predictions, encoded
    if int(confusion.sum()) != valid_total or minimum_vote is None or minimum_vote < 1:
        raise RuntimeError("full-tile sliding coverage/confusion invariant failed")
    metrics = metrics_from_confusion(confusion)
    metrics.update(
        {
            "loss": loss_sum / valid_total,
            "minimum_vote_count": minimum_vote,
            "overlap_voxels": overlap_total,
            "tiles": len(tile_paths),
        }
    )
    return metrics


def detailed_classes(metrics: dict[str, Any], class_names: list[str]) -> list[dict[str, Any]]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    gt, predicted, tp = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    rows = []
    for class_id in range(9):
        fp, fn = predicted[class_id] - tp[class_id], gt[class_id] - tp[class_id]
        precision = tp[class_id] / predicted[class_id] if predicted[class_id] else 0.0
        recall = tp[class_id] / gt[class_id] if gt[class_id] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "gt_points": int(gt[class_id]),
                "predicted_points": int(predicted[class_id]),
                "true_positive": int(tp[class_id]),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "iou": metrics["per_class_iou"][class_id],
                "never_predicted": int(predicted[class_id]) == 0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: Path, matrix: list[list[int]]) -> None:
    rows = [
        {"true\\pred": index, **{str(i): value for i, value in enumerate(row)}}
        for index, row in enumerate(matrix)
    ]
    write_csv(path, rows)


def run(config_path: Path, resume_path: Path | None = None) -> int:
    import MinkowskiEngine as ME
    import torch
    from torch.utils.data import DataLoader

    from datasets_loader.eclair_block_dataset import EclairBlockDataset, build_block_sparse_collate_fn
    from engine.trainer import set_random_seed
    from models.backbone.minkowski.MinkUNet34C import MinkUNet34C

    config_path = config_path.resolve()
    config = load_simple_yaml(config_path)
    validate_config(config)
    experiment_name = config["experiment_name"]
    is_merged = experiment_name in {
        "Baseline-Strong-Block50-MergedSmall",
        "Baseline-Strong-Block50-MergedSmall-AmpResumeGate",
        "Baseline-Strong-Block50-MergedSmall-AmpSafe",
    }
    block_size_m = float(config["data"]["block_size_m"])
    inference_stride_m = float(config["data"]["inference_stride_m"])
    block_size_voxels = int(round(block_size_m / 0.1))
    stride_voxels = int(round(inference_stride_m / 0.1))
    index_root = Path(config["data"]["block_index_root"])
    expected_train_blocks = (
        1410 if is_merged else (1425 if block_size_m == 50.0 else 1273)
    )
    output = Path(config["output_dir"]).resolve()
    if resume_path is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite formal Block75 run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    console = (output / "train_console.log").open(
        "a" if resume_path else "x", encoding="utf-8"
    )
    sys.stdout, sys.stderr = Tee(sys.__stdout__, console), Tee(sys.__stderr__, console)
    started_at = utc_now()
    mapping_path = Path("configs/dataset/class_mapping.json")
    weight_path = Path(config["loss"]["class_weight_file"])
    model_config_path = Path(config["model"]["config"])
    mapping = json.loads(mapping_path.read_text())
    weight_report = json.loads(weight_path.read_text())
    model_config = load_simple_yaml(model_config_path)
    if weight_report["class_ids"] != list(range(9)):
        raise ValueError("class weights are not ordered train IDs 0..8")
    class_names = [mapping["train_to_name"][str(i)] for i in range(9)]
    hashes = {
        "mapping_hash": canonical_hash(mapping),
        "class_weight_hash": sha256(weight_path),
        "model_config_hash": sha256(model_config_path),
        "train_config_hash": sha256(config_path),
        "block_index_hash": index_tree_hash(index_root),
    }
    if is_merged:
        hashes["merge_manifest_hash"] = sha256(
            Path("results/block50_smallblock/merge_manifest.csv")
        )
    seed = 42
    set_random_seed(seed)
    train_generator = torch.Generator().manual_seed(seed)
    device = torch.device("cuda:0")
    train_dataset = EclairBlockDataset(index_root=index_root, split="train")
    if len(train_dataset) != expected_train_blocks:
        raise ValueError(
            f"expected {expected_train_blocks} trainable blocks, found {len(train_dataset)}"
        )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, generator=train_generator,
        num_workers=0, pin_memory=True, collate_fn=build_block_sparse_collate_fn(),
    )
    val_paths = sorted(Path(config["data"]["val_dir"]).glob("*.npz"))
    if len(val_paths) != 59:
        raise ValueError("formal val tile count changed")
    model = MinkUNet34C(
        in_channels=model_config["in_channels"],
        num_classes=model_config["num_classes"],
        dimension=model_config["dimension"],
        stem_channels=model_config["stem_channels"],
        planes=model_config["planes"],
        layers=model_config["layers"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 39154665:
        raise ValueError("MinkUNet34C parameter count changed")
    class_weights_cpu = torch.tensor(weight_report["final_weights"], dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights_cpu.to(device), ignore_index=-100
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_epoch, best_epoch, best_miou = 1, 0, float("-inf")
    checkpoint_best, checkpoint_last = output / "checkpoint_best.pth", output / "checkpoint_last.pth"
    if resume_path:
        # Keep serialized CPU RNG and DataLoader-generator ByteTensors on CPU.
        # load_state_dict migrates model/optimizer tensors to their parameter device.
        state = torch.load(resume_path.resolve(), map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        restore_rng(torch, state["rng_state"])
        train_generator.set_state(state["train_generator_state"])
        if state["physical_batch_size"] != 1 or state["gradient_accumulation_steps"] != 2:
            raise ValueError("resume batch strategy mismatch")
        start_epoch = int(state["completed_epoch"]) + 1
        best_epoch, best_miou = int(state["best_epoch"]), float(state["best_val_miou"])
    manifest = {
        "experiment_name": experiment_name,
        "model": "MinkUNet34C",
        "parameter_count": parameter_count,
        "full_tile_training": False,
        "spatial_block_training": True,
        "block_size_m": block_size_m,
        "train_overlap": 0.0,
        "full_tile_sliding_inference": True,
        "inference_stride_m": inference_stride_m,
        "inference_overlap": 0.5,
        "logit_aggregation": "arithmetic_mean",
        "physical_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 2,
        "voxel_size": 0.1,
        "input_features": ["intensity", "red", "green", "blue", "Reflectance", "Deviation"],
        "loss": "balanced_cross_entropy",
        "sampler": "uniform_blocks",
        "architecture_changed": False,
        "voxel_size_changed": False,
        "features_changed": False,
        "label_mapping_changed": False,
        "point_dropping": False,
        "pseudo_labels": False,
        "change_from_block75": (
            "resource_window_only" if block_size_m == 50.0 else None
        ),
        "class_weights_changed": False,
        "optimizer_changed": False,
        "learning_rate_changed": False,
        "sampler_changed": False,
        "small_block_policy": (
            "deterministic_adjacent_merge" if is_merged else None
        ),
        "minimum_deepest_active_sites": 8 if is_merged else None,
        "batchnorm_changed": False,
        "voxel_duplication": False,
        "affected_small_blocks": 15 if is_merged else 0,
        "affected_tiles": 15 if is_merged else 0,
        "train_samples_before_merge": 1425 if is_merged else None,
        "train_samples_after_merge": 1410 if is_merged else None,
        "environment": {
            "python": platform.python_version(), "pytorch": torch.__version__,
            "cuda": torch.version.cuda, "minkowski_engine": ME.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
        "seed": seed, "config": config, **hashes, "git": git_state(),
        "started_at_utc": started_at, "ended_at_utc": None,
        "training_time_seconds": None, "best_epoch": None,
        "best_val_miou": None, "checkpoint_sha256": None,
        "test_used_for_selection": False, "class_aware_sampler_enabled": False,
        "resume_source": (
            str(resume_path.resolve()) if resume_path is not None else None
        ),
        "resume_completed_epoch": start_epoch - 1 if resume_path is not None else None,
        "resume_origin": config.get("resume_origin"),
        "resume_origin_completed_epoch": config.get("resume_origin_completed_epoch"),
        "amp_overflow_policy": "standard_gradscaler_skip",
        "gradient_clipping": False,
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    columns = [
        "epoch", "learning_rate", "train_loss", "val_loss", "val_OA", "val_mIoU",
        "val_mAcc", *(f"val_iou_class_{i}" for i in range(9)),
        "epoch_train_time_seconds", "epoch_val_time_seconds",
        "epoch_total_time_seconds", "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes", "amp_skipped_steps", "amp_skip_rate",
        "minimum_grad_scale", "ending_grad_scale", "consecutive_amp_skips",
        "amp_skip_samples",
    ]
    log_path = output / "training_log.csv"
    if not log_path.exists():
        with log_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for history_key in (
                "history_training_log_original", "history_training_log_gate"
            ):
                history_value = config.get(history_key)
                if not history_value:
                    continue
                with Path(history_value).open(
                    encoding="utf-8", newline=""
                ) as history_handle:
                    for history_row in csv.DictReader(history_handle):
                        writer.writerow(
                            {column: history_row.get(column, "") for column in columns}
                        )
    resume_best = config.get("resume_best_checkpoint")
    if resume_best and not checkpoint_best.exists():
        shutil.copy2(Path(resume_best), checkpoint_best)
    total_started = time.monotonic()
    global_peak_allocated = global_peak_reserved = 0
    current_context: dict[str, Any] = {}
    consecutive_amp_skips = 0
    try:
        end_epoch = int(config["training"].get("diagnostic_end_epoch", 100))
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_started = train_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_weighted_sum, valid_total = 0.0, 0
            amp_skipped_steps = 0
            optimizer_step_attempts = 0
            minimum_grad_scale = float(scaler.get_scale())
            amp_skip_samples: list[dict[str, Any]] = []
            for micro_index, batch in enumerate(train_loader, start=1):
                current_context = {
                    "epoch": epoch, "micro_batch_index": micro_index,
                    "tile_name": batch["tile_name"][0],
                    "block_id": batch["block_id"][0],
                    "merged_sample_id": batch["merged_sample_id"][0],
                    "source_block_ids": batch["source_block_ids"][0],
                    "voxel_count": int(batch["features"].shape[0]),
                    "accumulation_position": 1 if micro_index % 2 else 2,
                    "level4_active_sites": batch["level4_active_sites"][0],
                }
                import MinkowskiEngine as ME_local
                sparse_input = ME_local.SparseTensor(
                    features=batch["features"].to(device),
                    coordinates=batch["coordinates"], device=device,
                )
                labels = batch["labels"].to(device)
                valid_points = int((labels != -100).sum())
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = model(sparse_input)
                    raw_loss = criterion(logits.F, labels)
                if not torch.isfinite(raw_loss):
                    raise RuntimeError("non-finite training loss")
                scaler.scale(raw_loss / 2).backward()
                should_step = micro_index % 2 == 0 or micro_index == len(train_loader)
                if should_step:
                    optimizer_step_attempts += 1
                    scaler.unscale_(optimizer)
                    gradients_finite = all(
                        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    )
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    minimum_grad_scale = min(minimum_grad_scale, scale_after)
                    amp_step_skipped = scale_after < scale_before
                    if amp_step_skipped:
                        amp_skipped_steps += 1
                        consecutive_amp_skips += 1
                        amp_skip_samples.append(
                            {
                                "micro_batch_index": micro_index,
                                "tile_name": current_context["tile_name"],
                                "sample_id": current_context["merged_sample_id"],
                                "scale_before": scale_before,
                                "scale_after": scale_after,
                            }
                        )
                        print(
                            f"amp_skip epoch={epoch} micro_batch={micro_index} "
                            f"tile={current_context['tile_name']} "
                            f"sample={current_context['merged_sample_id']} "
                            f"gradients_finite={gradients_finite} "
                            f"scale_before={scale_before} scale_after={scale_after}",
                            flush=True,
                        )
                    else:
                        consecutive_amp_skips = 0
                    optimizer.zero_grad(set_to_none=True)
                    if consecutive_amp_skips >= 8:
                        raise RuntimeError("consecutive AMP skips reached 8")
                    if scale_after <= 1.0 and amp_step_skipped and consecutive_amp_skips >= 2:
                        raise RuntimeError("GradScaler remained overflowing at scale 1.0")
                loss_weighted_sum += float(raw_loss.detach()) * valid_points
                valid_total += valid_points
                print(
                    f"epoch={epoch} micro_batch={micro_index}/{len(train_loader)} "
                    f"tile={current_context['tile_name']} block={current_context['block_id']} "
                    f"voxels={current_context['voxel_count']} loss={float(raw_loss.detach()):.6f} "
                    f"optimizer_step={should_step}",
                    flush=True,
                )
                del sparse_input, logits, raw_loss, labels, batch
            amp_skip_rate = amp_skipped_steps / optimizer_step_attempts
            if amp_skip_rate > 0.01:
                raise RuntimeError(
                    f"epoch AMP skip rate exceeded 1%: {amp_skip_rate}"
                )
            train_seconds = time.monotonic() - train_started
            val_started = time.monotonic()
            val_metrics = evaluate_full_tiles(
                model, val_paths, device, class_weights_cpu,
                block_size_voxels, stride_voxels,
            )
            val_seconds = time.monotonic() - val_started
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            global_peak_allocated = max(global_peak_allocated, peak_allocated)
            global_peak_reserved = max(global_peak_reserved, peak_reserved)
            row = {
                "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": loss_weighted_sum / valid_total,
                "val_loss": val_metrics["loss"], "val_OA": val_metrics["oa"],
                "val_mIoU": val_metrics["miou"], "val_mAcc": val_metrics["macc"],
                **{f"val_iou_class_{i}": val_metrics["per_class_iou"][i] for i in range(9)},
                "epoch_train_time_seconds": train_seconds,
                "epoch_val_time_seconds": val_seconds,
                "epoch_total_time_seconds": time.monotonic() - epoch_started,
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
                "amp_skipped_steps": amp_skipped_steps,
                "amp_skip_rate": amp_skip_rate,
                "minimum_grad_scale": minimum_grad_scale,
                "ending_grad_scale": float(scaler.get_scale()),
                "consecutive_amp_skips": consecutive_amp_skips,
                "amp_skip_samples": json.dumps(
                    amp_skip_samples, ensure_ascii=False, separators=(",", ":")
                ),
            }
            with log_path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=columns).writerow(row)
            if val_metrics["miou"] > best_miou:
                best_miou, best_epoch = float(val_metrics["miou"]), epoch
                save_checkpoint(
                    checkpoint_best, torch, model, optimizer, scheduler, scaler,
                    train_generator, epoch=epoch, best_epoch=best_epoch,
                    best_miou=best_miou, val_metrics=val_metrics, hashes=hashes,
                    block_size_m=block_size_m,
                    inference_stride_m=inference_stride_m,
                )
            scheduler.step()
            save_checkpoint(
                checkpoint_last, torch, model, optimizer, scheduler, scaler,
                train_generator, epoch=epoch, best_epoch=best_epoch,
                best_miou=best_miou, val_metrics=val_metrics, hashes=hashes,
                block_size_m=block_size_m,
                inference_stride_m=inference_stride_m,
            )
            print(
                f"epoch_complete={epoch} train_loss={row['train_loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_miou={val_metrics['miou']:.6f}",
                flush=True,
            )
        if bool(config["training"].get("diagnostic_resume_gate", False)):
            unhealthy = [
                name for name, tensor in model.state_dict().items()
                if not bool(torch.isfinite(tensor).all())
            ]
            if unhealthy:
                raise RuntimeError(
                    f"non-finite model or BatchNorm state after AMP gate: {unhealthy[:5]}"
                )
            manifest.update(
                {
                    "ended_at_utc": utc_now(),
                    "status": "amp_resume_gate_complete",
                    "completed_epoch": end_epoch,
                    "test_evaluations": 0,
                    "amp_overflow_policy": "standard_gradscaler_skip",
                    "gradient_clipping": False,
                }
            )
            atomic_write_json(output / "run_manifest.json", manifest)
            return 0
    except BaseException as error:
        failure = {
            "status": "failed", "error_type": type(error).__name__,
            "error": str(error), "context": current_context,
            "timestamp_utc": utc_now(),
            "checkpoint_last_exists": checkpoint_last.exists(),
        }
        atomic_write_json(output / "failure_report.json", failure)
        raise
    training_seconds = time.monotonic() - total_started
    best = torch.load(checkpoint_best, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    replay_val = evaluate_full_tiles(
        model, val_paths, device, class_weights_cpu,
        block_size_voxels, stride_voxels,
    )
    with log_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    best_row = next(row for row in rows if int(row["epoch"]) == best_epoch)
    replay_ok = all(
        abs(replay_val[key] - float(best_row[column])) <= 1e-12
        for key, column in (
            ("loss", "val_loss"), ("oa", "val_OA"),
            ("miou", "val_mIoU"), ("macc", "val_mAcc"),
        )
    )
    if not replay_ok:
        atomic_write_json(
            output / "failure_report.json",
            {"status": "failed_val_replay", "best_epoch": best_epoch,
             "recomputed": replay_val, "logged": best_row},
        )
        raise RuntimeError("best checkpoint val replay differs from training log")
    val_classes = detailed_classes(replay_val, class_names)
    atomic_write_json(output / "metrics_val_best.json", {**replay_val, "classes": val_classes})
    write_confusion(output / "val_confusion_matrix.csv", replay_val["confusion_matrix"])
    write_csv(output / "val_per_class_metrics.csv", val_classes)
    write_csv(output / "val_prediction_diagnostics.csv", val_classes)
    test_paths = sorted(Path(config["data"]["test_dir"]).glob("*.npz"))
    if len(test_paths) != 106:
        raise ValueError("formal test tile count changed")
    test_metrics = evaluate_full_tiles(
        model, test_paths, device, class_weights_cpu,
        block_size_voxels, stride_voxels,
    )
    test_classes = detailed_classes(test_metrics, class_names)
    atomic_write_json(output / "metrics_test_best.json", {**test_metrics, "classes": test_classes})
    write_confusion(output / "test_confusion_matrix.csv", test_metrics["confusion_matrix"])
    write_csv(output / "test_per_class_metrics.csv", test_classes)
    write_csv(output / "test_prediction_diagnostics.csv", test_classes)
    simple = json.loads(Path("results/baseline_minkowski/metrics.json").read_text())
    comparison = [
        {"metric": "parameter_count", "Baseline-Simple": 204585, experiment_name: parameter_count},
        {"metric": "training_organization", "Baseline-Simple": "whole_tile", experiment_name: f"{block_size_m:g}m_blocks_full_tile_sliding_eval"},
        {"metric": "loss", "Baseline-Simple": "unweighted_ce", experiment_name: "balanced_ce"},
    ]
    for metric, simple_key, strong_key in (
        ("OA", "oa", "oa"), ("mIoU", "miou", "miou"), ("mAcc", "mean_class_accuracy", "macc")
    ):
        for split, strong in (("val", replay_val), ("test", test_metrics)):
            comparison.append({"metric": f"{split}_{metric}", "Baseline-Simple": simple[split][simple_key], experiment_name: strong[strong_key]})
    for split, simple_split, strong in (("val", simple["val"], replay_val), ("test", simple["test"], test_metrics)):
        for class_id in range(9):
            comparison.append({"metric": f"{split}_iou_class_{class_id}", "Baseline-Simple": simple_split["per_class_iou"][class_id], experiment_name: strong["per_class_iou"][class_id]})
        strong_classes = val_classes if split == "val" else test_classes
        comparison.append({"metric": f"{split}_never_predicted_count", "Baseline-Simple": sum(value == 0 for value in simple_split["per_class_iou"]), experiment_name: sum(row["never_predicted"] for row in strong_classes)})
    comparison.append({"metric": "interpretation", "Baseline-Simple": "naive lower baseline", experiment_name: "changes capacity, residual backbone, balanced CE, and spatial-block organization; not a single-factor ablation"})
    write_csv(output / "comparison_with_simple.csv", comparison)
    manifest.update(
        {
            "ended_at_utc": utc_now(), "training_time_seconds": training_seconds,
            "best_epoch": best_epoch, "best_val_miou": best_miou,
            "checkpoint_sha256": sha256(checkpoint_best),
            "peak_cuda_allocated_bytes": global_peak_allocated,
            "peak_cuda_reserved_bytes": global_peak_reserved,
            "val_replay_matches_log": True, "test_evaluations": 1,
        }
    )
    atomic_write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"best_epoch": best_epoch, "val": replay_val, "test": test_metrics}, indent=2))
    return 0
