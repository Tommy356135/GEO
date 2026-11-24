"""Semantic segmentation evaluation for the Baseline MinkowskiUNet."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from engine.trainer import _make_sparse_input


def evaluate(
    model: Any,
    data_loader: Iterable[Mapping[str, Any]],
    device: Any,
    *,
    num_classes: int = 9,
    ignore_index: int = -100,
    criterion: Any | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Return OA, mIoU, mean class accuracy, per-class IoU, and confusion."""
    import torch

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    loss_sum = 0.0
    loss_points = 0
    batch_count = 0
    with torch.no_grad():
        for batch in data_loader:
            sparse_input, labels = _make_sparse_input(batch, device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                logits = model(sparse_input)
                if logits.F.shape[0] != labels.shape[0] or logits.F.shape[1] != num_classes:
                    raise ValueError(
                        f"invalid logits/labels: {tuple(logits.F.shape)} vs {tuple(labels.shape)}"
                    )
            valid = labels != ignore_index
            valid_labels = labels[valid]
            if valid_labels.numel() == 0:
                batch_count += 1
                del sparse_input, logits, labels, valid, valid_labels, batch
                continue
            if torch.any((valid_labels < 0) | (valid_labels >= num_classes)):
                raise ValueError("evaluation labels are outside [0, num_classes) or ignore_index")
            predictions = logits.F.argmax(dim=1)[valid]
            encoded = valid_labels * num_classes + predictions
            confusion += torch.bincount(
                encoded.detach().cpu(), minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
            if criterion is not None:
                loss = criterion(logits.F, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite evaluation loss")
                loss_sum += float(loss) * int(valid_labels.numel())
                loss_points += int(valid_labels.numel())
                del loss
            batch_count += 1
            del sparse_input, logits, labels, valid, valid_labels, predictions, encoded, batch
    total = int(confusion.sum())
    if batch_count == 0 or total == 0:
        raise ValueError("evaluation data loader produced no valid labeled points")
    true_positive = confusion.diag().to(torch.float64)
    ground_truth = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    union = ground_truth + predicted - true_positive
    present_iou = union > 0
    present_accuracy = ground_truth > 0
    iou = true_positive[present_iou] / union[present_iou]
    class_accuracy = true_positive[present_accuracy] / ground_truth[present_accuracy]
    per_class_iou: list[float | None] = []
    for class_id in range(num_classes):
        per_class_iou.append(
            float(true_positive[class_id] / union[class_id])
            if union[class_id] > 0 else None
        )
    metrics: dict[str, Any] = {
        "oa": float(true_positive.sum() / total),
        "miou": float(iou.mean()),
        "mean_class_accuracy": float(class_accuracy.mean()),
        "per_class_iou": per_class_iou,
        "confusion_matrix": confusion.tolist(),
        "valid_points": total,
        "batches": batch_count,
    }
    if criterion is not None:
        metrics["loss"] = loss_sum / loss_points
    return metrics
