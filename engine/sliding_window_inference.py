"""Full-tile XY sliding-window inference with voxel-index logit aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _window_starts(minimum: int, maximum: int, size: int, stride: int) -> list[int]:
    if size <= 0 or stride <= 0 or stride > size:
        raise ValueError("window size/stride must satisfy 0 < stride <= size")
    starts = list(range(minimum, maximum + 1, stride))
    if not starts:
        starts = [minimum]
    return starts


def sliding_window_inference(
    model: Any,
    coordinates: np.ndarray,
    features: np.ndarray,
    device: Any,
    *,
    block_size_voxels: int,
    stride_voxels: int,
    num_classes: int = 9,
    use_amp: bool = False,
    geometry: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return averaged logits in the exact original voxel order."""
    import MinkowskiEngine as ME
    import torch

    coordinates = np.asarray(coordinates, dtype=np.int32)
    features = np.asarray(features, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must be int32 [N,3]")
    if features.shape != (len(coordinates), 6):
        raise ValueError("baseline features must be float32 [N,6]")
    if geometry is not None and geometry.shape != (len(coordinates), 12):
        raise ValueError("geometry must be float32 [N,12]")
    logits_sum = torch.zeros((len(coordinates), num_classes), dtype=torch.float32)
    vote_count = torch.zeros(len(coordinates), dtype=torch.int32)
    x_starts = _window_starts(
        int(coordinates[:, 0].min()),
        int(coordinates[:, 0].max()),
        block_size_voxels,
        stride_voxels,
    )
    y_starts = _window_starts(
        int(coordinates[:, 1].min()),
        int(coordinates[:, 1].max()),
        block_size_voxels,
        stride_voxels,
    )
    model.eval()
    windows = 0
    overlap_voxels = 0
    with torch.no_grad():
        for x0 in x_starts:
            for y0 in y_starts:
                indices = np.flatnonzero(
                    (coordinates[:, 0] >= x0)
                    & (coordinates[:, 0] < x0 + block_size_voxels)
                    & (coordinates[:, 1] >= y0)
                    & (coordinates[:, 1] < y0 + block_size_voxels)
                )
                if not len(indices):
                    continue
                local = coordinates[indices].copy()
                local -= np.asarray(
                    [x0, y0, int(local[:, 2].min())], dtype=np.int32
                )
                batched = np.column_stack(
                    (np.zeros(len(local), dtype=np.int32), local)
                )
                sparse_input = ME.SparseTensor(
                    features=torch.from_numpy(features[indices]).to(device),
                    coordinates=torch.from_numpy(batched),
                    device=device,
                )
                geometry_input = None if geometry is None else torch.from_numpy(geometry[indices]).to(device)
                # Evaluation is deliberately FP32. Training AMP policy is
                # independent and remains unchanged.
                with torch.autocast(
                    device_type="cuda", enabled=False
                ):
                    if geometry_input is None:
                        output = model(sparse_input)
                    elif model.__class__.__name__ == "GeometryConcatMinkUNet":
                        fused = ME.SparseTensor(features=torch.cat((sparse_input.F, geometry_input), 1), coordinate_map_key=sparse_input.coordinate_map_key, coordinate_manager=sparse_input.coordinate_manager)
                        output = model(fused)
                    else:
                        output = model(sparse_input, geometry_input)
                if output.F.shape != (len(indices), num_classes):
                    raise RuntimeError("sliding-window logits have wrong shape")
                if not torch.equal(output.C.cpu(), torch.from_numpy(batched)):
                    raise RuntimeError("Minkowski output order differs from input order")
                index_tensor = torch.from_numpy(indices)
                logits_sum[index_tensor] += output.F.detach().float().cpu()
                vote_count[index_tensor] += 1
                windows += 1
                del sparse_input, geometry_input, output
    if torch.any(vote_count == 0):
        missing = int((vote_count == 0).sum())
        raise RuntimeError(f"sliding windows missed {missing} voxels")
    averaged_logits = logits_sum / vote_count.to(torch.float32).unsqueeze(1)
    overlap_voxels = int((vote_count > 1).sum())
    return {
        "averaged_logits": averaged_logits,
        "validation_dtype": "float32",
        "logits_dtype": str(averaged_logits.dtype),
        "logits_finite": bool(torch.isfinite(averaged_logits).all()),
        "vote_count": vote_count,
        "window_count": windows,
        "overlap_voxels": overlap_voxels,
        "minimum_vote_count": int(vote_count.min()),
        "maximum_vote_count": int(vote_count.max()),
    }
