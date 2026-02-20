from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np


def _to_batched_vertices(vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(vertices, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected shape (B, V, 3) or (V, 3), got {arr.shape}")
    return arr


def _to_batched_vertex_values(values: np.ndarray, n_batch: int, n_vertices: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, ...]
    if arr.shape != (n_batch, n_vertices):
        raise ValueError(
            f"Expected shape ({n_batch}, {n_vertices}) or ({n_vertices},), got {arr.shape}"
        )
    return arr


def vertex_error_norm(pred_vertices: np.ndarray, gt_vertices: np.ndarray) -> np.ndarray:
    pred = _to_batched_vertices(pred_vertices)
    gt = _to_batched_vertices(gt_vertices)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    return np.linalg.norm(pred - gt, axis=-1)


def cne(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    local_scale: np.ndarray,
    tau: float = 1e-6,
    omega: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    err = vertex_error_norm(pred_vertices, gt_vertices)
    scale = _to_batched_vertex_values(local_scale, err.shape[0], err.shape[1])
    ratio = err / (scale + tau)
    return _reduce_metric(ratio, omega=omega, mask=mask)


def cne_log(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    local_scale: np.ndarray,
    eps: float = 1e-12,
    omega: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    err = vertex_error_norm(pred_vertices, gt_vertices)
    scale = _to_batched_vertex_values(local_scale, err.shape[0], err.shape[1])
    value = np.abs(np.log((err + eps) / np.maximum(scale, eps)))
    return _reduce_metric(value, omega=omega, mask=mask)


def _reduce_metric(
    value: np.ndarray,
    omega: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (value.shape[1],):
            raise ValueError(f"Mask must have shape ({value.shape[1]},), got {mask_arr.shape}")
        value = value[:, mask_arr]
        if omega is not None:
            omega = np.asarray(omega, dtype=np.float64)
            omega = omega[:, mask_arr] if omega.ndim == 2 else omega[mask_arr]

    if omega is None:
        return float(np.mean(value))

    omega_arr = np.asarray(omega, dtype=np.float64)
    if omega_arr.ndim == 1:
        omega_arr = omega_arr[None, :]
    if omega_arr.shape != value.shape:
        raise ValueError(f"omega/value shape mismatch: {omega_arr.shape} vs {value.shape}")

    weighted_sum = np.sum(value * omega_arr)
    normalizer = np.sum(omega_arr)
    if normalizer <= 0:
        raise ValueError("omega must contain positive weights")
    return float(weighted_sum / normalizer)


def summarize_cne_metrics(
    pred_vertices: np.ndarray,
    gt_vertices: np.ndarray,
    local_scale: np.ndarray,
    tau: float = 1e-6,
    eps: float = 1e-12,
    omega: Optional[np.ndarray] = None,
    region_masks: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Compute global and region-specific CNE summaries.

    region_masks is a mapping like {"hands": hand_mask, "face": face_mask, "body": body_mask}
    where each mask has shape (V,) and dtype bool.
    """

    metrics = {
        "cne": cne(pred_vertices, gt_vertices, local_scale, tau=tau),
        "cne_log": cne_log(pred_vertices, gt_vertices, local_scale, eps=eps),
    }

    if omega is not None:
        metrics["cne_perceptual"] = cne(
            pred_vertices, gt_vertices, local_scale, tau=tau, omega=omega
        )
        metrics["cne_log_perceptual"] = cne_log(
            pred_vertices, gt_vertices, local_scale, eps=eps, omega=omega
        )

    if region_masks:
        for region_name, region_mask in region_masks.items():
            metrics[f"{region_name}/cne"] = cne(
                pred_vertices,
                gt_vertices,
                local_scale,
                tau=tau,
                mask=region_mask,
            )
            metrics[f"{region_name}/cne_log"] = cne_log(
                pred_vertices,
                gt_vertices,
                local_scale,
                eps=eps,
                mask=region_mask,
            )
            if omega is not None:
                metrics[f"{region_name}/cne_perceptual"] = cne(
                    pred_vertices,
                    gt_vertices,
                    local_scale,
                    tau=tau,
                    omega=omega,
                    mask=region_mask,
                )
                metrics[f"{region_name}/cne_log_perceptual"] = cne_log(
                    pred_vertices,
                    gt_vertices,
                    local_scale,
                    eps=eps,
                    omega=omega,
                    mask=region_mask,
                )

    return metrics
