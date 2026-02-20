#!/usr/bin/env python3
"""Run inference/eval and export per-sample conditioning metric tables.

This script bridges raw SAM-3D-Body annotation shards and tools/analyze_conditioning.py.
It runs model inference per person annotation and writes a CSV with the required columns:
- kappa_geom, kappa_spec
- hand_error, face_error
- mpjpe, pve, cne

Expected workflow:
1) Download annotation shards via data/scripts/download.py
2) Run this script to produce *_metrics.csv
3) Run tools/analyze_conditioning.py on the exported CSV(s)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.conditioning import compute_conditioning
from sam_3d_body.metrics import cne


# MHR70 keypoint ranges in sam_3d_body/metadata/mhr70.py
FACE_IDXS = np.array([0, 1, 2, 3, 4], dtype=np.int64)  # nose/eyes/ears
RHAND_IDXS = np.arange(21, 43, dtype=np.int64)  # right hand + wrist
LHAND_IDXS = np.arange(43, 65, dtype=np.int64)  # left hand + wrist
HAND_IDXS = np.concatenate([RHAND_IDXS, LHAND_IDXS])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", required=True, type=str)
    parser.add_argument("--image-dir", required=True, type=str)
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument("--mhr-path", required=True, type=str)
    parser.add_argument("--output-file", required=True, type=str)
    parser.add_argument(
        "--max-samples",
        default=0,
        type=int,
        help="Optional cap on number of person annotations to process (0 = all).",
    )
    parser.add_argument(
        "--glob-pattern",
        default="*.parquet",
        type=str,
        help="Glob pattern for annotation tables inside --annotation-dir.",
    )
    return parser.parse_args()


def _get_img_name(row: Dict) -> str:
    dataset = row["dataset"]
    img_name = row["image"]
    if dataset == "coco":
        _, split, _ = img_name.split("_")
        return str(Path(split) / img_name)
    if dataset == "mpii":
        return str(Path("images") / img_name)
    if dataset == "aic":
        return str(Path("train") / "images" / img_name)
    return img_name


def _iter_rows(annotation_dir: Path, glob_pattern: str) -> Iterable[Dict]:
    files = sorted(annotation_dir.glob(glob_pattern))
    if not files:
        raise ValueError(f"No annotation files matched {glob_pattern!r} under {annotation_dir}")

    for ann_file in files:
        frame = pd.read_parquet(ann_file)
        for _, row in frame.iterrows():
            rec = row.to_dict()
            rec["_source_file"] = str(ann_file)
            yield rec


def _to_numpy_keypoints3d(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        try:
            arr = np.stack(list(value))
        except TypeError as exc:
            raise ValueError("keypoints_3d has object dtype but is not iterable") from exc

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected keypoints_3d shape (K,3), got {arr.shape}")
    return arr.astype(np.float64)


def _get_optional_keypoints3d(row: Dict) -> Optional[np.ndarray]:
    value = row.get("keypoints_3d")
    if _is_missing(value):
        return None
    return _to_numpy_keypoints3d(value)


def _is_missing(value) -> bool:
    if value is None:
        return True
    # Pandas nullable scalars / NaN-like values
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "nan", ""}:
        return True
    return False


def _to_camera_intrinsics(value, device: torch.device) -> Optional[torch.Tensor]:
    """Normalize cam_int from parquet rows to a 3x3 tensor.

    Returns None when intrinsics are missing so the estimator can use its fallback behavior.
    """
    if _is_missing(value):
        return None

    arr = np.asarray(value)
    if arr.size == 0:
        return None

    if arr.ndim == 0:
        scalar = arr.item()
        if _is_missing(scalar):
            return None

    if arr.dtype == object:
        # Some parquet rows store nested arrays with object dtype.
        if arr.shape == () and arr.item() is None:
            return None
        try:
            arr = np.stack(list(value))
        except TypeError:
            # Handle scalar objects (e.g., list stored as a single object cell).
            scalar = arr.item() if arr.shape == () else None
            if scalar is None:
                return None
            arr = np.asarray(scalar)

    if arr.shape == (3, 3):
        cam = arr
    elif arr.ndim == 1 and arr.size == 9:
        cam = arr.reshape(3, 3)
    else:
        raise ValueError(f"Expected cam_int shape (3,3) or (9,), got {arr.shape}")

    return torch.as_tensor(cam, dtype=torch.float32, device=device)


def _keypoint_error(pred: np.ndarray, gt: np.ndarray, idxs: Optional[np.ndarray] = None) -> float:
    if idxs is not None:
        pred = pred[idxs]
        gt = gt[idxs]
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def _build_gt_vertices(model, row: Dict, device: torch.device) -> np.ndarray:
    model_params = torch.as_tensor(np.asarray(row["model_params"]), dtype=torch.float32, device=device)[None]
    shape_params = torch.as_tensor(np.asarray(row["shape_params"]), dtype=torch.float32, device=device)[None]

    expr_params = None
    num_face_expr = 0
    if hasattr(model.head_pose, "num_face_comps"):
        num_face_expr = int(model.head_pose.num_face_comps)
    if hasattr(model.head_pose.mhr, "get_num_face_expression_blendshapes"):
        try:
            num_face_expr = int(model.head_pose.mhr.get_num_face_expression_blendshapes())
        except Exception:
            pass
    if num_face_expr > 0:
        expr_params = torch.zeros((1, num_face_expr), dtype=torch.float32, device=device)

    with torch.no_grad():
        gt_verts, _ = model.head_pose.mhr(shape_params, model_params, expr_params)
    return (gt_verts.squeeze(0).detach().cpu().numpy() / 100.0).astype(np.float64)


def main() -> None:
    args = parse_args()
    ann_dir = Path(args.annotation_dir)
    img_dir = Path(args.image_dir)
    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=args.checkpoint_path,
        device=str(device),
        mhr_path=args.mhr_path,
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    rows: List[Dict] = []
    processed = 0
    for rec in tqdm(_iter_rows(ann_dir, args.glob_pattern), desc="export-metrics"):
        if args.max_samples > 0 and processed >= args.max_samples:
            break

        try:
            image_path = img_dir / _get_img_name(rec)
            bbox = np.asarray(rec["bbox"], dtype=np.float32).reshape(1, 4)
            cam_int = _to_camera_intrinsics(rec.get("cam_int"), device)

            outputs = estimator.process_one_image(
                str(image_path),
                bboxes=bbox,
                cam_int=cam_int,
                inference_type="full",
            )
            if len(outputs) == 0:
                continue

            pred_vertices = np.asarray(outputs[0]["pred_vertices"], dtype=np.float64)
            pred_keypoints = np.asarray(outputs[0]["pred_keypoints_3d"], dtype=np.float64)
            gt_keypoints = _get_optional_keypoints3d(rec)
            gt_vertices = _build_gt_vertices(model, rec, device)

            cond = compute_conditioning(
                {
                    "vertices": pred_vertices,
                    "faces": estimator.faces,
                }
            )

            pve_value = float(np.linalg.norm(pred_vertices - gt_vertices, axis=-1).mean())
            cne_value = float(cne(pred_vertices, gt_vertices, cond["s"]))

            out_row = {
                "source_file": rec.get("_source_file", ""),
                "dataset": rec.get("dataset", ""),
                "image": rec.get("image", ""),
                "person_id": rec.get("person_id", -1),
                "kappa_geom": float(cond["kappa_geom"]),
                "kappa_spec": float(cond["kappa_spec"]),
                "mpjpe": _keypoint_error(pred_keypoints, gt_keypoints) if gt_keypoints is not None else np.nan,
                "hand_error": _keypoint_error(pred_keypoints, gt_keypoints, HAND_IDXS) if gt_keypoints is not None else np.nan,
                "face_error": _keypoint_error(pred_keypoints, gt_keypoints, FACE_IDXS) if gt_keypoints is not None else np.nan,
                "pve": pve_value,
                "cne": cne_value,
            }
            rows.append(out_row)
            processed += 1
        except Exception as exc:
            rows.append(
                {
                    "source_file": rec.get("_source_file", ""),
                    "dataset": rec.get("dataset", ""),
                    "image": rec.get("image", ""),
                    "person_id": rec.get("person_id", -1),
                    "error": str(exc),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_file, index=False)
    print(f"Wrote {len(frame)} rows to {out_file}")


if __name__ == "__main__":
    main()
