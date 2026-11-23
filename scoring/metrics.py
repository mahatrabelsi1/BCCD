"""Submission validation and the official metric: mean Average Precision at IoU 0.5 (mAP@0.5).

AP is computed per class with all-point interpolation (PASCAL VOC 2010+ style):
  * predictions of a class are sorted by descending score over the whole test set;
  * a prediction is a true positive if its IoU with an unmatched ground-truth box of the
    same class and image is >= 0.5 (best-IoU match); each ground-truth box can be matched once;
  * AP is the area under the monotone precision envelope of the precision/recall curve.
mAP@0.5 is the unweighted mean of the three per-class APs (RBC, WBC, Platelets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ["RBC", "WBC", "Platelets"]
IOU_THRESHOLD = 0.5
SUBMISSION_COLUMNS = ["image_id", "label", "score", "xmin", "ymin", "xmax", "ymax"]
MAX_BOXES_PER_IMAGE = 300


class SubmissionError(ValueError):
    """Raised for an invalid submission; the message is safe to show to the participant."""


def load_submission(path: str, images: pd.DataFrame) -> pd.DataFrame:
    """Read and validate a predictions CSV against the expected image list (image_id,width,height)."""
    try:
        pred = pd.read_csv(path, dtype={"image_id": str, "label": str})
    except Exception as e:  # noqa: BLE001 - any parse error is a submission error
        raise SubmissionError(f"Could not parse predictions CSV: {type(e).__name__}") from None

    if list(pred.columns) != SUBMISSION_COLUMNS:
        raise SubmissionError(f"Columns must be exactly: {','.join(SUBMISSION_COLUMNS)}")
    if pred.empty:
        raise SubmissionError("Submission has no rows")
    if pred.isna().any().any():
        raise SubmissionError("Submission contains empty / NaN values")

    pred["image_id"] = pred["image_id"].str.strip()
    pred["label"] = pred["label"].str.strip()

    unknown = set(pred["image_id"]) - set(images["image_id"])
    if unknown:
        raise SubmissionError(f"{len(unknown)} unknown image_id(s), e.g. {sorted(unknown)[0]!r}")

    bad_labels = set(pred["label"]) - set(CLASSES)
    if bad_labels:
        raise SubmissionError(f"Invalid label(s) {sorted(bad_labels)}; allowed: {CLASSES}")

    num_cols = ["score", "xmin", "ymin", "xmax", "ymax"]
    try:
        pred[num_cols] = pred[num_cols].astype(float)
    except (TypeError, ValueError):
        raise SubmissionError("score and box coordinates must be numeric") from None
    if not np.isfinite(pred[num_cols].to_numpy()).all():
        raise SubmissionError("score and box coordinates must be finite")
    if ((pred["score"] < 0) | (pred["score"] > 1)).any():
        raise SubmissionError("score must be in [0, 1]")
    if ((pred["xmax"] <= pred["xmin"]) | (pred["ymax"] <= pred["ymin"])).any():
        raise SubmissionError("Every box needs xmax > xmin and ymax > ymin")

    per_image = pred.groupby("image_id").size()
    if (per_image > MAX_BOXES_PER_IMAGE).any():
        raise SubmissionError(f"At most {MAX_BOXES_PER_IMAGE} boxes per image are allowed")

    # Clip to image bounds (pixel coordinates, origin top-left).
    size = images.set_index("image_id")
    w = pred["image_id"].map(size["width"]).astype(float)
    h = pred["image_id"].map(size["height"]).astype(float)
    pred["xmin"] = pred["xmin"].clip(lower=0).clip(upper=w)
    pred["xmax"] = pred["xmax"].clip(lower=0).clip(upper=w)
    pred["ymin"] = pred["ymin"].clip(lower=0).clip(upper=h)
    pred["ymax"] = pred["ymax"].clip(lower=0).clip(upper=h)
    return pred


def _iou(box: np.ndarray, gts: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], gts[:, 0])
    iy1 = np.maximum(box[1], gts[:, 1])
    ix2 = np.minimum(box[2], gts[:, 2])
    iy2 = np.minimum(box[3], gts[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_g = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    union = area_b + area_g - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(pred: pd.DataFrame, gt: pd.DataFrame, iou_thr: float = IOU_THRESHOLD) -> float:
    """AP for a single class. pred/gt are already filtered to that class."""
    n_gt = len(gt)
    if n_gt == 0:
        return float("nan")
    if pred.empty:
        return 0.0

    coords = ["xmin", "ymin", "xmax", "ymax"]
    gt_boxes = {iid: g[coords].to_numpy(float) for iid, g in gt.groupby("image_id")}
    matched = {iid: np.zeros(len(b), dtype=bool) for iid, b in gt_boxes.items()}

    # Stable sort so ties are resolved deterministically by file order.
    pred = pred.sort_values("score", ascending=False, kind="mergesort")
    tp = np.zeros(len(pred))
    for i, (iid, box) in enumerate(zip(pred["image_id"], pred[coords].to_numpy(float))):
        gts = gt_boxes.get(iid)
        if gts is None:
            continue
        ious = _iou(box, gts)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[iid][j]:
            matched[iid][j] = True
            tp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Return {'map50': float, 'ap_RBC': float, 'ap_WBC': float, 'ap_Platelets': float}."""
    out = {}
    for c in CLASSES:
        out[f"ap_{c}"] = average_precision(pred[pred["label"] == c], gt[gt["label"] == c])
    out["map50"] = float(np.nanmean([out[f"ap_{c}"] for c in CLASSES]))
    return out
