"""
metrics_coco.py — COCO-style detection metrics for paper-grade reporting.

Why this exists:
  perception/eval/metrics.py reports mAP@0.5 only — useful for sprint tracking
  but insufficient for any peer-reviewed publication. Detection venues require
  COCO mAP averaged over 10 IoU thresholds (0.50:0.05:0.95) plus per-area AP
  (AP_S, AP_M, AP_L) so reviewers can see whether gains come from easy large
  objects or hard small ones. This module computes all of those in one pass.

Implementation notes:
  - Single-class only (the project is binary tomato-vs-not). For multi-class
    extensions, run this per class and average.
  - Greedy IoU matching same as metrics.compute_ap_iou_threshold for backward
    consistency.
  - 101-point interpolation (COCO standard) for AP integration. We do NOT use
    the all-points trapezoidal rule because the COCO toolkit standard is
    101-point, and reviewers compare against that.
  - Area buckets follow COCO: small <32², medium 32²-96², large >96² (units
    are pixels in the detection's own coordinate space — 518×518 here).

Sprint 4: Phase 2.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# COCO standard IoU sweep and area thresholds.
_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_AREA_SMALL_MAX = 32 * 32        # < 1024 px²
_AREA_MEDIUM_MAX = 96 * 96       # < 9216 px²
_RECALL_INTERP = np.linspace(0.0, 1.0, 101)  # 101-point interpolation


Box = tuple[float, float, float, float]


def _iou_box(b1: Box, b2: Box) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _area_bucket(box: Box) -> str:
    area = _box_area(box)
    if area < _AREA_SMALL_MAX:
        return "small"
    if area < _AREA_MEDIUM_MAX:
        return "medium"
    return "large"


def _ap_at_iou(
    all_detections: list[tuple[Path, list[dict]]],
    gt_by_image: dict[str, list[Box]],
    iou_threshold: float,
    area_filter: str | None = None,
) -> tuple[float, float, float]:
    """Return (AP, precision, recall) at one IoU threshold.

    area_filter restricts both GT and TP-attribution to one COCO area bucket.
    Detections in other area buckets are still scored against (so they can be
    FPs against unmatched GTs in *this* bucket). GTs in other buckets count
    as ignored — neither denominator nor TP contributors.
    """
    all_scores: list[float] = []
    all_tp: list[bool] = []
    num_gt_total = 0

    for img_path, dets in all_detections:
        key = str(Path(img_path).resolve())
        gt_boxes_full = gt_by_image.get(key, [])

        if area_filter:
            in_bucket = [_area_bucket(g) == area_filter for g in gt_boxes_full]
        else:
            in_bucket = [True] * len(gt_boxes_full)
        num_gt_total += sum(in_bucket)

        if not gt_boxes_full and not dets:
            continue
        if not dets:
            continue

        sorted_dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        gt_matched = [False] * len(gt_boxes_full)

        for d in sorted_dets:
            if area_filter and _area_bucket(tuple(d["box"])) != area_filter:
                # Skip detections outside the bucket — they should not appear
                # in the in-bucket PR curve regardless of whether they match.
                continue
            all_scores.append(float(d["score"]))
            box = tuple(float(v) for v in d["box"])
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gt_boxes_full):
                if gt_matched[j] or not in_bucket[j]:
                    continue
                iou = _iou_box(box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold and best_j >= 0:
                gt_matched[best_j] = True
                all_tp.append(True)
            else:
                all_tp.append(False)

    if num_gt_total == 0:
        return 0.0, 0.0, 0.0

    if not all_tp:
        return 0.0, 0.0, 0.0

    # Sort by score so PR curve is monotone in score
    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp, dtype=np.float32)[order]
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recalls = tp_cum / num_gt_total

    # 101-point interpolation: at each interp recall, take max precision at
    # any recall >= interp recall (COCO + PASCAL VOC convention).
    ap = 0.0
    for r in _RECALL_INTERP:
        mask = recalls >= r
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p
    ap /= len(_RECALL_INTERP)

    return float(ap), float(precisions[-1]), float(recalls[-1])


def compute_coco_metrics(
    all_detections: list[tuple[Path, list[dict]]],
    gt_by_image: dict[str, list[Box]],
) -> dict[str, float]:
    """Compute the standard COCO detection metric set (single class).

    Returns a dict with:
      mAP        — mean over IoU 0.50:0.05:0.95
      AP50       — AP at IoU=0.50
      AP75       — AP at IoU=0.75
      AP_small   — mAP averaged over IoU thresholds, GTs+dets with area<32²
      AP_medium  — mAP averaged over IoU thresholds, 32²<=area<96²
      AP_large   — mAP averaged over IoU thresholds, area>=96²
      precision  — precision at IoU=0.5 (matches legacy metrics.py)
      recall     — recall at IoU=0.5 (matches legacy metrics.py)
    """
    ap50, p50, r50 = _ap_at_iou(all_detections, gt_by_image, 0.5)
    ap75, _, _ = _ap_at_iou(all_detections, gt_by_image, 0.75)

    ap_per_iou = []
    for thr in _IOU_THRESHOLDS:
        ap_t, _, _ = _ap_at_iou(all_detections, gt_by_image, float(thr))
        ap_per_iou.append(ap_t)
    map_all = float(np.mean(ap_per_iou))

    def _bucket_map(bucket: str) -> float:
        vals = [
            _ap_at_iou(all_detections, gt_by_image, float(thr), area_filter=bucket)[0]
            for thr in _IOU_THRESHOLDS
        ]
        return float(np.mean(vals))

    return {
        "mAP":       map_all,
        "AP50":      ap50,
        "AP75":      ap75,
        "AP_small":  _bucket_map("small"),
        "AP_medium": _bucket_map("medium"),
        "AP_large":  _bucket_map("large"),
        "precision": p50,
        "recall":    r50,
    }


def format_coco_table(metrics: dict[str, float]) -> str:
    """Pretty-printed COCO metric block for stdout."""
    return (
        "── COCO metrics (single-class tomato) ──\n"
        f"  mAP @[.5:.95]:  {metrics['mAP']:.4f}\n"
        f"  AP @0.50:       {metrics['AP50']:.4f}\n"
        f"  AP @0.75:       {metrics['AP75']:.4f}\n"
        f"  AP_small:       {metrics['AP_small']:.4f}\n"
        f"  AP_medium:      {metrics['AP_medium']:.4f}\n"
        f"  AP_large:       {metrics['AP_large']:.4f}\n"
        f"  Precision @.5:  {metrics['precision']:.4f}\n"
        f"  Recall @.5:     {metrics['recall']:.4f}"
    )
