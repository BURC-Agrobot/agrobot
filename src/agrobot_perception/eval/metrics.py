"""mAP@0.5 and precision/recall for run_eval.py."""

from __future__ import annotations

from pathlib import Path


def _iou_box(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap_iou_threshold(
    all_detections: list[tuple[Path, list[dict]]],
    gt_by_image: dict[str, list[tuple[float, float, float, float]]],
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Compute average precision at one IoU threshold (single class).

    all_detections: list of (image_path, list of {"box": [x1,y1,x2,y2], "score": float}).
    gt_by_image: map from resolved image path to list of (x1,y1,x2,y2) gt boxes.
    Returns: (AP, precision, recall).
    """
    all_scores = []
    all_tp = []
    num_gt_total = 0

    for img_path, dets in all_detections:
        key = str(Path(img_path).resolve())
        gt_boxes = gt_by_image.get(key, [])
        num_gt_total += len(gt_boxes)

        if not gt_boxes and not dets:
            continue
        if not gt_boxes:
            for d in dets:
                all_scores.append(d["score"])
                all_tp.append(False)
            continue
        if not dets:
            continue

        # Sort by score descending
        sorted_dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        gt_matched = [False] * len(gt_boxes)

        for d in sorted_dets:
            all_scores.append(d["score"])
            box = d["box"]
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                iou = _iou_box(
                    (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    gt,
                )
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

    # Precision-recall curve: at each score threshold, compute prec and rec
    tp_cum = 0
    fp_cum = 0
    precisions = []
    recalls = []
    for tp in all_tp:
        if tp:
            tp_cum += 1
        else:
            fp_cum += 1
        prec = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0
        rec = tp_cum / num_gt_total if num_gt_total > 0 else 0
        precisions.append(prec)
        recalls.append(rec)

    # AP = area under PR curve (11-point interpolation or trapezoidal)
    if not recalls or not precisions:
        return 0.0, 0.0, 0.0
    ap = 0.0
    for i in range(len(recalls) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
    # Final precision/recall
    prec_final = precisions[-1] if precisions else 0.0
    rec_final = recalls[-1] if recalls else 0.0
    return ap, prec_final, rec_final
