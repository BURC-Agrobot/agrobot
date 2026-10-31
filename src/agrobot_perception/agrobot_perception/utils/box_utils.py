"""Compare and filter [x1, y1, x2, y2] detection boxes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Return the intersection over union of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def nms_detections(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Apply Non-Maximum Suppression. Keeps highest-scoring per overlap cluster."""
    # Import here so box_iou works for metrics code that does not need OpenCV
    import cv2

    if len(detections) <= 1:
        return detections
    boxes = np.array([d["box"] for d in detections], dtype=np.float32)
    scores = np.array([d["score"] for d in detections], dtype=np.float32)
    # cv2.dnn.NMSBoxes expects [x, y, w, h]
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = boxes[:, 0]  # x
    xywh[:, 1] = boxes[:, 1]  # y
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
    indices = cv2.dnn.NMSBoxes(
        xywh.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    if len(indices) == 0:
        return detections
    kept = np.asarray(indices).flatten()
    return [detections[int(i)] for i in kept]
