"""
tta.py — Test-Time Augmentation wrapper for any detector with the standard
detect(preprocessed_chw) -> list[{box, score, label, mask}] interface.

Why TTA (Phase 1.2):
  Detection mAP is sensitive to single-shot quirks: SAM2 AMG places its grid
  deterministically, so masks for objects near grid boundaries are systematically
  worse than masks near grid centres. Horizontal flipping reorients those
  boundary cases. Quadrant cropping (delegated to the underlying detector via
  its own --amg-crops switch) raises effective resolution for small objects
  whose diameter falls below the AMG grid spacing (518/28 = 18.5 px at S4.12).

  TTA almost always helps mAP because mAP is the area under the PR curve and
  TTA adds independent score samples — the union of two reasonable detector
  passes has strictly more recall than either alone, and final NMS limits the
  precision cost.

Why this codebase's TTA does NOT include true 518/700 multi-scale:
  The detector's coverage-weighted scoring is hard-wired to the 37x37 DINOv2
  patch grid (= 518/14). Running at 700² would change the grid to 50x50 and
  invalidate every patch->mask coverage assumption inside SAM2AMGDetector.
  A pseudo-multi-scale via crop-and-resize is already implemented as
  --amg-crops (perception/agrobot_perception/detectors/sam2_amg_detector.py
  line 431, _generate_quadrant_masks). This wrapper composes cleanly with that
  flag — when both --tta and --amg-crops are set, each TTA pass internally
  generates full-image + 4 quadrant proposals.

Architecture:
  TTAWrapper(detector).detect(chw) runs detector twice (original + hflip),
  un-flips the flipped detections, concatenates, applies a final NMS, and
  caps at max_detections. The inner detector is otherwise unmodified.

Sprint 4: Phase 1.2 — gate experiment for the rest of the paper plan.
"""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class _DetectorLike(Protocol):
    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]: ...


def _box_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Box NMS — same call as sam2_amg_detector._nms for bit-identical behaviour."""
    if len(detections) <= 1 or iou_threshold <= 0:
        return detections
    boxes = np.array([d["box"] for d in detections], dtype=np.float32)
    scores = np.array([d["score"] for d in detections], dtype=np.float32)
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = boxes[:, 0]
    xywh[:, 1] = boxes[:, 1]
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    indices = cv2.dnn.NMSBoxes(
        xywh.tolist(), scores.tolist(),
        score_threshold=0.0, nms_threshold=iou_threshold,
    )
    if len(indices) == 0:
        return detections
    kept = np.asarray(indices).flatten()
    return [detections[int(i)] for i in kept]


def _flip_detections(dets: list[dict], image_width: int) -> list[dict]:
    """Mirror box and mask along the vertical axis. x_new = W - x_old."""
    out: list[dict] = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        new_box = [float(image_width - x2), float(y1),
                   float(image_width - x1), float(y2)]
        new_d = dict(d)
        new_d["box"] = new_box
        if "mask" in d and d["mask"] is not None:
            new_d["mask"] = np.ascontiguousarray(d["mask"][:, ::-1])
        out.append(new_d)
    return out


class TTAWrapper:
    """Wraps a detector to run (original + horizontal-flip) and merge results.

    The wrapper presents the same .detect(preprocessed_chw) -> list[dict]
    interface as the underlying detector, so it slots into run_eval.py without
    other changes.

    Args:
        base: the underlying detector (any object with .detect()).
        nms_iou_threshold: IoU threshold applied to the merged detection set.
            Use 0.5 to mirror sam2_amg_detector's default and the eval CLI.
        max_detections: cap on the merged set after NMS+sort.
        do_hflip: enable horizontal-flip pass. True is the standard setting;
            set False if you only want crop-augmentation via the inner detector.
    """

    def __init__(
        self,
        base: _DetectorLike,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 30,
        do_hflip: bool = True,
    ) -> None:
        self._base = base
        self._nms_iou = nms_iou_threshold
        self._max_detections = max_detections
        self._do_hflip = do_hflip

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        # Pass 1: original input. The inner detector applies its own filter+NMS+cap.
        d1 = self._base.detect(preprocessed_chw)

        merged: list[dict] = list(d1)

        if self._do_hflip:
            # Slice the width axis; .copy() because the inner detector reads
            # the array as a contiguous numpy buffer and downstream pytorch
            # forwards reject non-contiguous strides on some devices.
            flipped = np.ascontiguousarray(preprocessed_chw[:, :, ::-1])
            d2 = self._base.detect(flipped)
            d2 = _flip_detections(d2, image_width=preprocessed_chw.shape[2])
            merged.extend(d2)

        if not merged:
            return []

        # Final cross-pass NMS removes the systematic duplicate where the same
        # tomato is detected by both passes at slightly different boxes.
        merged = _box_nms(merged, self._nms_iou)
        merged.sort(key=lambda d: d["score"], reverse=True)
        return merged[: self._max_detections]
