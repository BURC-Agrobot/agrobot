"""
image_utils.py — Image preprocessing utilities for the perception pipeline.

This module is the single place where raw camera frames are conditioned before
being passed to any ML model. Centralizing this here means:
  - Sprint 2 (DINOv2/SAM2): just call `preprocess_for_dino()` — no boilerplate.
  - Sprint 3 (AMD edge): we can swap the backend (OpenCV → ROCm-accelerated) here
    without touching any node code.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple


# ─── Type Aliases ─────────────────────────────────────────────────────────────
# BGR image as produced by OpenCV and cv_bridge.
BGRImage = np.ndarray
# RGB image as required by PyTorch / most ML models.
RGBImage = np.ndarray


def bgr_to_rgb(image: BGRImage) -> RGBImage:
    """Convert an OpenCV BGR image to RGB.

    cv_bridge.imgmsg_to_cv2() returns BGR by default (OpenCV convention).
    All ML models (DINOv2, SAM2) expect RGB. Always call this first.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_with_aspect(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image to target_size while preserving aspect ratio.

    Pads with black (0) on the shorter dimension to reach the exact target size.
    This is the standard "letterboxing" used by YOLO-style detectors and DINOv2.

    Args:
        image: Input image (H, W, C).
        target_size: (width, height) tuple.
        interpolation: OpenCV interpolation flag. INTER_LINEAR for downscaling,
                       INTER_CUBIC for upscaling.

    Returns:
        Resized and padded image of shape (target_height, target_width, C).
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Create black canvas at target size and paste the resized image centered.
    canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas


def normalize_imagenet(image: RGBImage) -> np.ndarray:
    """Normalize a uint8 RGB image using ImageNet mean/std.

    DINOv2, SAM2, and most ViT-based models expect this normalization.
    Output is float32 in range ~[-2.1, 2.6].

    Args:
        image: uint8 RGB image (H, W, 3), values in [0, 255].

    Returns:
        float32 array (H, W, 3) normalized with ImageNet statistics.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (image.astype(np.float32) / 255.0 - mean) / std


def preprocess_for_dino(
    image: BGRImage,
    input_size: Tuple[int, int] = (518, 518),
) -> np.ndarray:
    """Full preprocessing pipeline for DINOv2.

    DINOv2 uses a patch size of 14, so input dimensions must be multiples of 14.
    518 = 37 * 14. This is the standard size used in the DINOv2 paper.

    Pipeline: BGR → RGB → letterbox resize → ImageNet normalize → CHW

    Args:
        image: Raw BGR frame from cv_bridge.
        input_size: (width, height). Default 518x518 for DINOv2-base/large.

    Returns:
        float32 array of shape (3, H, W), ready for torch.from_numpy().
    """
    rgb = bgr_to_rgb(image)
    resized = resize_with_aspect(rgb, input_size)
    normalized = normalize_imagenet(resized)
    # HWC → CHW (PyTorch convention: channels first)
    return np.transpose(normalized, (2, 0, 1))


def draw_detection_overlay(
    image: BGRImage,
    boxes: list,
    labels: list,
    scores: list,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> BGRImage:
    """Draw bounding boxes and labels onto a BGR image for visualization.

    Used by the debug publisher in TomatoDetectorNode to render detections
    as a `/agrobot/debug_image` topic viewable in Foxglove or rqt_image_view.

    Args:
        image: BGR image to draw on (modified in place).
        boxes: List of (x1, y1, x2, y2) pixel-coordinate bounding boxes.
        labels: List of string labels corresponding to each box.
        scores: List of float confidence scores.
        color: BGR color tuple for box outline.

    Returns:
        The annotated BGR image (same object as input, modified in place).
    """
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        text = f"{label}: {score:.2f}"
        # Background rectangle for text readability.
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            image, text, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return image
