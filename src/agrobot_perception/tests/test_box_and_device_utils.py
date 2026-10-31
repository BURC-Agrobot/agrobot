"""Tests for the box, device and image helpers shared across perception."""

import numpy as np
import pytest
import torch

from agrobot_perception.utils.box_utils import box_iou, nms_detections
from agrobot_perception.utils.device import select_device
from agrobot_perception.utils.image_utils import denormalize_imagenet, normalize_imagenet


def test_box_iou_of_identical_boxes_is_one():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_box_iou_of_separate_boxes_is_zero():
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_box_iou_of_half_overlap():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_box_iou_of_empty_boxes_is_zero():
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_nms_keeps_the_best_of_overlapping_boxes():
    low = {"box": [0, 0, 10, 10], "score": 0.5}
    high = {"box": [1, 0, 11, 10], "score": 0.9}
    far = {"box": [50, 50, 60, 60], "score": 0.7}
    assert nms_detections([low, high, far], 0.5) == [high, far]


def test_nms_returns_single_detection_unchanged():
    detection = {"box": [0, 0, 10, 10], "score": 0.5}
    assert nms_detections([detection], 0.5) == [detection]


def test_force_cpu_overrides_any_gpu(monkeypatch):
    monkeypatch.setenv("AGROBOT_FORCE_CPU", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device() == torch.device("cpu")


def test_cuda_is_used_when_available(monkeypatch):
    monkeypatch.delenv("AGROBOT_FORCE_CPU", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device() == torch.device("cuda")


def test_denormalize_reverses_normalize():
    image = np.array([[[0, 128, 255], [10, 20, 30]]], dtype=np.uint8)
    chw = np.transpose(normalize_imagenet(image), (2, 0, 1))
    restored = denormalize_imagenet(chw)
    assert restored.dtype == np.uint8
    assert np.abs(restored.astype(int) - image.astype(int)).max() <= 1
