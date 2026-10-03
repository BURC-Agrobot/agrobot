"""
test_image_utils.py — Unit tests for agrobot_perception.utils.image_utils

These tests run with `bazel test //perception:image_utils_test`.
They do NOT require ROS 2 or Docker — only numpy and opencv, which Bazel
provides hermetically from the pip lock file.

Design principle: test the mathematical invariants of each function,
not the implementation. If we swap cv2.resize for a GPU-accelerated
version in Sprint 3, these tests should still pass unchanged.
"""

import numpy as np
import pytest

from agrobot_perception.utils.image_utils import (
    bgr_to_rgb,
    resize_with_aspect,
    normalize_imagenet,
    preprocess_for_dino,
)


class TestBgrToRgb:
    def test_channel_order_is_swapped(self):
        """Blue channel (index 0 in BGR) becomes red channel (index 2 in RGB)."""
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # pure blue in BGR
        rgb = bgr_to_rgb(bgr)
        assert rgb[:, :, 2].mean() == 255, "Blue channel should map to RGB index 2"
        assert rgb[:, :, 0].mean() == 0, "Red channel (RGB index 0) should be zero"

    def test_output_shape_preserved(self):
        bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        rgb = bgr_to_rgb(bgr)
        assert rgb.shape == bgr.shape


class TestResizeWithAspect:
    def test_output_is_exact_target_size(self):
        """Output must always match target_size exactly."""
        image = np.zeros((100, 300, 3), dtype=np.uint8)  # wide image
        result = resize_with_aspect(image, target_size=(518, 518))
        assert result.shape == (518, 518, 3)

    def test_square_image_no_padding(self):
        """A square input to a square target should fill the canvas completely."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        result = resize_with_aspect(image, target_size=(100, 100))
        # No padding: all pixels should be non-zero (mean ≈ 128, not 0).
        assert result.mean() > 100

    def test_wide_image_has_vertical_padding(self):
        """A 2:1 wide image into a square should have black bars top and bottom."""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        result = resize_with_aspect(image, target_size=(100, 100))
        # Top row must be black padding.
        assert result[0, 50, 0] == 0, "Top row should be black padding"

    def test_preserves_dtype(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        result = resize_with_aspect(image, (100, 100))
        assert result.dtype == np.uint8


class TestNormalizeImagenet:
    def test_output_dtype_is_float32(self):
        image = np.ones((10, 10, 3), dtype=np.uint8) * 128
        result = normalize_imagenet(image)
        assert result.dtype == np.float32

    def test_zero_pixel_maps_to_negative_value(self):
        """A black pixel (0, 0, 0) should normalize to a negative value
        since ImageNet mean > 0."""
        black = np.zeros((1, 1, 3), dtype=np.uint8)
        result = normalize_imagenet(black)
        assert result.min() < 0

    def test_output_range_is_reasonable(self):
        """All uint8 inputs should map to roughly [-2.5, 2.5]."""
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = normalize_imagenet(image)
        assert result.min() >= -3.0
        assert result.max() <= 3.0


class TestPreprocessForDino:
    def test_output_shape_is_chw(self):
        """DINOv2 expects (3, H, W) — channels first."""
        bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess_for_dino(bgr, input_size=(518, 518))
        assert result.shape == (3, 518, 518), f"Expected (3, 518, 518), got {result.shape}"

    def test_output_is_float32(self):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess_for_dino(bgr)
        assert result.dtype == np.float32

    def test_input_size_is_respected(self):
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess_for_dino(bgr, input_size=(224, 224))
        assert result.shape == (3, 224, 224)
