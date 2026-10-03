#!/usr/bin/env python3
"""
build_val_gt_csv.py — Convert Laboro Tomato YOLO val labels to val_gt.csv.

Why this script exists:
  run_eval.py computes mAP@0.5 by comparing detector output boxes against a
  ground truth CSV. The detector outputs boxes in 518×518 letterboxed pixel
  space (the same space DINOv2 sees). The YOLO labels are in normalized
  original-image space. This script bridges the two by applying the same
  letterbox transform that preprocess_for_dino() uses.

Coordinate transform (per image):
  YOLO: class cx cy w h  (normalised to [0,1] relative to original image dims)
  Step 1: multiply by original image dims → pixel coords in original space
  Step 2: apply letterbox scale = min(518/orig_w, 518/orig_h)
  Step 3: add padding offset (pad_x, pad_y) = ((518 - new_w)//2, (518 - new_h)//2)
  Result: x1,y1,x2,y2 in 518×518 space — directly comparable to detector output

Why per-image dimension reading:
  Most Laboro Tomato images are 4032×3024 (12MP landscape), but we read each
  image's actual dims rather than hardcoding. One different-resolution image
  silently breaks all IoU calculations if dims are assumed.

Usage (from repo root):
  python3 perception/tools/build_val_gt_csv.py \
    --val-images data/Laboro-Tomato/val/images \
    --val-labels data/Laboro-Tomato/val/labels \
    --output data/val_gt.csv

  Then run eval with mAP:
    AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
      python3 perception/eval/run_eval.py \\
      --val-list data/val_list.txt \\
      --gt-csv data/val_gt.csv \\
      --confidence 0.3
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_INPUT_SIZE = 518  # DINOv2 input resolution


def _letterbox_box(
    cx_n: float, cy_n: float, w_n: float, h_n: float,
    orig_w: int, orig_h: int,
    target: int = _INPUT_SIZE,
) -> tuple[float, float, float, float]:
    """Convert one YOLO box to target×target letterboxed pixel coordinates.

    Args:
        cx_n, cy_n, w_n, h_n: YOLO normalised centre + size.
        orig_w, orig_h: Original image pixel dimensions.
        target: Square target size (default 518).

    Returns:
        (x1, y1, x2, y2) in target×target pixel space.
    """
    # Step 1: denormalise to original pixel space
    cx = cx_n * orig_w
    cy = cy_n * orig_h
    bw = w_n  * orig_w
    bh = h_n  * orig_h
    x1_orig = cx - bw / 2
    y1_orig = cy - bh / 2
    x2_orig = cx + bw / 2
    y2_orig = cy + bh / 2

    # Step 2: compute letterbox transform (same logic as resize_with_aspect)
    scale = min(target / orig_w, target / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2

    # Step 3: apply scale + padding
    x1 = x1_orig * scale + pad_x
    y1 = y1_orig * scale + pad_y
    x2 = x2_orig * scale + pad_x
    y2 = y2_orig * scale + pad_y

    # Clamp to canvas bounds
    x1 = max(0.0, min(float(target), x1))
    y1 = max(0.0, min(float(target), y1))
    x2 = max(0.0, min(float(target), x2))
    y2 = max(0.0, min(float(target), y2))

    return x1, y1, x2, y2


def build_csv(
    val_images_dir: Path,
    val_labels_dir: Path,
    output_path: Path,
    val_list: Path | None = None,
) -> None:
    # Determine which images to process
    if val_list and val_list.exists():
        repo_root = Path(__file__).resolve().parent.parent.parent
        image_paths = []
        with open(val_list) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = repo_root / p
                image_paths.append(p)
        logger.info("Using %d images from val_list.txt.", len(image_paths))
    else:
        image_paths = sorted(val_images_dir.glob("*.jpg")) + \
                      sorted(val_images_dir.glob("*.png"))
        logger.info("No val_list provided — using all %d images in %s.",
                    len(image_paths), val_images_dir)

    rows_written = 0
    images_with_gt = 0
    skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "x1", "y1", "x2", "y2", "label"])

        for img_path in image_paths:
            label_path = val_labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                skipped += 1
                continue

            # Read actual image dims — do NOT hardcode 4032×3024
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning("Cannot read image: %s", img_path)
                skipped += 1
                continue
            orig_h, orig_w = img.shape[:2]

            image_had_boxes = False
            with open(label_path) as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    _, cx_n, cy_n, w_n, h_n = (float(p) for p in parts[:5])

                    x1, y1, x2, y2 = _letterbox_box(
                        cx_n, cy_n, w_n, h_n, orig_w, orig_h
                    )

                    # Skip degenerate boxes (can happen with very small YOLO annotations)
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        continue

                    # image_path stored relative to repo root for portability
                    repo_root = Path(__file__).resolve().parent.parent.parent
                    try:
                        rel_path = img_path.relative_to(repo_root)
                    except ValueError:
                        rel_path = img_path

                    writer.writerow([str(rel_path), f"{x1:.2f}", f"{y1:.2f}",
                                     f"{x2:.2f}", f"{y2:.2f}", "tomato"])
                    rows_written += 1
                    image_had_boxes = True

            if image_had_boxes:
                images_with_gt += 1

    logger.info(
        "Wrote %d GT boxes from %d images to %s. Skipped %d.",
        rows_written, images_with_gt, output_path, skipped,
    )
    logger.info("Average %.1f boxes per image.", rows_written / max(images_with_gt, 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Laboro Tomato YOLO val labels to val_gt.csv "
                    "in 518×518 letterboxed pixel space for run_eval.py."
    )
    parser.add_argument(
        "--val-images", type=Path, required=True,
        help="Path to val/images/ directory.",
    )
    parser.add_argument(
        "--val-labels", type=Path, required=True,
        help="Path to val/labels/ directory (YOLO .txt files).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/val_gt.csv"),
        help="Output CSV path (default: data/val_gt.csv).",
    )
    parser.add_argument(
        "--val-list", type=Path, default=None,
        help="Optional: data/val_list.txt to restrict to the exact eval image set.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    val_images = args.val_images if args.val_images.is_absolute() \
        else repo_root / args.val_images
    val_labels = args.val_labels if args.val_labels.is_absolute() \
        else repo_root / args.val_labels
    output = args.output if args.output.is_absolute() else repo_root / args.output
    val_list = (args.val_list if args.val_list.is_absolute() else repo_root / args.val_list) \
        if args.val_list else None

    if not val_images.exists():
        logger.error("val-images not found: %s", val_images)
        sys.exit(1)
    if not val_labels.exists():
        logger.error("val-labels not found: %s", val_labels)
        sys.exit(1)

    build_csv(val_images, val_labels, output, val_list=val_list)


if __name__ == "__main__":
    main()
