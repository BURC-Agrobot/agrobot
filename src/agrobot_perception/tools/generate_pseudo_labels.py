#!/usr/bin/env python3
"""
generate_pseudo_labels.py — Phase 3.2 self-training pseudo-label generator.

Runs the current best detector on the train set (no GT labels needed at
inference time), keeps high-confidence detections, converts masks to COCO
polygons, and writes a new COCO JSON. The LoRA trainer
(finetune_dino_mc_lora.py) can then ingest both the original train.json AND
this pseudo JSON via --coco-json + --coco-json-aux for a self-training cycle.

Filtering criteria (defaults match the plan):
  score   >= --tau-score        (e.g. 0.45 — well above the deploy threshold)
  pred_iou >= --tau-pred-iou    (e.g. 0.85 — SAM2 confident in mask shape)
  area in [--min-area, --max-area]  (e.g. [200, 30000] px in 518² space)

These thresholds are intentionally STRICT. Self-training works only when the
pseudo-labels are clean enough that the model gains representation rather than
amplifying its own errors. A loose τ_score creates a noisy-student feedback loop
that degrades. Better to keep fewer, cleaner pseudo-labels.

Why COCO-polygon output (not bbox-only):
  finetune_dino_mc_lora.py uses polygon-rasterized coverage masks as the
  per-patch positive weight. Bbox-only pseudo-labels would re-introduce the
  "30% leaf inside box" noise the polygon supervision was designed to fix.
  We rasterize each mask via cv2.findContours and keep the largest contour
  (single-blob assumption — the SAM2 prompt was a single point).

Sprint 4: Phase 3.2.

Usage [NUCBOX, ~50 min CPU for 643 images]:
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/tools/generate_pseudo_labels.py \\
    --train-list data/train_list.txt \\
    --output data/Laboro-Tomato/annotations/pseudo_train.json \\
    --detector sam2_amg --amg-points 28 --max-detections 60 \\
    --confidence 0.0 --nms-iou 0.5 \\
    --dino-weight 0.7 \\
    --query-embedding models/query_embedding_k4.pt \\
    --negative-embedding models/negative_embedding.pt \\
    --negative-weight 1.0 \\
    --dino-lora-path models/dino_mc_lora.pt \\
    --tau-score 0.45 --tau-pred-iou 0.85 \\
    --min-area 200 --max-area 30000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _setup_path(repo_root: Path) -> None:
    perception_dir = repo_root / "perception"
    if str(perception_dir) not in sys.path:
        sys.path.insert(0, str(perception_dir))


def _select_device():
    import torch
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mask_to_polygon(mask: np.ndarray) -> list[float] | None:
    """Largest external contour as flat [x0,y0,x1,y1,...]. None if degenerate."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    if mask.sum() < 4:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cnt.shape[0] < 3:
        return None
    return cnt.reshape(-1).astype(float).tolist()


def _polygon_in_orig_space(
    polygon_518: list[float],
    orig_w: int, orig_h: int,
) -> list[float]:
    """Reverse the letterbox transform to put the polygon back in original-image
    pixel coordinates so the COCO JSON is interchangeable with train.json
    (which stores polygons in original pixel space)."""
    scale = min(518 / orig_w, 518 / orig_h)
    new_w = int(orig_w * scale); new_h = int(orig_h * scale)
    pad_x = (518 - new_w) // 2
    pad_y = (518 - new_h) // 2
    pts = np.asarray(polygon_518, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] = (pts[:, 0] - pad_x) / scale
    pts[:, 1] = (pts[:, 1] - pad_y) / scale
    pts[:, 0] = np.clip(pts[:, 0], 0, orig_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, orig_h - 1)
    return pts.reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, required=True,
                        help="data/train_list.txt — one path per line.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output COCO JSON of pseudo-labels.")
    parser.add_argument("--detector", choices=["sam2_amg", "sam2_semantic"],
                        default="sam2_amg")
    parser.add_argument("--amg-points", type=int, default=28)
    parser.add_argument("--max-detections", type=int, default=60)
    parser.add_argument("--confidence", type=float, default=0.0,
                        help="In-detector confidence; keep at 0 and rely on --tau-score "
                             "for the self-training threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--dino-weight", type=float, default=0.7)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--query-embedding", type=Path, default=None)
    parser.add_argument("--negative-embedding", type=Path, default=None)
    parser.add_argument("--dino-lora-path", type=Path, default=None)
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--tau-score", type=float, default=0.45,
                        help="Minimum score to keep as pseudo-positive.")
    parser.add_argument("--tau-pred-iou", type=float, default=0.85,
                        help="Minimum SAM2 pred_iou to keep as pseudo-positive.")
    parser.add_argument("--min-area", type=int, default=200,
                        help="Minimum mask pixel count in 518² space.")
    parser.add_argument("--max-area", type=int, default=30000,
                        help="Maximum mask pixel count in 518² space (filters huge "
                             "background blobs that occasionally pass score gates).")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    repo_root = _repo_root()
    _setup_path(repo_root)
    os.chdir(repo_root)

    from agrobot_perception.utils.image_utils import preprocess_for_dino
    from agrobot_perception.detectors.sam2_amg_detector import SAM2AMGDetector

    def _abs_str(p: Path | None) -> str | None:
        if p is None:
            return None
        return str(p if p.is_absolute() else repo_root / p)

    detector = SAM2AMGDetector(
        device=_select_device(),
        sam2_checkpoint=_abs_str(args.sam2_checkpoint),
        confidence_threshold=args.confidence,
        max_detections=args.max_detections,
        points_per_side=args.amg_points,
        dino_score_weight=args.dino_weight,
        nms_iou_threshold=args.nms_iou,
        negative_weight=args.negative_weight,
        query_embedding_path=_abs_str(args.query_embedding),
        negative_embedding_path=_abs_str(args.negative_embedding),
        dino_lora_path=_abs_str(args.dino_lora_path),
    )

    train_list = args.train_list if args.train_list.is_absolute() \
        else repo_root / args.train_list
    img_paths: list[Path] = []
    with open(train_list) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                p = Path(line)
                img_paths.append(p if p.is_absolute() else repo_root / p)
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    coco = {
        "info": {"description": "Pseudo-labels from Phase 3.2 self-training"},
        "licenses": [{"id": 1, "name": "Pseudo-labels"}],
        "categories": [{"id": 1, "name": "tomato", "supercategory": "fruit"}],
        "images": [],
        "annotations": [],
    }

    next_image_id = 0
    next_ann_id = 0
    n_kept = 0
    n_dropped_score = 0
    n_dropped_iou = 0
    n_dropped_area = 0
    n_dropped_polygon = 0

    total = len(img_paths)
    for i, img_path in enumerate(img_paths):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]
        preprocessed = preprocess_for_dino(bgr, input_size=(518, 518))
        dets = detector.detect(preprocessed)

        kept_for_image = 0
        for d in dets:
            if d["score"] < args.tau_score:
                n_dropped_score += 1
                continue
            if d.get("pred_iou", 0.0) < args.tau_pred_iou:
                n_dropped_iou += 1
                continue
            mask = d.get("mask")
            if mask is None:
                continue
            area = int((mask > 0).sum())
            if area < args.min_area or area > args.max_area:
                n_dropped_area += 1
                continue
            poly_518 = _mask_to_polygon((mask > 0).astype(np.uint8))
            if poly_518 is None or len(poly_518) < 6:
                n_dropped_polygon += 1
                continue
            poly_orig = _polygon_in_orig_space(poly_518, orig_w, orig_h)

            x1, y1, x2, y2 = (float(b) for b in d["box"])
            scale = min(518 / orig_w, 518 / orig_h)
            pad_x = (518 - int(orig_w * scale)) // 2
            pad_y = (518 - int(orig_h * scale)) // 2
            x1o = (x1 - pad_x) / scale; y1o = (y1 - pad_y) / scale
            x2o = (x2 - pad_x) / scale; y2o = (y2 - pad_y) / scale

            coco["annotations"].append({
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": 1,
                "segmentation": [poly_orig],
                "bbox": [x1o, y1o, x2o - x1o, y2o - y1o],
                "area": float(area / (scale * scale)),  # original-space area
                "iscrowd": 0,
                "score": float(d["score"]),
                "pred_iou": float(d.get("pred_iou", 0.0)),
            })
            next_ann_id += 1
            kept_for_image += 1

        if kept_for_image > 0:
            coco["images"].append({
                "id": next_image_id,
                "file_name": img_path.name,
                "width": orig_w,
                "height": orig_h,
                "license": 1,
            })
            next_image_id += 1
            n_kept += kept_for_image

        if (i + 1) % 20 == 0:
            logger.info(
                "  [%d/%d] kept=%d  drops: score=%d iou=%d area=%d poly=%d",
                i + 1, total, n_kept,
                n_dropped_score, n_dropped_iou, n_dropped_area, n_dropped_polygon,
            )

    out = args.output if args.output.is_absolute() else repo_root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(coco, f)

    logger.info("")
    logger.info("Wrote %s", out)
    logger.info("Kept %d pseudo-positive masks across %d images.",
                n_kept, len(coco["images"]))
    logger.info("Drop reasons: score=%d  pred_iou=%d  area=%d  polygon=%d",
                n_dropped_score, n_dropped_iou, n_dropped_area, n_dropped_polygon)


if __name__ == "__main__":
    main()
