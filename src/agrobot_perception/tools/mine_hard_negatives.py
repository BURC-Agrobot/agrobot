#!/usr/bin/env python3
"""
mine_hard_negatives.py — DEPRECATED (Sprint 4 Phase 2.1, val-leak risk).

THIS TOOL MINES NEGATIVES USING THE VAL SET'S GROUND TRUTH. The resulting
embedding is therefore not validation-independent and using it inflates val
mAP. Any artifact built by this script (e.g. models/hard_negative_embedding.pt)
has been moved to models/quarantine/ and must NOT be used in eval runs that
will be reported in the paper.

Replacement (Phase 2.2 / Phase 3): mine hard negatives on the TRAIN set using
either (a) the train COCO polygons to define what counts as background, or
(b) self-training pseudo-labels for a strict label-free variant.

Original docstring follows for historical reference only:

  Build a hard-negative DINOv2 embedding from false-positive detections.

Why this replaces the all-background negative:
  build_query_embedding.py builds the negative from ~patch_mask: every patch
  outside a GT bounding box. This includes sky, soil, wood beams — easy
  negatives that are already far from the tomato cluster in DINOv2 feature
  space. The contrastive term λ·neg_sim suppresses them but they were never a
  threat. The real confusers are leaves and stems that score HIGH on the tomato
  query. Those are the patches this script mines.

Protocol:
  1. Run the current-best detector on the val set.
  2. For each detection, check IoU against GT. If IoU < 0.5 → false positive.
  3. Extract DINOv2 patch tokens from the FP mask region (coverage-weighted).
  4. Stack all FP patches → mean → L2-normalise → save to hard_negative_embedding.pt.

Usage [MAC]:
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/tools/mine_hard_negatives.py \\
    --val-list data/val_list.txt \\
    --gt-csv data/val_gt.csv \\
    --output models/hard_negative_embedding.pt \\
    --amg-points 20 --confidence 0.2 --negative-weight 0.0

  Then eval with the mined embedding:
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/eval/run_eval.py \\
    --val-list data/val_list.txt --gt-csv data/val_gt.csv \\
    --detector sam2_amg --amg-points 20 --confidence 0.2 \\
    --negative-weight 1.2 --nms-iou 0.5

[MAC] / [NUCBOX] — runs on CPU. ~same wall time as one eval run.
Sprint 4: hard negative mining for contrastive embedding improvement (E5).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14
_DINO_INPUT_SIZE = 518
_DINO_GRID = _DINO_INPUT_SIZE // _DINO_PATCH_SIZE  # 37
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _select_device() -> torch.device:
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _iou_box(
    b1: tuple[float, float, float, float],
    b2: tuple[float, float, float, float],
) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _mask_to_coverage(seg: np.ndarray) -> np.ndarray:
    """518×518 bool mask → (1369,) float32 coverage weights."""
    seg_f = seg[:_DINO_GRID * _DINO_PATCH_SIZE, :_DINO_GRID * _DINO_PATCH_SIZE].astype(np.float32)
    blocks = seg_f.reshape(_DINO_GRID, _DINO_PATCH_SIZE, _DINO_GRID, _DINO_PATCH_SIZE)
    return blocks.mean(axis=(1, 3)).reshape(-1)  # (1369,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine hard-negative DINOv2 embeddings from FP detections on the val set."
    )
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("models/hard_negative_embedding.pt"),
        help="Output path for the hard-negative embedding tensor.",
    )
    parser.add_argument("--amg-points", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument(
        "--negative-weight", type=float, default=0.0,
        help=(
            "Contrastive weight during mining run. Set to 0.0 so the detector "
            "is not yet biased by the (soft) existing negative — we want to "
            "capture all high-scoring FPs, not just the ones that survive "
            "contrastive suppression."
        ),
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP/FP classification.")
    parser.add_argument("--max-fp-patches", type=int, default=200_000,
                        help="Cap on total FP patch vectors to keep (memory guard).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "perception"))

    from agrobot_perception.utils.image_utils import preprocess_for_dino
    from agrobot_perception.detectors.sam2_amg_detector import SAM2AMGDetector

    device = _select_device()
    logger.info("Device: %s", device)

    # Load val image list
    val_list = args.val_list if args.val_list.is_absolute() else repo_root / args.val_list
    image_paths: list[Path] = []
    with open(val_list) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            image_paths.append(p if p.is_absolute() else repo_root / p)

    if not image_paths:
        logger.error("No images in val list.")
        sys.exit(1)

    # Load GT
    gt_path = args.gt_csv if args.gt_csv.is_absolute() else repo_root / args.gt_csv
    gt_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_key = str(Path(row["image_path"].strip()).resolve())
            gt_by_image.setdefault(img_key, []).append(
                (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            )

    # Load detector (negative_weight=0 so we collect un-suppressed FPs)
    detector = SAM2AMGDetector(
        device=device,
        confidence_threshold=args.confidence,
        points_per_side=args.amg_points,
        nms_iou_threshold=args.nms_iou,
        negative_weight=args.negative_weight,
    )

    # Load DINOv2 separately (detector already holds it, but we need patch tokens
    # separately for all FP masks — reuse the detector's internal _dino model)
    dino = detector._dino

    fp_patches: list[torch.Tensor] = []
    n_fp = 0

    logger.info("Running inference on %d val images to collect FP detections...", len(image_paths))

    for idx, img_path in enumerate(image_paths):
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        preprocessed = preprocess_for_dino(bgr, input_size=(_DINO_INPUT_SIZE, _DINO_INPUT_SIZE))
        dets = detector.detect(preprocessed)

        if not dets:
            continue

        gt_boxes = gt_by_image.get(str(img_path.resolve()), [])

        # Get DINOv2 patch tokens for this image
        tensor = torch.from_numpy(preprocessed).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = dino.forward_features(tensor)
        patch_norms = F.normalize(
            feats["x_norm_patchtokens"].squeeze(0).float(), dim=1
        )  # (1369, 768)

        # Classify each detection as TP or FP
        gt_matched = [False] * len(gt_boxes)
        for det in sorted(dets, key=lambda d: d["score"], reverse=True):
            box = tuple(det["box"])
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                iou = _iou_box(box, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= args.iou_threshold and best_j >= 0:
                gt_matched[best_j] = True
                # TP — skip
            else:
                # FP — extract patch tokens weighted by mask coverage
                mask = det.get("mask")
                if mask is None:
                    continue
                coverage = _mask_to_coverage(mask.astype(bool))  # (1369,)
                w = torch.from_numpy(coverage).to(device)
                w_sum = w.sum()
                if w_sum < 1e-6:
                    continue
                # Coverage-weighted mean of patch tokens → one (768,) vector per FP
                weighted_mean = (patch_norms * w.unsqueeze(1)).sum(dim=0) / w_sum
                fp_patches.append(weighted_mean.cpu())
                n_fp += 1

        if sum(len(t.shape) for t in fp_patches) > 0 and n_fp >= args.max_fp_patches:
            logger.info("Reached max_fp_patches cap (%d). Stopping early.", args.max_fp_patches)
            break

        if (idx + 1) % 20 == 0:
            logger.info("  %d/%d images, %d FP patches collected.", idx + 1, len(image_paths), n_fp)

    if not fp_patches:
        logger.error("No FP patches collected. Lower --confidence or check GT CSV path.")
        sys.exit(1)

    logger.info("Total FP patch vectors: %d", len(fp_patches))

    all_fp = torch.stack(fp_patches, dim=0)  # (N, 768)
    mean_fp = all_fp.mean(dim=0)             # (768,)
    hard_neg = F.normalize(mean_fp, dim=0)   # L2-normalised

    out = args.output if args.output.is_absolute() else repo_root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hard_neg, str(out))
    logger.info("Saved hard-negative embedding to %s (shape=%s)", out, tuple(hard_neg.shape))
    logger.info("")
    logger.info("Eval with the new hard-negative embedding:")
    logger.info(
        "  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES=\"\" PYTHONPATH=perception "
        "python3 perception/eval/run_eval.py "
        "--val-list data/val_list.txt --gt-csv data/val_gt.csv "
        "--detector sam2_amg --amg-points 20 --confidence 0.2 "
        "--negative-weight 1.2 --nms-iou 0.5 "
        "--query-embedding models/query_embedding.pt"
    )
    logger.info(
        "  Use --negative-embedding models/hard_negative_embedding.pt in run_eval.py "
        "to A/B against the original background-mean negative_embedding.pt."
    )


if __name__ == "__main__":
    main()
