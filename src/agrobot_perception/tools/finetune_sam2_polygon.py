#!/usr/bin/env python3
"""
finetune_sam2_polygon.py — Fine-tune SAM2 mask decoder using COCO polygon GT.

Why this replaces finetune_sam2_decoder.py (rect-proxy version):
  The rect-proxy training overfit to loss=0.012 but mAP remained 0.0000.
  Root cause: rectangular GT masks teach the decoder to fill boxes, not produce
  tomato-shaped masks. A rectangle against a round GT annotation has IoU ≤ 0.78
  by geometry — most detections never reach IoU@0.5 against GT.

  Laboro Tomato ships COCO JSON with per-instance polygon segmentation
  (data/Laboro-Tomato/annotations/train.json). Each polygon traces the actual
  tomato outline (~23 points). Rasterizing these polygons as GT masks gives the
  decoder a correct, round target. Expected mAP improvement: significant.

Training changes vs rect version:
  - GT mask: rasterized COCO polygon (round) not filled bbox (rectangle)
  - Epochs: 5 not 10 (rect version overfit; polygons are harder, less overfit risk)
  - Everything else identical: freeze encoder, train decoder, BCE+Dice, AdamW

Usage (from repo root):
  # Smoke-test: 10 images, 2 epochs (~3 min on NucBox CPU)
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/tools/finetune_sam2_polygon.py \\
    --coco-json data/Laboro-Tomato/annotations/train.json \\
    --train-images data/Laboro-Tomato/train/images \\
    --sam2-checkpoint models/sam2/sam2.1_hiera_small.pt \\
    --output models/sam2/sam2_tomato_finetuned.pt \\
    --epochs 2 --max-images 10

  # Full training: 643 images, 5 epochs (~45–60 min on NucBox CPU)
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/tools/finetune_sam2_polygon.py \\
    --coco-json data/Laboro-Tomato/annotations/train.json \\
    --train-images data/Laboro-Tomato/train/images \\
    --sam2-checkpoint models/sam2/sam2.1_hiera_small.pt \\
    --output models/sam2/sam2_tomato_finetuned.pt \\
    --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_INPUT_SIZE = 518
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


def load_coco_annotations(
    coco_json: Path,
    images_dir: Path,
    max_images: int = 0,
) -> list[dict]:
    """Load COCO annotations and match to image files.

    Returns list of dicts:
        {
          "image_path": Path,
          "orig_w": int,
          "orig_h": int,
          "instances": [{"bbox_xywh": [...], "polygon": [...flat xy...]}]
        }
    """
    with open(coco_json) as f:
        coco = json.load(f)

    # Build image id → metadata map
    img_meta: dict[int, dict] = {im["id"]: im for im in coco["images"]}

    # Build image id → annotations map
    ann_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        if not ann.get("segmentation"):
            continue
        iid = ann["image_id"]
        ann_by_img.setdefault(iid, []).append(ann)

    records = []
    img_ids = list(img_meta.keys())
    if max_images > 0:
        img_ids = img_ids[:max_images]

    for iid in img_ids:
        meta = img_meta[iid]
        img_path = images_dir / meta["file_name"]
        if not img_path.exists():
            continue
        anns = ann_by_img.get(iid, [])
        if not anns:
            continue

        instances = []
        for ann in anns:
            seg = ann["segmentation"]
            if not seg or not isinstance(seg[0], list):
                continue
            instances.append({
                "bbox_xywh": ann["bbox"],
                "polygon": seg[0],  # flat [x0,y0,x1,y1,...] in original pixel coords
            })

        if instances:
            records.append({
                "image_path": img_path,
                "orig_w": meta["width"],
                "orig_h": meta["height"],
                "instances": instances,
            })

    logger.info("Loaded %d images with COCO polygon annotations.", len(records))
    return records


def _letterbox_rgb(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """BGR → RGB uint8 518×518 letterboxed. Returns (rgb_hwc, scale, pad_x, pad_y)."""
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(rgb, (new_w, new_h))
    canvas = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
    pad_x = (_INPUT_SIZE - new_w) // 2
    pad_y = (_INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _polygon_to_518_mask(
    polygon_flat: list[float],
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray | None:
    """Rasterize a COCO polygon into 518×518 letterboxed binary mask.

    polygon_flat: flat [x0,y0,x1,y1,...] in original image pixel coords.
    Returns float32 (518, 518) mask, or None if degenerate.
    """
    pts = np.array(polygon_flat, dtype=np.float64).reshape(-1, 2)
    # Apply letterbox transform
    pts[:, 0] = pts[:, 0] * scale + pad_x
    pts[:, 1] = pts[:, 1] * scale + pad_y
    pts = np.clip(pts, 0, _INPUT_SIZE - 1).astype(np.int32)

    mask = np.zeros((_INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
    cv2.fillPoly(mask, [pts], 1.0)

    if mask.sum() < 4:
        return None
    return mask



def _sample_interior_points(
    polygon_flat: list[float],
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    n_points: int = 3,
) -> list[tuple[float, float]] | None:
    """Sample n_points interior points from a COCO polygon in 518×518 space.

    Strategy: centroid + jittered copies. The centroid of a convex polygon is
    always interior. For non-convex shapes, jittered copies may land outside —
    we filter those with a point-in-polygon test.

    This is needed because SAM2 AMG uses point prompts at runtime but the
    previous fine-tuning used box prompts. The mask decoder is a cross-attention
    transformer: box prompts emit 2 corner tokens, point prompts emit 1 point +
    1 label token. Training on the wrong prompt type is a systematic distribution
    shift that caps mask quality regardless of GT polygon quality.
    """
    pts_orig = np.array(polygon_flat, dtype=np.float64).reshape(-1, 2)
    if len(pts_orig) < 3:
        return None

    # Centroid in original space, then project into 518×518 letterbox
    cx = pts_orig[:, 0].mean() * scale + pad_x
    cy = pts_orig[:, 1].mean() * scale + pad_y

    cx = float(np.clip(cx, 0, _INPUT_SIZE - 1))
    cy = float(np.clip(cy, 0, _INPUT_SIZE - 1))

    interior = [(cx, cy)]

    # Add jittered points within ±25% of the bounding box extent, keeping only
    # those that fall inside the (letterboxed) polygon via OpenCV pointPolygonTest
    pts_518 = pts_orig.copy()
    pts_518[:, 0] = pts_518[:, 0] * scale + pad_x
    pts_518[:, 1] = pts_518[:, 1] * scale + pad_y
    pts_518 = np.clip(pts_518, 0, _INPUT_SIZE - 1).astype(np.float32)

    bb_w = pts_518[:, 0].max() - pts_518[:, 0].min()
    bb_h = pts_518[:, 1].max() - pts_518[:, 1].min()
    jitter_x = max(bb_w * 0.2, 2.0)
    jitter_y = max(bb_h * 0.2, 2.0)

    rng = np.random.default_rng()
    attempts = 0
    while len(interior) < n_points and attempts < 30:
        attempts += 1
        jx = float(rng.uniform(-jitter_x, jitter_x))
        jy = float(rng.uniform(-jitter_y, jitter_y))
        px = float(np.clip(cx + jx, 0, _INPUT_SIZE - 1))
        py = float(np.clip(cy + jy, 0, _INPUT_SIZE - 1))
        # cv2.pointPolygonTest returns > 0 for inside, 0 on boundary, < 0 outside
        if cv2.pointPolygonTest(pts_518.reshape(-1, 1, 2), (px, py), False) >= 0:
            interior.append((px, py))

    return interior if interior else None


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f   = pred.reshape(-1)
    target_f = target.reshape(-1)
    inter = (pred_f * target_f).sum()
    return 1.0 - (2.0 * inter + eps) / (pred_f.sum() + target_f.sum() + eps)


def _combined_loss(logit: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    bce  = F.binary_cross_entropy_with_logits(logit, gt)
    dice = _dice_loss(torch.sigmoid(logit), gt)
    return 0.5 * bce + 0.5 * dice


def _run_decoder(
    sam2_model,
    image_embed,
    high_res_feats,
    device: torch.device,
    box_tensor: torch.Tensor | None = None,
    point_coords: torch.Tensor | None = None,
    point_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run frozen prompt encoder + trainable mask decoder. Returns (518,518) logit.

    Accepts either box_tensor or (point_coords, point_labels) — never both.
    Caller is responsible for passing the right prompt type for the target
    runtime: box prompts for box-prompted predictors, point prompts for AMG.
    """
    points_arg = None
    if point_coords is not None and point_labels is not None:
        points_arg = (point_coords, point_labels)

    with torch.no_grad():
        sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
            points=points_arg, boxes=box_tensor, masks=None
        )
    decoder_out = sam2_model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    low_res = decoder_out[0]  # (1, 1, H/4, W/4)
    logit = F.interpolate(
        low_res.float(), size=(_INPUT_SIZE, _INPUT_SIZE),
        mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0)  # (518, 518) with grad_fn
    return logit


def train(
    records: list[dict],
    sam2_ckpt: Path,
    output_path: Path,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
) -> None:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Suppress SAM2's per-frame INFO logs — must happen after import
    # so the loggers actually exist when we set their levels.
    for _name in logging.root.manager.loggerDict:
        if _name.startswith("sam2"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    _SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

    logger.info("Loading SAM2 from %s...", sam2_ckpt)
    sam2_model = build_sam2(_SAM2_CFG, str(sam2_ckpt), device=device)

    for name, param in sam2_model.named_parameters():
        if "image_encoder" in name or "sam_prompt_encoder" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    trainable = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in sam2_model.parameters())
    logger.info("Parameters: %d trainable / %d total (%.1f%% frozen).",
                trainable, total, 100.0 * (1 - trainable / total))

    predictor = SAM2ImagePredictor(sam2_model)
    optimizer = torch.optim.AdamW(
        [p for p in sam2_model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(records), eta_min=lr / 10,
    )

    best_loss = float("inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    with logging_redirect_tqdm():
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches  = 0
            random.shuffle(records)

            bar = tqdm(
                records,
                desc=f"Epoch {epoch:>{len(str(epochs))}}/{epochs}",
                unit="img",
                ncols=72,
                leave=False,
            )

            for rec in bar:
                bgr = cv2.imread(str(rec["image_path"]))
                if bgr is None:
                    continue

                orig_w, orig_h = rec["orig_w"], rec["orig_h"]
                rgb_hwc, scale, pad_x, pad_y = _letterbox_rgb(bgr)

                logging.disable(logging.INFO)
                with torch.no_grad():
                    predictor.set_image(rgb_hwc)
                logging.disable(logging.NOTSET)

                image_embed    = predictor._features["image_embed"]
                high_res_feats = predictor._features.get("high_res_feats")

                image_loss = torch.tensor(0.0, device=device)
                n_boxes    = 0

                for inst in rec["instances"]:
                    gt_mask_np = _polygon_to_518_mask(
                        inst["polygon"], orig_w, orig_h, scale, pad_x, pad_y
                    )
                    if gt_mask_np is None:
                        continue
                    gt_mask = torch.from_numpy(gt_mask_np).to(device)

                    # Sample interior points matching the AMG runtime prompt type.
                    # Average loss over n_points random foreground points so the
                    # decoder sees a distribution of point positions rather than
                    # always the centroid (which would overfit to a single prompt).
                    interior_pts = _sample_interior_points(
                        inst["polygon"], orig_w, orig_h, scale, pad_x, pad_y,
                        n_points=3,
                    )
                    if not interior_pts:
                        continue

                    inst_loss = torch.tensor(0.0, device=device)
                    for px, py in interior_pts:
                        # SAM2 prompt encoder expects (B, N, 2) coords and (B, N) labels
                        pt_coords = torch.tensor([[[px, py]]], dtype=torch.float32, device=device)
                        pt_labels = torch.tensor([[1]], dtype=torch.int32, device=device)
                        logit = _run_decoder(
                            sam2_model, image_embed, high_res_feats, device,
                            point_coords=pt_coords, point_labels=pt_labels,
                        )
                        inst_loss = inst_loss + _combined_loss(logit, gt_mask)

                    image_loss = image_loss + inst_loss / len(interior_pts)
                    n_boxes   += 1

                if n_boxes == 0:
                    continue

                image_loss = image_loss / n_boxes
                optimizer.zero_grad()
                image_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in sam2_model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += image_loss.item()
                n_batches  += 1

            bar.close()

            if n_batches == 0:
                logger.warning("Epoch %d: no batches processed.", epoch)
                continue

            mean_loss = epoch_loss / n_batches
            saved = ""
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(sam2_model.state_dict(), str(output_path))
                saved = "  ✓ saved"
            tqdm.write(f"  └─ Epoch {epoch}/{epochs}  loss={mean_loss:.4f}  best={best_loss:.4f}{saved}")

    print()
    logger.info("Training complete. Best loss: %.4f", best_loss)
    logger.info("Checkpoint: %s", output_path)
    logger.info("")
    logger.info("Run eval:")
    logger.info(
        "  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES=\"\" PYTHONPATH=perception "
        "python3 perception/eval/run_eval.py "
        "--val-list data/val_list.txt --gt-csv data/val_gt.csv --confidence 0.3"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM2 mask decoder with COCO polygon GT masks."
    )
    parser.add_argument("--coco-json", type=Path, required=True,
                        help="Path to annotations/train.json (COCO format).")
    parser.add_argument("--train-images", type=Path, required=True,
                        help="Path to train/images/ directory.")
    parser.add_argument("--sam2-checkpoint", type=Path,
                        default=Path("models/sam2/sam2.1_hiera_small.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("models/sam2/sam2_tomato_finetuned.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-images", type=int, default=0,
                        help="0 = all. Use 10 for smoke-test.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else repo_root / p

    coco_json    = _abs(args.coco_json)
    train_images = _abs(args.train_images)
    sam2_ckpt    = _abs(args.sam2_checkpoint)
    output       = _abs(args.output)

    for p, name in [(coco_json, "coco-json"), (train_images, "train-images"),
                    (sam2_ckpt, "sam2-checkpoint")]:
        if not p.exists():
            logger.error("%s not found: %s", name, p)
            sys.exit(1)

    device = _select_device()
    logger.info("Device: %s", device)

    records = load_coco_annotations(coco_json, train_images, max_images=args.max_images)
    if not records:
        logger.error("No valid records found.")
        sys.exit(1)

    train(records, sam2_ckpt, output, device, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
