#!/usr/bin/env python3
"""
finetune_sam2_decoder.py — Fine-tune SAM2 mask decoder on Laboro Tomato.

Architecture decision — freeze encoder, train decoder only:
  SAM2 has three trainable components: image encoder (ViT-L, ~300M params),
  prompt encoder (~few thousand params), mask decoder (~4M params). The image
  encoder is already excellent at encoding agricultural scenes from SA-V
  pretraining. Freezing it reduces memory by ~95% and training time from days
  to hours on CPU. The mask decoder is what maps "box prompt around a tomato
  region" → "tight tomato mask" — exactly what needs to change.

Loss function — BCE + Dice:
  BCE (binary cross-entropy) treats each pixel independently. With a 50×50px
  tomato in a 518×518 frame, 97.7% of pixels are background. BCE alone learns
  "predict all zeros." Dice loss directly optimizes the F1 score of foreground
  pixels, forcing the decoder to get the mask shape right regardless of the
  foreground/background imbalance.

Ground truth masks:
  Laboro Tomato ships YOLO box labels, not pixel masks. We rasterize each YOLO
  box as a filled binary rectangle — this is standard practice for box-supervised
  SAM fine-tuning. The decoder learns to produce masks that fit inside the GT
  boxes; SAM2's architecture then naturally produces tighter masks than the
  rectangular proxy via its upsampling head.

Usage (from repo root):
  # Smoke-test: 10 images, 2 epochs (~2 min on CPU, confirms the loop runs)
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/tools/finetune_sam2_decoder.py \\
    --train-images data/Laboro-Tomato/train/images \\
    --train-labels data/Laboro-Tomato/train/labels \\
    --sam2-checkpoint models/sam2/sam2.1_hiera_small.pt \\
    --output models/sam2/sam2_tomato_finetuned.pt \\
    --epochs 2 --max-images 10

  # Full training: 643 images, 10 epochs (~45–90 min on NucBox CPU)
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/tools/finetune_sam2_decoder.py \\
    --train-images data/Laboro-Tomato/train/images \\
    --train-labels data/Laboro-Tomato/train/labels \\
    --sam2-checkpoint models/sam2/sam2.1_hiera_small.pt \\
    --output models/sam2/sam2_tomato_finetuned.pt \\
    --epochs 10

After training, run eval with the fine-tuned checkpoint:
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/eval/run_eval.py \\
    --val-list data/val_list.txt \\
    --gt-csv data/val_gt.csv \\
    --sam2-checkpoint models/sam2/sam2_tomato_finetuned.pt \\
    --confidence 0.3
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_INPUT_SIZE = 518
_DINO_PATCH_SIZE = 14
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


def _preprocess(bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    """BGR uint8 → RGB float32 518×518 letterboxed.

    Returns:
        rgb_hwc: uint8 HWC for SAM2 set_image()
        pad_x, pad_y: letterbox padding in pixels
    """
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(rgb, (new_w, new_h))
    canvas = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
    pad_x = (_INPUT_SIZE - new_w) // 2
    pad_y = (_INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, pad_x, pad_y, scale


def _yolo_to_518_box(
    cx_n: float, cy_n: float, w_n: float, h_n: float,
    orig_w: int, orig_h: int,
) -> tuple[float, float, float, float] | None:
    """Convert YOLO normalised box to 518×518 letterboxed pixel coords."""
    scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (_INPUT_SIZE - new_w) // 2
    pad_y = (_INPUT_SIZE - new_h) // 2

    cx = cx_n * orig_w * scale + pad_x
    cy = cy_n * orig_h * scale + pad_y
    bw = w_n  * orig_w * scale
    bh = h_n  * orig_h * scale

    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2

    x1 = max(0.0, min(_INPUT_SIZE - 1, x1))
    y1 = max(0.0, min(_INPUT_SIZE - 1, y1))
    x2 = max(0.0, min(_INPUT_SIZE - 1, x2))
    y2 = max(0.0, min(_INPUT_SIZE - 1, y2))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _make_gt_mask(
    x1: float, y1: float, x2: float, y2: float,
) -> np.ndarray:
    """Rasterize a box as a filled binary mask (518×518, bool).

    We use the box itself as the proxy GT mask because Laboro Tomato ships
    YOLO boxes, not pixel masks. The decoder learns to fit masks inside boxes;
    SAM2's upsampling head naturally produces tighter-than-box masks.
    """
    mask = np.zeros((_INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
    ix1, iy1 = int(x1), int(y1)
    ix2, iy2 = min(_INPUT_SIZE, int(x2) + 1), min(_INPUT_SIZE, int(y2) + 1)
    mask[iy1:iy2, ix1:ix2] = 1.0
    return mask


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.

    pred, target: (H, W) tensors, pred is sigmoid-activated logit, target is {0,1}.
    Dice = 2 * |pred ∩ target| / (|pred| + |target|)
    Loss = 1 - Dice
    """
    pred_flat   = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + eps) / (pred_flat.sum() + target_flat.sum() + eps)
    return 1.0 - dice


def _combined_loss(
    logit_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """BCE + Dice loss, weighted 50/50.

    logit_mask: (H, W) raw logit from SAM2 mask decoder (before sigmoid)
    gt_mask:    (H, W) float32 {0, 1}
    """
    bce  = F.binary_cross_entropy_with_logits(logit_mask, gt_mask)
    dice = _dice_loss(torch.sigmoid(logit_mask), gt_mask)
    return bce_weight * bce + (1.0 - bce_weight) * dice


def load_dataset(
    images_dir: Path,
    labels_dir: Path,
    max_images: int = 0,
) -> list[tuple[Path, Path]]:
    """Return list of (image_path, label_path) pairs with valid labels."""
    image_files = sorted(images_dir.glob("*.jpg")) + \
                  sorted(images_dir.glob("*.png"))
    if max_images > 0:
        image_files = image_files[:max_images]

    pairs = []
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
    logger.info("Dataset: %d images with labels.", len(pairs))
    return pairs


def _run_decoder_with_grad(
    sam2_model: "torch.nn.Module",
    image_embed: "torch.Tensor",
    high_res_feats: "list | None",
    box_tensor: "torch.Tensor",
    device: "torch.device",
) -> "torch.Tensor":
    """Call SAM2 prompt encoder + mask decoder directly, keeping gradients.

    Why not use predictor.predict():
      SAM2ImagePredictor.predict() is decorated with @torch.no_grad() and
      returns numpy arrays. Both break the computation graph — torch.from_numpy()
      on the returned logits has no grad_fn, so .backward() fails.

    Instead we call sub-components directly:
      1. sam_prompt_encoder (frozen) — no_grad, encodes the box prompt
      2. sam_mask_decoder (trainable) — gradients flow here

    Args:
        sam2_model: The SAM2 model with frozen encoder, trainable decoder.
        image_embed: Cached image embeddings from set_image() — (1, C, H, W).
        high_res_feats: Optional list of high-res feature maps from the encoder.
        box_tensor: (1, 4) float32 tensor [x1, y1, x2, y2] in 518×518 coords.
        device: Inference device.

    Returns:
        logit: (518, 518) float32 tensor WITH grad_fn attached.
    """
    # Prompt encoder is frozen — no grad needed.
    with torch.no_grad():
        sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
            points=None,
            boxes=box_tensor,   # (1, 4) xyxy
            masks=None,
        )

    # Mask decoder — gradients flow through here.
    # Return signature: (low_res_masks, iou_predictions, sam_tokens_out, obj_score_logits)
    # low_res_masks: (1, num_masks, H/4, W/4)
    decoder_out = sam2_model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    low_res_masks = decoder_out[0]  # (1, 1, H/4, W/4)

    # Upsample to 518×518 — must use bilinear (not nearest) to keep grad_fn.
    logit = F.interpolate(
        low_res_masks.float(),
        size=(_INPUT_SIZE, _INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)  # (518, 518) WITH grad_fn

    return logit


def train(
    pairs: list[tuple[Path, Path]],
    sam2_ckpt: Path,
    output_path: Path,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
) -> None:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

    logger.info("Loading SAM2 from %s...", sam2_ckpt)
    sam2_model = build_sam2(_SAM2_CFG, str(sam2_ckpt), device=device)

    # Freeze image encoder and prompt encoder — only train mask decoder.
    for name, param in sam2_model.named_parameters():
        if "image_encoder" in name or "sam_prompt_encoder" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    trainable = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in sam2_model.parameters())
    logger.info(
        "Parameters: %d trainable / %d total (%.1f%% frozen).",
        trainable, total, 100.0 * (1 - trainable / total),
    )

    # predictor.set_image() handles image preprocessing and encoder forward pass.
    # We use it only to cache image embeddings — then call the decoder directly.
    predictor = SAM2ImagePredictor(sam2_model)

    optimizer = torch.optim.AdamW(
        [p for p in sam2_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    # Cosine decay to lr/10 over the full training run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(pairs), eta_min=lr / 10
    )

    best_loss = float("inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        random.shuffle(pairs)

        for img_path, label_path in pairs:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            orig_h, orig_w = bgr.shape[:2]
            rgb_hwc, pad_x, pad_y, scale = _preprocess(bgr)

            with open(label_path) as f:
                lines = [l.strip().split() for l in f if len(l.strip().split()) >= 5]
            if not lines:
                continue

            # Image encoder forward pass — frozen, no grad needed.
            with torch.no_grad():
                predictor.set_image(rgb_hwc)

            # Cache image embeddings for reuse across all boxes in this image.
            image_embed   = predictor._features["image_embed"]      # (1, C, H, W)
            high_res_feats = predictor._features.get("high_res_feats")  # list or None

            image_loss = torch.tensor(0.0, device=device)
            n_boxes = 0

            for parts in lines:
                _, cx_n, cy_n, w_n, h_n = (float(p) for p in parts[:5])
                box = _yolo_to_518_box(cx_n, cy_n, w_n, h_n, orig_w, orig_h)
                if box is None:
                    continue

                x1, y1, x2, y2 = box
                gt_mask_np = _make_gt_mask(x1, y1, x2, y2)
                gt_mask = torch.from_numpy(gt_mask_np).to(device)

                # (1, 4) box tensor — prompt encoder expects xyxy float
                box_tensor = torch.tensor(
                    [[x1, y1, x2, y2]], dtype=torch.float32, device=device
                )

                # Decoder forward with grad — this is where training happens.
                logit = _run_decoder_with_grad(
                    sam2_model, image_embed, high_res_feats, box_tensor, device
                )

                loss = _combined_loss(logit, gt_mask)
                image_loss = image_loss + loss
                n_boxes += 1

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
            n_batches += 1

        if n_batches == 0:
            logger.warning("Epoch %d: no batches processed.", epoch)
            continue

        mean_loss = epoch_loss / n_batches
        logger.info("Epoch %d/%d — loss: %.4f", epoch, epochs, mean_loss)

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(sam2_model.state_dict(), str(output_path))
            logger.info("  Saved best checkpoint (loss=%.4f) to %s.", best_loss, output_path)

    logger.info("Training complete. Best loss: %.4f", best_loss)
    logger.info("Fine-tuned checkpoint: %s", output_path)
    logger.info("")
    logger.info("Run eval with fine-tuned weights:")
    logger.info(
        "  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES=\"\" PYTHONPATH=perception "
        "python3 perception/eval/run_eval.py "
        "--val-list data/val_list.txt "
        "--gt-csv data/val_gt.csv "
        "--confidence 0.3"
    )
    logger.info(
        "  (No --sam2-checkpoint needed: detector auto-loads %s when present.)",
        output_path.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM2 mask decoder on Laboro Tomato bounding boxes."
    )
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument(
        "--sam2-checkpoint", type=Path,
        default=Path("models/sam2/sam2.1_hiera_small.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("models/sam2/sam2_tomato_finetuned.pt"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Limit images (0=all). Use 10 for smoke-test.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else repo_root / p

    train_images = _abs(args.train_images)
    train_labels = _abs(args.train_labels)
    sam2_ckpt    = _abs(args.sam2_checkpoint)
    output       = _abs(args.output)

    for p, name in [(train_images, "train-images"), (train_labels, "train-labels"),
                    (sam2_ckpt, "sam2-checkpoint")]:
        if not p.exists():
            logger.error("%s not found: %s", name, p)
            sys.exit(1)

    device = _select_device()
    logger.info("Device: %s", device)

    pairs = load_dataset(train_images, train_labels, max_images=args.max_images)
    if not pairs:
        logger.error("No valid image/label pairs found.")
        sys.exit(1)

    train(pairs, sam2_ckpt, output, device, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
