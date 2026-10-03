#!/usr/bin/env python3
"""
build_text_prototypes.py — Strict label-free DINOv2-space prototype builder (Phase 4.1).

Why this exists for the paper's stretch claim:
  build_query_embedding.py builds the k=4 prototype from train YOLO bbox
  labels — supervised. For the strict label-free table row, we need a
  prototype that uses ZERO Laboro labels. SigLIP gives us a free
  "is-tomato-yes/no" oracle via text-image contrastive alignment. We use it
  to find tomato-like regions in unlabeled train images, then average their
  DINOv2 patch features into the prototype the existing detector expects.

Algorithm:
  For each unlabeled train image:
    1. Run SAM2 AMG to get mask proposals (no fine-tune; vanilla checkpoint).
    2. For each mask: crop bbox -> SigLIP image encode -> cosine to text prompt
       ("a photograph of a ripe tomato"). Top-K across all train images become
       the "pseudo-positive" mask set.
    3. For each pseudo-positive mask: DINOv2 forward on the parent image,
       extract patch tokens weighted by mask coverage, mean-pool -> one
       DINOv2-space vector per pseudo-positive mask.
    4. k-means (k=--num-prototypes) on the collected vectors -> centroids.
    5. Save (k, 768) tensor to --output. Drop-in replacement for
       models/query_embedding_k4.pt.

This touches NO Laboro label files. The only "supervision" is the text prompt
"tomato" — which is the entire point of zero-shot transfer.

Sprint 4: Phase 4.1.

Usage [NUCBOX, ~2h CPU for 643 images]:
  AGROBOT_FORCE_CPU=1 PYTHONPATH=perception \\
    python3 perception/tools/build_text_prototypes.py \\
    --train-images data/Laboro-Tomato/train/images \\
    --output models/query_embedding_text_k4.pt \\
    --output-negative models/negative_embedding_text.pt \\
    --num-prototypes 4 --top-k 1500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14
_DINO_INPUT_SIZE = 518
_DINO_GRID = _DINO_INPUT_SIZE // _DINO_PATCH_SIZE
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def _select_device() -> torch.device:
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _preprocess_for_dino(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE)).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)))


def _mask_to_coverage(seg: np.ndarray) -> np.ndarray:
    seg_f = seg[:_DINO_GRID * _DINO_PATCH_SIZE, :_DINO_GRID * _DINO_PATCH_SIZE].astype(np.float32)
    blocks = seg_f.reshape(_DINO_GRID, _DINO_PATCH_SIZE, _DINO_GRID, _DINO_PATCH_SIZE)
    return blocks.mean(axis=(1, 3))


def _kmeans(patches: torch.Tensor, k: int, n_iter: int = 100) -> torch.Tensor:
    """Same cosine-space k-means++ as build_query_embedding.py for compatibility."""
    n, d = patches.shape
    if n <= k:
        padded = torch.zeros(k, d)
        padded[:n] = patches
        return F.normalize(padded, dim=1)
    idx = torch.randint(0, n, (1,)).item()
    centroids = patches[idx].unsqueeze(0)
    for _ in range(k - 1):
        dists = 1.0 - (patches @ centroids.T)
        min_d = dists.min(dim=1).values.clamp(min=0.0)
        probs = min_d / min_d.sum()
        new_idx = torch.multinomial(probs, 1).item()
        centroids = torch.cat([centroids, patches[new_idx].unsqueeze(0)], dim=0)
    centroids = F.normalize(centroids, dim=1)
    for _ in range(n_iter):
        sims = patches @ centroids.T
        assignments = sims.argmax(dim=1)
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            members = patches[assignments == j]
            if members.shape[0] > 0:
                new_centroids[j] = members.mean(dim=0)
            else:
                farthest = (patches @ centroids.T).min(dim=1).values.argmax()
                new_centroids[j] = patches[farthest]
        new_centroids = F.normalize(new_centroids, dim=1)
        if (new_centroids - centroids).norm() < 1e-5:
            break
        centroids = new_centroids
    return centroids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output (k, 768) DINOv2-space prototype tensor.")
    parser.add_argument("--output-negative", type=Path, default=None,
                        help="Optional: output background prototype "
                             "(mean DINOv2 feature of the LOWEST-scoring SigLIP masks).")
    parser.add_argument("--num-prototypes", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1500,
                        help="How many highest-SigLIP-scoring masks across the dataset "
                             "to use as positive samples for k-means.")
    parser.add_argument("--bottom-k", type=int, default=2000,
                        help="How many lowest-SigLIP-scoring masks to use for "
                             "the negative prototype (background).")
    parser.add_argument("--amg-points", type=int, default=20)
    parser.add_argument("--positive-prompt", type=str,
                        default="a photograph of a ripe tomato on a vine")
    parser.add_argument("--siglip-model", type=str,
                        default="google/siglip-base-patch16-224")
    parser.add_argument("--sam2-checkpoint", type=Path,
                        default=Path("models/sam2/sam2.1_hiera_small.pt"),
                        help="Vanilla SAM2 — do NOT pass the polygon-fine-tuned weights "
                             "for the strict label-free regime.")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)
    train_images = args.train_images if args.train_images.is_absolute() \
        else repo_root / args.train_images
    sam2_ckpt = args.sam2_checkpoint if args.sam2_checkpoint.is_absolute() \
        else repo_root / args.sam2_checkpoint

    if not train_images.exists():
        logger.error("train-images not found: %s", train_images); sys.exit(1)
    if not sam2_ckpt.exists():
        logger.error("sam2-checkpoint not found: %s", sam2_ckpt); sys.exit(1)

    device = _select_device()
    logger.info("Device: %s", device)

    logger.info("Loading DINOv2 (%s)...", _DINO_MODEL_NAME)
    dino = torch.hub.load("facebookresearch/dinov2", _DINO_MODEL_NAME,
                          pretrained=True).eval().to(device)

    logger.info("Loading SAM2 from %s (vanilla, NO polygon fine-tune)...", sam2_ckpt)
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    sam2_model = build_sam2(_DEFAULT_SAM2_CFG, str(sam2_ckpt), device=device)
    amg = SAM2AutomaticMaskGenerator(
        model=sam2_model, points_per_side=args.amg_points,
        pred_iou_thresh=0.70, stability_score_thresh=0.80,
        crop_n_layers=0, min_mask_region_area=100,
    )

    logger.info("Loading SigLIP (%s)...", args.siglip_model)
    from transformers import AutoModel, AutoProcessor
    siglip_proc = AutoProcessor.from_pretrained(args.siglip_model)
    siglip = AutoModel.from_pretrained(args.siglip_model).to(device).eval()

    def _as_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
            return x.last_hidden_state[:, 0]
        raise TypeError(f"Unexpected SigLIP output type {type(x).__name__}")

    with torch.no_grad():
        text_in = siglip_proc(
            text=[args.positive_prompt], padding="max_length", return_tensors="pt",
        )
        text_in = {k: v.to(device) for k, v in text_in.items()}
        text_emb = F.normalize(_as_tensor(siglip.get_text_features(**text_in)), dim=1)

    image_files = sorted(train_images.glob("*.jpg")) + sorted(train_images.glob("*.png"))
    if args.max_images > 0:
        image_files = image_files[: args.max_images]
    logger.info("Found %d training images.", len(image_files))

    # Pass 1: collect (image_idx, mask, siglip_score) for ALL masks across the dataset.
    # We hold patch_norms in memory only when we re-process the top/bottom-K images.
    candidates: list[tuple[int, np.ndarray, float]] = []
    for i, img_path in enumerate(image_files):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE))
        masks_data = amg.generate(rgb_resized)
        if not masks_data:
            continue

        # SigLIP-score every mask in this image in one batch.
        crops = []
        valid = []
        for j, m in enumerate(masks_data):
            seg = m["segmentation"]
            x, y, w, h = m["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(_DINO_INPUT_SIZE, int(x + w)), min(_DINO_INPUT_SIZE, int(y + h))
            if (x2 - x1) < 8 or (y2 - y1) < 8 or seg.sum() < 50:
                continue
            crops.append(Image.fromarray(rgb_resized[y1:y2, x1:x2]))
            valid.append(j)
        if not crops:
            continue

        with torch.no_grad():
            img_in = siglip_proc(images=crops, return_tensors="pt")
            img_in = {k: v.to(device) for k, v in img_in.items()}
            img_emb = F.normalize(_as_tensor(siglip.get_image_features(**img_in)), dim=1)
        sims = (img_emb @ text_emb.T).squeeze(1).cpu().numpy()
        for j, s in zip(valid, sims):
            candidates.append((i, masks_data[j]["segmentation"], float(s)))

        if (i + 1) % 25 == 0:
            logger.info("  pass1 [%d/%d] candidates=%d", i + 1, len(image_files), len(candidates))

    if not candidates:
        logger.error("No candidate masks. Increase --amg-points or check inputs.")
        sys.exit(1)

    candidates.sort(key=lambda c: c[2], reverse=True)
    pos = candidates[: args.top_k]
    neg = candidates[-args.bottom_k:] if args.output_negative else []
    logger.info("pass1 done: %d total candidates. Pos top-%d siglip range %.3f..%.3f",
                len(candidates), len(pos), pos[-1][2], pos[0][2])

    # Pass 2: for the top-K positives (and bottom-K negatives), recompute DINOv2
    # patches and accumulate coverage-weighted mean per mask.
    pos_vectors: list[torch.Tensor] = []
    neg_vectors: list[torch.Tensor] = []

    by_image: dict[int, list[tuple[np.ndarray, str]]] = {}
    for img_idx, seg, _ in pos:
        by_image.setdefault(img_idx, []).append((seg, "pos"))
    for img_idx, seg, _ in neg:
        by_image.setdefault(img_idx, []).append((seg, "neg"))

    for k_idx, (img_idx, items) in enumerate(by_image.items()):
        img_path = image_files[img_idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        tensor = _preprocess_for_dino(bgr).unsqueeze(0).to(device)
        with torch.no_grad():
            features = dino.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"].squeeze(0)  # (1369, D)
        for seg, kind in items:
            cov = _mask_to_coverage(seg).reshape(-1)
            if cov.sum() < 1e-6:
                continue
            cov_t = torch.from_numpy(cov).to(device)
            v = (patch_tokens * cov_t.unsqueeze(1)).sum(dim=0) / cov_t.sum()
            v = F.normalize(v, dim=0).cpu()
            (pos_vectors if kind == "pos" else neg_vectors).append(v)

        if (k_idx + 1) % 25 == 0:
            logger.info("  pass2 [%d/%d images]", k_idx + 1, len(by_image))

    out = args.output if args.output.is_absolute() else repo_root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    if pos_vectors:
        pos_t = torch.stack(pos_vectors)
        pos_t = F.normalize(pos_t, dim=1)
        prototype = _kmeans(pos_t, k=args.num_prototypes)
        torch.save(prototype, str(out))
        logger.info("Saved positive prototype to %s (shape=%s)", out, prototype.shape)
    else:
        logger.error("No positive vectors collected.")
        sys.exit(1)

    if args.output_negative and neg_vectors:
        neg_out = args.output_negative if args.output_negative.is_absolute() \
            else repo_root / args.output_negative
        neg_out.parent.mkdir(parents=True, exist_ok=True)
        neg_t = torch.stack(neg_vectors)
        neg_t = F.normalize(neg_t, dim=1)
        neg_proto = F.normalize(neg_t.mean(dim=0), dim=0)
        torch.save(neg_proto, str(neg_out))
        logger.info("Saved negative prototype to %s (shape=%s)", neg_out, neg_proto.shape)


if __name__ == "__main__":
    main()
