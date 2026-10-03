#!/usr/bin/env python3
"""
build_query_embedding.py — Build a DINOv2 tomato query embedding from labeled training images.

Replaces the hardcoded red/orange RGB prior in dino_sam2_detector.py with a
mean patch embedding computed from actual tomato pixels in the Laboro Tomato
training set. The resulting vector is saved to models/query_embedding.pt and
loaded automatically by DINOv2SAM2Detector at startup.

Why this matters:
  The RGB prior fires on red/orange patches. Laboro Tomato has green and yellow
  tomatoes, variable lighting, and cluttered backgrounds — the prior misses almost
  everything. A data-driven embedding encodes shape, texture, and context from
  real tomato images, not just colour.

Usage (from repo root):
  # [MAC] host (uses MPS)
  PYTHONPATH=perception python3 perception/tools/build_query_embedding.py \
    --train-images data/Laboro-Tomato/train/images \
    --train-labels data/Laboro-Tomato/train/labels \
    --output models/query_embedding.pt

  # [NUCBOX] inside ROCm container
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
    python3 perception/tools/build_query_embedding.py \
    --train-images data/Laboro-Tomato/train/images \
    --train-labels data/Laboro-Tomato/train/labels \
    --output models/query_embedding.pt

Runtime: ~3–8 min on CPU for 643 images (DINOv2 forward pass per image).
Output:  models/query_embedding.pt — float32 tensor of shape (768,), L2-normalised.
         models/negative_embedding.pt — (optional) patches outside GT boxes, for contrastive scoring.
"""

from __future__ import annotations

import argparse
from typing import Optional
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


def _preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC → float32 CHW tensor, ImageNet-normalised, resized to 518×518."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE)).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)))


def _yolo_to_pixel(cx: float, cy: float, w: float, h: float,
                   img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """YOLO normalised cx,cy,w,h → pixel x1,y1,x2,y2."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (
        max(0, x1), max(0, y1),
        min(img_w - 1, x2), min(img_h - 1, y2),
    )


def _boxes_to_patch_mask(
    boxes_pixel: list[tuple[int, int, int, int]],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Return a (37,37) boolean mask of DINOv2 patch grid cells that overlap any GT box.

    The 518×518 input is letterboxed from (orig_h, orig_w). Each patch covers
    14×14 pixels in the 518×518 space. We project GT boxes into that space and
    mark all overlapping patch cells.
    """
    scale = min(_DINO_INPUT_SIZE / orig_w, _DINO_INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (_DINO_INPUT_SIZE - new_w) // 2
    pad_y = (_DINO_INPUT_SIZE - new_h) // 2

    mask = np.zeros((_DINO_GRID, _DINO_GRID), dtype=bool)

    for x1, y1, x2, y2 in boxes_pixel:
        # Scale box into 518×518 letterboxed space
        sx1 = int(x1 * scale) + pad_x
        sy1 = int(y1 * scale) + pad_y
        sx2 = int(x2 * scale) + pad_x
        sy2 = int(y2 * scale) + pad_y

        # Convert to patch grid coordinates
        gx1 = max(0, sx1 // _DINO_PATCH_SIZE)
        gy1 = max(0, sy1 // _DINO_PATCH_SIZE)
        gx2 = min(_DINO_GRID - 1, sx2 // _DINO_PATCH_SIZE)
        gy2 = min(_DINO_GRID - 1, sy2 // _DINO_PATCH_SIZE)

        if gx2 >= gx1 and gy2 >= gy1:
            mask[gy1:gy2 + 1, gx1:gx2 + 1] = True

    return mask


def _kmeans_prototypes(patches: torch.Tensor, k: int, n_iter: int = 100) -> torch.Tensor:
    """Compute k L2-normalised prototype vectors from patch embeddings via k-means.

    Using k-means rather than the global mean because the tomato appearance
    distribution in Laboro is multi-modal: green unripe, yellow transitional,
    red ripe, and occluded/partial each occupy distinct regions of DINOv2 space.
    The global mean sits equidistant from all modes and represents none well.

    Returns: (k, D) float32 tensor, each row L2-normalised.
    """
    n, d = patches.shape
    if n <= k:
        # Pad with zeros if fewer patches than prototypes (degenerate case)
        padded = torch.zeros(k, d)
        padded[:n] = patches
        return F.normalize(padded, dim=1)

    # Initialise centroids with k-means++ seeding for stable convergence
    idx = torch.randint(0, n, (1,)).item()
    centroids = patches[idx].unsqueeze(0)  # (1, D)
    for _ in range(k - 1):
        # Distance of each patch to the nearest existing centroid
        dists = 1.0 - (patches @ centroids.T)  # (N, c); cosine distance
        min_dists = dists.min(dim=1).values     # (N,)
        min_dists = min_dists.clamp(min=0.0)
        probs = min_dists / min_dists.sum()
        new_idx = torch.multinomial(probs, 1).item()
        centroids = torch.cat([centroids, patches[new_idx].unsqueeze(0)], dim=0)

    centroids = F.normalize(centroids, dim=1)  # (k, D)

    for _ in range(n_iter):
        # Assignment: nearest centroid (cosine similarity)
        sims = patches @ centroids.T  # (N, k)
        assignments = sims.argmax(dim=1)  # (N,)
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            members = patches[assignments == j]
            if members.shape[0] > 0:
                new_centroids[j] = members.mean(dim=0)
            else:
                # Dead cluster: re-seed from farthest point
                farthest = (patches @ centroids.T).min(dim=1).values.argmax()
                new_centroids[j] = patches[farthest]
        new_centroids = F.normalize(new_centroids, dim=1)
        if (new_centroids - centroids).norm() < 1e-5:
            break
        centroids = new_centroids

    return centroids  # (k, D)


def _load_dino_with_lora(device: torch.device, lora_path: Optional[Path] = None):
    """Load DINOv2, optionally applying saved LoRA adapter weights.

    LoRA weights are adapter-key-only state dicts produced by finetune_dino_lora.py.
    We reconstruct the LoRA modules by importing the same inject_lora function
    so the adapter architecture is guaranteed to match.
    """
    dino = torch.hub.load("facebookresearch/dinov2", _DINO_MODEL_NAME, pretrained=True)
    if lora_path is not None and lora_path.exists():
        try:
            from perception.tools.finetune_dino_lora import inject_lora
        except ImportError:
            # Fallback path when running from repo root
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from finetune_dino_lora import inject_lora  # type: ignore[import]

        # Default to rank=8, lora_blocks=4 matching the training defaults.
        # If you trained with different values, pass them via CLI in the future.
        dino = inject_lora(dino, rank=8, lora_blocks=4)
        lora_state = torch.load(str(lora_path), map_location=device)
        missing, unexpected = dino.load_state_dict(lora_state, strict=False)
        logger.info("Loaded LoRA weights from %s (%d keys).", lora_path, len(lora_state))
        if unexpected:
            logger.warning("Unexpected LoRA keys: %s", unexpected[:5])

    dino.eval().to(device)
    return dino


def build_embedding(
    train_images_dir: Path,
    train_labels_dir: Path,
    output_path: Path,
    output_negative_path: Optional[Path] = None,
    max_images: int = 0,
    num_prototypes: int = 1,
    lora_path: Optional[Path] = None,
) -> None:
    device = _select_device()
    logger.info("Device: %s", device)

    logger.info("Loading DINOv2 (%s) via torch.hub...", _DINO_MODEL_NAME)
    dino = _load_dino_with_lora(device, lora_path=lora_path)
    logger.info("DINOv2 loaded.")

    image_files = sorted(train_images_dir.glob("*.jpg")) + \
                  sorted(train_images_dir.glob("*.png"))
    if max_images > 0:
        image_files = image_files[:max_images]

    logger.info("Found %d training images.", len(image_files))

    all_patch_embeddings: list[torch.Tensor] = []
    all_negative_embeddings: list[torch.Tensor] = []
    skipped = 0

    for i, img_path in enumerate(image_files):
        label_path = train_labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            skipped += 1
            continue

        orig_h, orig_w = bgr.shape[:2]

        # Parse YOLO labels: class cx cy w h
        boxes_pixel = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = (float(p) for p in parts[:5])
                boxes_pixel.append(_yolo_to_pixel(cx, cy, bw, bh, orig_w, orig_h))

        if not boxes_pixel:
            skipped += 1
            continue

        patch_mask = _boxes_to_patch_mask(boxes_pixel, orig_w, orig_h)
        if not patch_mask.any():
            skipped += 1
            continue

        tensor = _preprocess(bgr).unsqueeze(0).to(device)

        with torch.no_grad():
            features = dino.forward_features(tensor)

        # patch_tokens: (1, 1369, 768) → (37, 37, 768)
        patch_tokens = features["x_norm_patchtokens"].squeeze(0)  # (1369, 768)
        patch_grid = patch_tokens.reshape(_DINO_GRID, _DINO_GRID, -1)  # (37, 37, 768)

        # Extract only the patches that overlap GT boxes
        tomato_patches = patch_grid[patch_mask]  # (N, 768)
        if tomato_patches.shape[0] == 0:
            skipped += 1
            continue

        all_patch_embeddings.append(tomato_patches.cpu())

        # Negative: patches outside GT boxes (leaves, stems, background)
        if output_negative_path is not None:
            negative_mask = ~patch_mask
            if negative_mask.any():
                negative_patches = patch_grid[negative_mask]  # (M, 768)
                all_negative_embeddings.append(negative_patches.cpu())

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d images, collected %d patch vectors so far.",
                        i + 1, len(image_files),
                        sum(t.shape[0] for t in all_patch_embeddings))

    if not all_patch_embeddings:
        logger.error("No patch embeddings collected. Check --train-images and --train-labels paths.")
        sys.exit(1)

    logger.info("Skipped %d images (no label / unreadable / empty box).", skipped)

    all_patches = torch.cat(all_patch_embeddings, dim=0)  # (Total_patches, 768)
    logger.info("Total tomato patch vectors: %d", all_patches.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if num_prototypes > 1:
        logger.info("Running k-means (k=%d) on tomato patches...", num_prototypes)
        # L2-normalise patches before k-means so cosine distance == Euclidean distance
        patches_normed = F.normalize(all_patches, dim=1)
        query = _kmeans_prototypes(patches_normed, k=num_prototypes)  # (k, 768)
        torch.save(query, str(output_path))
        logger.info("Saved %d-prototype query embedding to %s (shape=%s)", num_prototypes, output_path, query.shape)
    else:
        mean_embedding = all_patches.mean(dim=0)  # (768,)
        query = F.normalize(mean_embedding, dim=0)  # L2-normalised
        torch.save(query, str(output_path))
        logger.info("Saved query embedding to %s (shape=%s)", output_path, query.shape)

    if output_negative_path is not None and all_negative_embeddings:
        neg_patches = torch.cat(all_negative_embeddings, dim=0)
        total_neg = neg_patches.shape[0]
        if total_neg > 100_000:
            idx = torch.randperm(total_neg)[:100_000]
            neg_patches = neg_patches[idx]
            logger.info("Subsampled to 100k negative patches (from %d)", total_neg)
        logger.info("Total negative patch vectors: %d", neg_patches.shape[0])
        neg_mean = neg_patches.mean(dim=0)
        neg_embedding = F.normalize(neg_mean, dim=0)
        output_negative_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(neg_embedding, str(output_negative_path))
        logger.info("Saved negative embedding to %s (contrastive scoring)", output_negative_path)
    elif output_negative_path is not None:
        logger.warning("No negative patches collected. Skipping negative embedding.")

    logger.info("Done. Re-run eval to get detections:")
    logger.info("  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES=\"\" PYTHONPATH=perception "
                "python3 perception/eval/run_eval.py --val-list data/val_list.txt --confidence 0.2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DINOv2 tomato query embedding from Laboro Tomato training set."
    )
    parser.add_argument(
        "--train-images", type=Path, required=True,
        help="Path to Laboro Tomato train/images/ directory.",
    )
    parser.add_argument(
        "--train-labels", type=Path, required=True,
        help="Path to Laboro Tomato train/labels/ directory (YOLO format .txt files).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("models/query_embedding.pt"),
        help="Output path for the query embedding tensor (default: models/query_embedding.pt).",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Limit number of images processed (0 = all). Useful for a quick smoke-test.",
    )
    parser.add_argument(
        "--output-negative", type=Path, default=None,
        help="Also build negative embedding from patches outside GT boxes (for contrastive scoring).",
    )
    parser.add_argument(
        "--dino-lora-path", type=Path, default=None,
        help=(
            "Optional path to LoRA adapter weights produced by finetune_dino_lora.py. "
            "When provided, DINOv2 is loaded with LoRA adapters applied before building "
            "the query embedding. Produces a LoRA-adapted embedding for use at eval time "
            "with --dino-lora-path on run_eval.py."
        ),
    )
    parser.add_argument(
        "--num-prototypes", type=int, default=1,
        help=(
            "Number of k-means prototype vectors to build (default 1 = single mean). "
            "Use 4 to separate green/yellow/red/occluded ripeness modes. "
            "Saved as a (k, 768) tensor; detector uses max-over-k scoring."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)

    train_images = args.train_images if args.train_images.is_absolute() \
        else repo_root / args.train_images
    train_labels = args.train_labels if args.train_labels.is_absolute() \
        else repo_root / args.train_labels
    output = args.output if args.output.is_absolute() else repo_root / args.output

    if not train_images.exists():
        logger.error("train-images dir not found: %s", train_images)
        sys.exit(1)
    if not train_labels.exists():
        logger.error("train-labels dir not found: %s", train_labels)
        sys.exit(1)

    output_neg = None
    if args.output_negative:
        output_neg = args.output_negative if args.output_negative.is_absolute() \
            else repo_root / args.output_negative

    lora_path = None
    if args.dino_lora_path:
        lora_path = args.dino_lora_path if args.dino_lora_path.is_absolute() \
            else repo_root / args.dino_lora_path

    build_embedding(
        train_images, train_labels, output,
        output_negative_path=output_neg,
        max_images=args.max_images,
        num_prototypes=args.num_prototypes,
        lora_path=lora_path,
    )


if __name__ == "__main__":
    main()
