#!/usr/bin/env python3
"""
finetune_dino_mc_lora.py — Mask-Conditioned LoRA fine-tuning of DINOv2 (Phase 3.1).

Why this replaces perception/tools/finetune_dino_lora.py (which collapsed at
mAP=0.035):
  The original file had three independent defects that each materially hurt
  training. We fix all three here so the contribution can be ablated cleanly
  in the paper:

  Defect 1 (FIXED here): "Trivial positive" — the original used feature
    dropout on the same anchor as the "augmented view". That is not a SimCLR
    view; it is noise. We use real image-level augmentations (color jitter,
    gaussian blur, optional horizontal flip) and treat the patch tokens at the
    *same spatial location* under two augmentations as the positive pair.

  Defect 2 (FIXED here): "Box-positive supervision is noisy" — the original
    used axis-aligned bounding boxes as the per-patch positive mask, so for
    round tomatoes ~30% of "positive" patches were actually leaf/stem. We
    rasterize the COCO polygon segmentations from train.json, downsample to
    the 37x37 DINOv2 patch grid as float coverage weights, and weight every
    contrastive term by the patch's coverage. Background patches inside the
    box no longer contribute as positives.

  Defect 3 (FIXED here): "NT-Xent denominator double-counts the positive"
    `loss = pos_sim - logsumexp(cat([pos_sim], all_sims))` had pos_sim already
    in all_sims (only self-similarity at index i was masked, but the positive
    at index N+i was not). Standard NT-Xent puts the positive in the denominator
    exactly once. We use the canonical SimCLR formulation here.

  Defect 4 (FIXED here): "Per-image batching" — the original gradient step
    saw negatives only from the same image. Cross-image hard negatives are
    essential because intra-image background patches are easy negatives that
    the model already separates well. We accumulate patches from N=8 images
    per step so negatives include patches from leaves on other plants, soil
    in other backgrounds, etc.

Loss formulation (canonical SimCLR / NT-Xent, mask-weighted):

  For each image i in the batch:
    coverage_i  : (37,37) float in [0,1]   -- fraction of each patch inside
                                             the polygon mask
    view_A_i    : (1369, D) DINOv2 patch tokens of augmented view A
    view_B_i    : (1369, D) DINOv2 patch tokens of augmented view B
                  (A and B differ in color/blur; horizontal flip is applied
                  symmetrically so positions still correspond when we undo it)

  For each spatial position p:
    anchor   = view_A_i[p]   (L2-normalised)
    positive = view_B_i[p_after_flip]
    weight   = coverage_i[p]

  Negatives for anchor (i, p) are:
    - positives from all other images j != i (cross-image hard negatives)
    - positives from same image at OTHER positions q != p (intra-image hard negs)
    - explicit background-sampled tokens from coverage<<1 (clear-background hard negs)

  Loss per anchor:
    L(i,p) = -weight * log( exp(s_pos/tau) / (exp(s_pos/tau) + sum exp(s_neg/tau)) )
  Total loss = sum / sum(weights). The weighting nullifies background-anchor
  contributions and emphasises confident-tomato positions.

Sprint 4: Phase 3.1.

Usage [NUCBOX CPU, ~36h for 643 images, 8 epochs]:
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/tools/finetune_dino_mc_lora.py \\
    --coco-json data/Laboro-Tomato/annotations/train.json \\
    --train-images data/Laboro-Tomato/train/images \\
    --output models/dino_mc_lora.pt \\
    --epochs 8 --rank 8 --lora-blocks 4 \\
    --batch-size 8 --pos-per-image 16 --neg-per-image 16

After training, rebuild the query embedding with the LoRA backbone:
  PYTHONPATH=perception python3 perception/tools/build_query_embedding.py \\
    --train-images data/Laboro-Tomato/train/images \\
    --train-labels data/Laboro-Tomato/train/labels \\
    --output models/query_embedding_mc_lora_k4.pt \\
    --num-prototypes 4 \\
    --output-negative models/negative_embedding_mc_lora.pt \\
    --dino-lora-path models/dino_mc_lora.pt

Then sweep --confidence in {0.10, 0.15, 0.20, 0.25, 0.30, 0.35} via the
post-filter sweep tool. Cosine score scale shifts after adaptation; the same
numerical threshold from S4.12 is meaningless.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14
_DINO_INPUT_SIZE = 518
_DINO_GRID = _DINO_INPUT_SIZE // _DINO_PATCH_SIZE  # 37
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _select_device() -> torch.device:
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── LoRA injection — reused from finetune_dino_lora for adapter compatibility ─

def _inject_lora_qkv(model: nn.Module, rank: int, lora_blocks: int) -> nn.Module:
    """Same module surgery as finetune_dino_lora.inject_lora — kept here so the
    adapter weights produced by either trainer load into the same layers and
    are interchangeable for ablations.

    Wraps the fused qkv Linear and the proj Linear in the last `lora_blocks`
    DINOv2 transformer blocks. Freezes everything else.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from finetune_dino_lora import LoRALinear  # adapter module

    for p in model.parameters():
        p.requires_grad_(False)
    blocks = list(model.blocks)[-lora_blocks:]
    n = 0
    for blk in blocks:
        if hasattr(blk, "attn") and isinstance(blk.attn.qkv, nn.Linear):
            blk.attn.qkv = LoRALinear(blk.attn.qkv, rank=rank, alpha=float(rank * 2))
            n += 1
        if hasattr(blk, "attn") and hasattr(blk.attn, "proj") and isinstance(blk.attn.proj, nn.Linear):
            blk.attn.proj = LoRALinear(blk.attn.proj, rank=rank, alpha=float(rank * 2))
            n += 1
    logger.info("Injected LoRA into %d Linear layers (rank=%d, last %d blocks).", n, rank, lora_blocks)
    return model


# ── Polygon -> letterboxed coverage ───────────────────────────────────────────

def _letterbox(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = min(_DINO_INPUT_SIZE / orig_w, _DINO_INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(rgb, (new_w, new_h))
    canvas = np.zeros((_DINO_INPUT_SIZE, _DINO_INPUT_SIZE, 3), dtype=np.uint8)
    pad_x = (_DINO_INPUT_SIZE - new_w) // 2
    pad_y = (_DINO_INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _polygons_to_coverage(
    polygons: list[list[float]],
    orig_w: int, orig_h: int,
    scale: float, pad_x: int, pad_y: int,
) -> np.ndarray:
    """Rasterize union of COCO polygons into 518x518 mask, then average-pool to
    the 37x37 patch coverage grid (float fraction)."""
    mask518 = np.zeros((_DINO_INPUT_SIZE, _DINO_INPUT_SIZE), dtype=np.float32)
    for poly in polygons:
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
        pts[:, 0] = pts[:, 0] * scale + pad_x
        pts[:, 1] = pts[:, 1] * scale + pad_y
        pts = np.clip(pts, 0, _DINO_INPUT_SIZE - 1).astype(np.int32)
        cv2.fillPoly(mask518, [pts], 1.0)
    blocks = mask518.reshape(_DINO_GRID, _DINO_PATCH_SIZE, _DINO_GRID, _DINO_PATCH_SIZE)
    return blocks.mean(axis=(1, 3))  # (37, 37)


# ── Real image augmentations ─────────────────────────────────────────────────

def _augment_image(rgb: np.ndarray, rng: np.random.Generator,
                   flip: bool) -> np.ndarray:
    """Apply color jitter + gaussian blur + optional horizontal flip.

    Not affine/crop — those would break the spatial-correspondence assumption
    that view_A patch (gy, gx) and view_B patch (gy, gx_after_flip) point at
    the same physical region.
    """
    img = rgb.astype(np.float32)

    # Color jitter: small per-channel multiplicative + additive shift.
    bright = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.85, 1.15)
    img = (img - 128.0) * contrast + 128.0
    img = img * bright
    # Saturation: convert to HSV
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = rng.uniform(0.85, 1.15)
    hue = rng.uniform(-5, 5)
    hsv[..., 0] = (hsv[..., 0] + hue) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    # Gaussian blur with small probability.
    if rng.random() < 0.4:
        ksize = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    if flip:
        img = img[:, ::-1, :].copy()

    return np.clip(img, 0, 255).astype(np.uint8)


def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 RGB HWC -> float32 CHW ImageNet-normalised."""
    f = rgb.astype(np.float32) / 255.0
    f = (f - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(np.transpose(f, (2, 0, 1)))


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_records_one(coco_json: Path, images_dir: Path) -> list[dict]:
    """Load records from a single COCO JSON file."""
    with open(coco_json) as f:
        coco = json.load(f)

    img_meta: dict[int, dict] = {im["id"]: im for im in coco["images"]}
    ann_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg[0], list):
            continue
        ann_by_img.setdefault(ann["image_id"], []).append(seg[0])

    records = []
    for iid, meta in img_meta.items():
        polys = ann_by_img.get(iid)
        if not polys:
            continue
        path = images_dir / meta["file_name"]
        if not path.exists():
            continue
        records.append({
            "image_path": path,
            "orig_w": meta["width"],
            "orig_h": meta["height"],
            "polygons": polys,
        })
    return records


def _load_records(
    coco_jsons: list[Path], images_dir: Path, max_images: int,
) -> list[dict]:
    """Load + union polygons from one or more COCO JSON files (Phase 3.2 uses
    GT polygons + self-training pseudo polygons in the second cycle).

    Records with the same file_name across JSONs are MERGED: their polygon
    lists are concatenated. Identical polygons across JSONs are not deduped
    because the rasterized union of overlapping polygons is the same mask.
    """
    by_path: dict[str, dict] = {}
    for j in coco_jsons:
        for rec in _load_records_one(j, images_dir):
            key = str(rec["image_path"])
            if key in by_path:
                by_path[key]["polygons"].extend(rec["polygons"])
            else:
                by_path[key] = rec
    records = list(by_path.values())
    if max_images > 0:
        records = records[:max_images]

    n_polys = sum(len(r["polygons"]) for r in records)
    logger.info(
        "Loaded %d images, %d polygons total (across %d COCO JSON file(s)).",
        len(records), n_polys, len(coco_jsons),
    )
    return records


# ── Mask-conditioned NT-Xent (batched, cross-image, no double-counted positive) ─

def _mc_nt_xent(
    anchors: torch.Tensor,        # (B*K, D)   L2-normalised
    positives: torch.Tensor,      # (B*K, D)   L2-normalised
    background: torch.Tensor,     # (B*L, D)   L2-normalised
    weights: torch.Tensor,        # (B*K,)     coverage weight per anchor
    temperature: float,
) -> torch.Tensor:
    """Canonical SimCLR / NT-Xent over a batch of K positives per image plus
    L explicit background-sampled negatives per image.

    For each anchor i:
      positive   = positives[i]
      negatives  = positives[j != i] U background[*]   (no background[i] exclusion
                  because background-sampled tokens are not paired with anchors)

    s = anchors @ all.T / tau where all = [positives; background]
    log_softmax over s with index i pointing at positives[i] gives the per-anchor loss.
    The positive appears EXACTLY ONCE in the denominator (it is positives[i] in the
    [positives; background] concatenation, summed over the softmax denominator).

    Loss is the coverage-weighted mean.
    """
    N = anchors.shape[0]
    if N == 0 or positives.shape[0] != N:
        return torch.zeros((), device=anchors.device, requires_grad=True)

    # Combined "all candidates" set.
    all_cands = torch.cat([positives, background], dim=0)  # (N + M, D)
    s = (anchors @ all_cands.T) / temperature              # (N, N+M)

    # Mask out same-anchor self-match in the *positives* slice. anchors[i] and
    # positives[i] are different views of the same patch, so anchor i must NOT
    # see positives[i] as a negative — it is the actual positive (handled by
    # log_softmax pointing index i at it). The diagonal of s[:, :N] would be
    # exactly that pair, so leaving it as-is is correct (it is the positive).
    # We still must mask anchor-vs-OWN-position when both are background — but
    # that does not happen because background patches are disjoint from anchor
    # positions by construction.

    log_probs = s.log_softmax(dim=1)
    pos_log_probs = log_probs[torch.arange(N, device=anchors.device),
                              torch.arange(N, device=anchors.device)]  # (N,)
    weighted = -pos_log_probs * weights
    denom = weights.sum().clamp(min=1e-6)
    return weighted.sum() / denom


# ── Training step ─────────────────────────────────────────────────────────────

def _sample_positions(
    coverage: np.ndarray,
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
    pos_threshold: float = 0.4,
    neg_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick (n_pos, n_neg) flat indices into the 37x37 grid.

    Positives drawn from cells with coverage > pos_threshold (clearly inside
    polygon). Negatives from coverage < neg_threshold (clearly outside).
    Returns (pos_idx, neg_idx, pos_weights). All arrays may be shorter than
    requested if the image has too little tomato area.
    """
    flat = coverage.reshape(-1)
    pos_pool = np.where(flat > pos_threshold)[0]
    neg_pool = np.where(flat < neg_threshold)[0]
    if len(pos_pool) == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32))

    pos_idx = pos_pool if len(pos_pool) <= n_pos else rng.choice(pos_pool, n_pos, replace=False)
    neg_idx = neg_pool if len(neg_pool) <= n_neg else rng.choice(neg_pool, n_neg, replace=False)
    pos_weights = flat[pos_idx].astype(np.float32)
    return pos_idx, neg_idx, pos_weights


def _flip_flat_idx(idx: np.ndarray) -> np.ndarray:
    """Map flat index in 37x37 grid to its horizontally-flipped position.
    For position (gy, gx) the flip is (gy, _DINO_GRID-1-gx).
    """
    gy = idx // _DINO_GRID
    gx = idx % _DINO_GRID
    return gy * _DINO_GRID + (_DINO_GRID - 1 - gx)


def train(
    records: list[dict],
    output_path: Path,
    device: torch.device,
    epochs: int,
    rank: int,
    lora_blocks: int,
    lr: float,
    temperature: float,
    batch_size: int,
    pos_per_image: int,
    neg_per_image: int,
    use_hflip: bool,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    logger.info("Loading DINOv2 (%s)...", _DINO_MODEL_NAME)
    dino = torch.hub.load("facebookresearch/dinov2", _DINO_MODEL_NAME, pretrained=True)
    dino = _inject_lora_qkv(dino, rank=rank, lora_blocks=lora_blocks)
    dino.train().to(device)

    trainable = [p for p in dino.parameters() if p.requires_grad]
    logger.info(
        "Trainable params: %d (%.3f%% of model)",
        sum(p.numel() for p in trainable),
        100.0 * sum(p.numel() for p in trainable) / sum(p.numel() for p in dino.parameters()),
    )

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    n_steps = max(1, len(records) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * n_steps, eta_min=lr / 10,
    )

    best_loss = float("inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        random.shuffle(records)
        bar = tqdm(range(0, len(records) - batch_size + 1, batch_size),
                   desc=f"Epoch {epoch}/{epochs}", unit="batch", ncols=80, leave=False)
        epoch_loss = 0.0
        n_batches = 0

        for start in bar:
            batch_recs = records[start:start + batch_size]

            # Build views A and B for each image in batch + sample positions.
            tensors_a, tensors_b = [], []
            anchors_meta: list[dict] = []  # one per image; positions and flip flag
            valid_recs = []

            for rec in batch_recs:
                bgr = cv2.imread(str(rec["image_path"]))
                if bgr is None:
                    continue

                rgb_box, scale, pad_x, pad_y = _letterbox(bgr)
                coverage = _polygons_to_coverage(
                    rec["polygons"], rec["orig_w"], rec["orig_h"],
                    scale, pad_x, pad_y,
                )
                pos_idx, neg_idx, pos_w = _sample_positions(
                    coverage, pos_per_image, neg_per_image, rng,
                )
                if len(pos_idx) == 0:
                    continue

                # Two augmented views; B optionally flipped.
                view_a = _augment_image(rgb_box, rng, flip=False)
                flip_b = use_hflip and rng.random() < 0.5
                view_b = _augment_image(rgb_box, rng, flip=flip_b)

                tensors_a.append(_to_tensor(view_a))
                tensors_b.append(_to_tensor(view_b))
                anchors_meta.append({
                    "pos_idx": pos_idx,
                    "neg_idx": neg_idx,
                    "pos_w": pos_w,
                    "flip_b": flip_b,
                })
                valid_recs.append(rec)

            if len(tensors_a) < 2:
                # Need at least two images for cross-image negatives to mean anything.
                continue

            batch_a = torch.stack(tensors_a).to(device)  # (B, 3, 518, 518)
            batch_b = torch.stack(tensors_b).to(device)

            # One forward per view (torch.cat the two batches for a single forward).
            combined = torch.cat([batch_a, batch_b], dim=0)  # (2B, 3, 518, 518)
            features = dino.forward_features(combined)
            patch_tokens = features["x_norm_patchtokens"]  # (2B, 1369, D)
            patch_norms = F.normalize(patch_tokens, dim=2)

            B = batch_a.shape[0]
            tokens_a = patch_norms[:B]
            tokens_b = patch_norms[B:]

            # Gather anchor (view_A), positive (view_B at same/flipped pos),
            # and negative (view_A at clear-bg pos) embeddings per image.
            anchor_list, positive_list, bg_list, weight_list = [], [], [], []
            for i, meta in enumerate(anchors_meta):
                pos_idx = torch.from_numpy(meta["pos_idx"]).to(device)
                neg_idx = torch.from_numpy(meta["neg_idx"]).to(device)
                pos_idx_b = torch.from_numpy(
                    _flip_flat_idx(meta["pos_idx"]) if meta["flip_b"] else meta["pos_idx"]
                ).to(device)

                anchor_list.append(tokens_a[i, pos_idx])               # (k, D)
                positive_list.append(tokens_b[i, pos_idx_b])           # (k, D)
                bg_list.append(tokens_a[i, neg_idx])                   # (l, D)
                weight_list.append(torch.from_numpy(meta["pos_w"]).to(device))

            anchors = torch.cat(anchor_list, dim=0)                    # (sum_k, D)
            positives = torch.cat(positive_list, dim=0)                # (sum_k, D)
            background = torch.cat(bg_list, dim=0)                     # (sum_l, D)
            weights = torch.cat(weight_list, dim=0)                    # (sum_k,)

            loss = _mc_nt_xent(anchors, positives, background, weights, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        bar.close()
        if n_batches == 0:
            logger.warning("Epoch %d: no usable batches.", epoch)
            continue
        mean_loss = epoch_loss / n_batches
        marker = ""
        if mean_loss < best_loss:
            best_loss = mean_loss
            lora_state = {k: v for k, v in dino.state_dict().items() if "lora_" in k}
            torch.save(lora_state, str(output_path))
            marker = "  ✓ saved"
        tqdm.write(f"  └─ Epoch {epoch}/{epochs}  loss={mean_loss:.4f}  best={best_loss:.4f}{marker}")

    logger.info("Done. Best loss=%.4f. Saved to %s", best_loss, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", type=Path, required=True,
                        help="data/Laboro-Tomato/annotations/train.json")
    parser.add_argument("--coco-json-aux", type=Path, nargs="*", default=[],
                        help="Additional COCO JSON files to UNION with --coco-json. "
                             "Used by Phase 3.2 self-training to add pseudo-labels "
                             "produced by perception/tools/generate_pseudo_labels.py.")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/dino_mc_lora.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="NT-Xent temperature. 0.1 is a milder contrast than the "
                             "old file's 0.07 — at 0.07 with the bug-free loss the "
                             "training was unstable on small batches.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Images per gradient step. >=2 for cross-image negatives.")
    parser.add_argument("--pos-per-image", type=int, default=16)
    parser.add_argument("--neg-per-image", type=int, default=16)
    parser.add_argument("--use-hflip", action="store_true", default=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)
    coco = args.coco_json if args.coco_json.is_absolute() else repo_root / args.coco_json
    imgs = args.train_images if args.train_images.is_absolute() else repo_root / args.train_images
    out = args.output if args.output.is_absolute() else repo_root / args.output

    for p, name in [(coco, "coco-json"), (imgs, "train-images")]:
        if not p.exists():
            logger.error("%s not found: %s", name, p)
            sys.exit(1)

    aux_paths: list[Path] = []
    for ap in args.coco_json_aux:
        ap_abs = ap if ap.is_absolute() else repo_root / ap
        if not ap_abs.exists():
            logger.error("--coco-json-aux not found: %s", ap_abs)
            sys.exit(1)
        aux_paths.append(ap_abs)

    device = _select_device()
    logger.info("Device: %s", device)

    records = _load_records([coco, *aux_paths], imgs, args.max_images)
    if not records:
        logger.error("No records found.")
        sys.exit(1)

    train(
        records, out, device,
        epochs=args.epochs, rank=args.rank, lora_blocks=args.lora_blocks,
        lr=args.lr, temperature=args.temperature,
        batch_size=args.batch_size,
        pos_per_image=args.pos_per_image, neg_per_image=args.neg_per_image,
        use_hflip=args.use_hflip, seed=args.seed,
    )


if __name__ == "__main__":
    main()
