#!/usr/bin/env python3
"""
finetune_dino_lora.py — LoRA fine-tuning of DINOv2 ViT-B/14 on Laboro Tomato patches.

Why LoRA over full fine-tune:
  DINOv2 ViT-B/14 has 86M parameters. Fine-tuning all of them on 643 images
  causes catastrophic forgetting: the universal visual representations that make
  DINOv2 useful for zero-shot scoring are destroyed. LoRA (Low-Rank Adaptation)
  freezes all weights and injects trainable low-rank matrices into the Q and V
  projections of the last N transformer blocks. At rank=8, this adds ~0.7M
  parameters (0.8% of total) while keeping the frozen backbone intact.

  The scoring function in SAM2AMGDetector and SAM2SemanticDetector is entirely
  cosine similarity in DINOv2 feature space. LoRA adapters shift the feature
  space to be more discriminative for tomato vs background patches without
  changing the scoring function architecture.

Loss:
  NT-Xent (SimCLR-style) contrastive loss over mini-batches of (anchor, positive,
  negatives). Positives = patches from the same tomato instance (random augment).
  Negatives = patches from background regions in the same image. Temperature τ=0.07.

  L_NT-Xent = -log( exp(cos(z_a, z_p)/τ) / Σ_k exp(cos(z_a, z_k)/τ) )

  This is equivalent to encouraging the model to place tomato patches together
  and background patches far away in L2-normalised embedding space.

Usage [NUCBOX] (requires ROCm unblocked — see docs/SPRINT3_ROCM_ISSUE.md):
  PYTHONPATH=perception \\
    python3 perception/tools/finetune_dino_lora.py \\
    --train-images data/Laboro-Tomato/train/images \\
    --train-labels data/Laboro-Tomato/train/labels \\
    --output models/dino_lora.pt \\
    --epochs 10 --rank 8 --lora-blocks 4

Usage [MAC] (smoke-test on CPU, ~30min for 10 images):
  AGROBOT_FORCE_CPU=1 PYTHONPATH=perception \\
    python3 perception/tools/finetune_dino_lora.py \\
    --train-images data/Laboro-Tomato/train/images \\
    --train-labels data/Laboro-Tomato/train/labels \\
    --output models/dino_lora.pt \\
    --epochs 2 --max-images 10

After training, point the detector at the LoRA model:
  AGROBOT_FORCE_CPU=1 PYTHONPATH=perception \\
    python3 perception/eval/run_eval.py \\
    --detector sam2_amg --amg-points 20 --confidence 0.2 \\
    --negative-weight 1.0 --nms-iou 0.5 \\
    --dino-lora-path models/dino_lora.pt

Sprint 4: E6 — DINOv2 LoRA adaptation. Compute-blocked until ROCm 7.3 resolved.
"""

from __future__ import annotations

import argparse
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
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _select_device() -> torch.device:
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── LoRA implementation ───────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank adapter.

    Original weight W_0 ∈ R^{d_out × d_in} is frozen.
    LoRA adds: h = W_0 x + (B A) x,  where A ∈ R^{r × d_in}, B ∈ R^{d_out × r}.
    A is init Gaussian, B is init zeros → adapter output is zero at training start
    so the model is identical to the pretrained baseline at epoch 0.

    Scaling by α/r (following the original LoRA paper) keeps the effective
    learning rate invariant to rank choice.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.linear = linear          # frozen
        d_out, d_in = linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale = alpha / rank

        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def inject_lora(model: nn.Module, rank: int, lora_blocks: int) -> nn.Module:
    """Replace Q and V projections in the last `lora_blocks` transformer blocks with LoRA.

    DINOv2 ViT-B has 12 transformer blocks. We adapt only the last N because
    those encode the most task-specific high-level semantics; early blocks
    encode low-level structure that is universal across tasks.

    Critical: freeze ALL backbone parameters first so only the newly created
    lora_A and lora_B tensors (which default to requires_grad=True) are trained.
    Without this, every parameter that was already requires_grad=True (the full
    86M backbone) gets passed to the optimizer — full fine-tuning, not LoRA.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    # DINOv2 stores blocks at model.blocks (a ModuleList of TransformerBlock)
    blocks = list(model.blocks)
    target_blocks = blocks[-lora_blocks:]

    n_replaced = 0
    for block in target_blocks:
        # DINOv2's attention module: block.attn.qkv is a single fused Linear (3*D, D)
        # We wrap the entire qkv projection — LoRA on Q+V+K is a superset of Q+V only
        # but simpler to implement without splitting the fused projection.
        if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
            original = block.attn.qkv
            if isinstance(original, nn.Linear):
                block.attn.qkv = LoRALinear(original, rank=rank, alpha=float(rank * 2))
                n_replaced += 1
        # Some DINOv2 variants also have a separate proj layer
        if hasattr(block, "attn") and hasattr(block.attn, "proj"):
            original = block.attn.proj
            if isinstance(original, nn.Linear):
                block.attn.proj = LoRALinear(original, rank=rank, alpha=float(rank * 2))
                n_replaced += 1

    logger.info("Injected LoRA into %d Linear layers (rank=%d, last %d blocks).", n_replaced, rank, lora_blocks)
    return model


# ── Data loading ──────────────────────────────────────────────────────────────

def _yolo_to_pixel(cx: float, cy: float, w: float, h: float,
                   img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)


def _boxes_to_patch_masks(boxes_pixel, orig_w, orig_h):
    """Return (pos_mask, neg_mask) as (37,37) bool arrays."""
    scale = min(_DINO_INPUT_SIZE / orig_w, _DINO_INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale); new_h = int(orig_h * scale)
    pad_x = (_DINO_INPUT_SIZE - new_w) // 2
    pad_y = (_DINO_INPUT_SIZE - new_h) // 2
    pos = np.zeros((_DINO_GRID, _DINO_GRID), dtype=bool)
    for x1, y1, x2, y2 in boxes_pixel:
        gx1 = max(0, int(x1 * scale + pad_x) // _DINO_PATCH_SIZE)
        gy1 = max(0, int(y1 * scale + pad_y) // _DINO_PATCH_SIZE)
        gx2 = min(_DINO_GRID - 1, int(x2 * scale + pad_x) // _DINO_PATCH_SIZE)
        gy2 = min(_DINO_GRID - 1, int(y2 * scale + pad_y) // _DINO_PATCH_SIZE)
        if gx2 >= gx1 and gy2 >= gy1:
            pos[gy1:gy2 + 1, gx1:gx2 + 1] = True
    neg = ~pos
    return pos, neg


def _preprocess(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE)).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)))


def load_records(images_dir: Path, labels_dir: Path, max_images: int = 0) -> list[dict]:
    """Load image paths and YOLO bounding box labels."""
    files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if max_images > 0:
        files = files[:max_images]

    records = []
    for img_path in files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = (float(p) for p in parts[:5])
                boxes.append(_yolo_to_pixel(cx, cy, bw, bh, orig_w, orig_h))
        if boxes:
            records.append({
                "image_path": img_path,
                "orig_w": orig_w, "orig_h": orig_h,
                "boxes": boxes,
            })
    logger.info("Loaded %d labeled images.", len(records))
    return records


# ── NT-Xent loss ──────────────────────────────────────────────────────────────

def nt_xent_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent contrastive loss.

    anchors:   (N, D) L2-normalised tomato patch embeddings
    positives: (N, D) L2-normalised augmented tomato patch embeddings
    negatives: (M, D) L2-normalised background patch embeddings

    For each anchor i, the positive is positives[i]. All other anchors,
    their positives, and all negatives are treated as negatives.

    L = -1/N Σ_i log( exp(cos(a_i, p_i)/τ) / Σ_j≠i exp(cos(a_i, z_j)/τ) )
    """
    N = anchors.shape[0]
    # Concatenate all embeddings: [anchors | positives | negatives]
    all_embeds = torch.cat([anchors, positives, negatives], dim=0)  # (2N+M, D)

    # Similarity matrix between anchors and all embeddings
    sims = anchors @ all_embeds.T  # (N, 2N+M)
    sims = sims / temperature

    # Positive pairs: anchor i pairs with positives[i] at index N+i
    pos_indices = torch.arange(N, 2 * N, device=anchors.device)  # [N, N+1, ..., 2N-1]

    loss = torch.tensor(0.0, device=anchors.device)
    for i in range(N):
        pos_sim = sims[i, pos_indices[i]]
        # Mask out self-similarity (anchor i with itself)
        mask = torch.ones(sims.shape[1], dtype=torch.bool, device=anchors.device)
        mask[i] = False
        all_sims = sims[i][mask]
        # Log-sum-exp: loss for anchor i
        loss = loss + pos_sim - torch.logsumexp(torch.cat([pos_sim.unsqueeze(0), all_sims]), dim=0)

    return -loss / N


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    records: list[dict],
    output_path: Path,
    device: torch.device,
    epochs: int = 10,
    rank: int = 8,
    lora_blocks: int = 4,
    lr: float = 1e-4,
    temperature: float = 0.07,
    min_patches: int = 4,
    max_neg_patches: int = 32,
) -> None:
    logger.info("Loading DINOv2 (%s)...", _DINO_MODEL_NAME)
    dino = torch.hub.load("facebookresearch/dinov2", _DINO_MODEL_NAME, pretrained=True)
    dino = inject_lora(dino, rank=rank, lora_blocks=lora_blocks)
    dino.train().to(device)

    trainable_params = [p for p in dino.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in dino.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(
        "Parameters: %d trainable / %d total (%.2f%% of model).",
        trainable_count, total_params, 100.0 * trainable_count / total_params,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(records), eta_min=lr / 10,
    )

    best_loss = float("inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0
        random.shuffle(records)

        bar = tqdm(records, desc=f"Epoch {epoch}/{epochs}", unit="img", ncols=72, leave=False)
        for rec in bar:
            bgr = cv2.imread(str(rec["image_path"]))
            if bgr is None:
                continue

            tensor = _preprocess(bgr).unsqueeze(0).to(device)

            feats = dino.forward_features(tensor)
            patch_tokens = feats["x_norm_patchtokens"].squeeze(0)  # (1369, D)

            pos_mask, neg_mask = _boxes_to_patch_masks(rec["boxes"], rec["orig_w"], rec["orig_h"])
            pos_flat = torch.from_numpy(pos_mask.reshape(-1))
            neg_flat = torch.from_numpy(neg_mask.reshape(-1))

            pos_patches = patch_tokens[pos_flat]   # (P, D)
            neg_patches  = patch_tokens[neg_flat]  # (N_neg, D)

            if pos_patches.shape[0] < min_patches:
                continue

            # Sub-sample negatives to keep batch size manageable
            if neg_patches.shape[0] > max_neg_patches:
                idx = torch.randperm(neg_patches.shape[0])[:max_neg_patches]
                neg_patches = neg_patches[idx]

            # Augment positives: random dropout of patch features as augmentation.
            # Drop 20% of feature dimensions to zero — a simple form of patch augmentation
            # that creates diverse positive pairs without requiring image-level transforms.
            dropout_mask = (torch.rand_like(pos_patches) > 0.2).float()
            pos_aug = F.normalize(pos_patches * dropout_mask, dim=1)
            anchors  = F.normalize(pos_patches, dim=1)
            neg_norm = F.normalize(neg_patches, dim=1)

            loss = nt_xent_loss(anchors, pos_aug, neg_norm, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        bar.close()

        if n_batches == 0:
            logger.warning("Epoch %d: no batches processed.", epoch)
            continue

        mean_loss = epoch_loss / n_batches
        saved = ""
        if mean_loss < best_loss:
            best_loss = mean_loss
            # Save only the LoRA adapter weights (not the full frozen backbone)
            lora_state = {k: v for k, v in dino.state_dict().items() if "lora_" in k}
            torch.save(lora_state, str(output_path))
            saved = "  ✓ saved"
        tqdm.write(f"  └─ Epoch {epoch}/{epochs}  loss={mean_loss:.4f}  best={best_loss:.4f}{saved}")

    logger.info("Training complete. Best loss: %.4f", best_loss)
    logger.info("LoRA adapters saved to: %s", output_path)
    logger.info("")
    logger.info(
        "Rebuild query embedding with LoRA model then run eval:\n"
        "  PYTHONPATH=perception python3 perception/tools/build_query_embedding.py \\\n"
        "    --train-images data/Laboro-Tomato/train/images \\\n"
        "    --train-labels data/Laboro-Tomato/train/labels \\\n"
        "    --output models/query_embedding_lora.pt \\\n"
        "    --dino-lora-path models/dino_lora.pt\n"
        "\n"
        "  PYTHONPATH=perception python3 perception/eval/run_eval.py \\\n"
        "    --detector sam2_amg --amg-points 20 --confidence 0.2 \\\n"
        "    --negative-weight 1.0 --nms-iou 0.5 \\\n"
        "    --dino-lora-path models/dino_lora.pt \\\n"
        "    --query-embedding models/query_embedding_lora.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of DINOv2 ViT-B/14 for tomato patch discrimination."
    )
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("models/dino_lora.pt"),
        help="Output path for LoRA adapter weights (adapter keys only, not full model).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank (default 8).")
    parser.add_argument(
        "--lora-blocks", type=int, default=4,
        help="Number of final transformer blocks to adapt (default 4 of 12).",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07, help="NT-Xent temperature.")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else repo_root / p

    train_images = _abs(args.train_images)
    train_labels = _abs(args.train_labels)
    output       = _abs(args.output)

    for p, name in [(train_images, "train-images"), (train_labels, "train-labels")]:
        if not p.exists():
            logger.error("%s not found: %s", name, p)
            sys.exit(1)

    device = _select_device()
    logger.info("Device: %s", device)

    records = load_records(train_images, train_labels, max_images=args.max_images)
    if not records:
        logger.error("No valid records found.")
        sys.exit(1)

    train(
        records, output, device,
        epochs=args.epochs, rank=args.rank,
        lora_blocks=args.lora_blocks, lr=args.lr,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
