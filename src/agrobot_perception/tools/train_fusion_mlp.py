#!/usr/bin/env python3
"""
train_fusion_mlp.py — Train the late-fusion MLP head on dumped train detections.

Workflow (Phase 2.2):
  1. Build a train-list (e.g. data/train_list.txt) and train_gt.csv from
     data/Laboro-Tomato/annotations/train.json polygons -> bbox.
  2. Run perception/eval/run_eval.py with --siglip --fusion-features-out
     features_train.jsonl over the train list at conf=0 to dump every detection
     with its 7 features attached.
  3. This script loads the JSONL + train GT, labels each detection as positive
     (IoU>=0.5 against any GT in the same image) or negative, trains the MLP
     with BCE loss, saves models/fusion_mlp.pt.

Why BCE on per-detection IoU>=0.5:
  We want the MLP to output a high score for detections that mAP@0.5 will
  count as TP and a low score for FPs. BCE on this exact label is the most
  direct signal short of differentiable mAP (which would require a much
  bigger optimisation framework).

Why not class-balanced loss:
  Detection imbalance is mild (typically 30-50% TPs at conf=0, since SAM2 AMG
  is selective). pos_weight is exposed via --pos-weight if needed.

Sprint 4: Phase 2.2.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def _load_gt(gt_csv: Path) -> dict[str, list[tuple]]:
    gt: dict[str, list[tuple]] = {}
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            try:
                key = str(Path(row["image_path"]).resolve())
                box = (float(row["x1"]), float(row["y1"]),
                       float(row["x2"]), float(row["y2"]))
            except (KeyError, ValueError):
                continue
            gt.setdefault(key, []).append(box)
    return gt


def _build_dataset(
    features_jsonl: Path,
    image_paths: list[Path],
    gt_by_image: dict[str, list[tuple]],
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Read dumped features + assign per-detection TP/FP labels via greedy IoU."""
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    with open(features_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            idx = obj["image_index"]
            if idx >= len(image_paths):
                continue
            key = str(Path(image_paths[idx]).resolve())
            gts = gt_by_image.get(key, [])

            # Greedy IoU matching at iou_threshold; once a GT is matched it
            # cannot be claimed by a lower-scoring detection — same convention
            # as compute_ap_iou_threshold in metrics.py.
            sorted_dets = sorted(
                obj["detections"], key=lambda d: d["score"], reverse=True,
            )
            gt_used = [False] * len(gts)
            for d in sorted_dets:
                box = tuple(d["box"])
                best_iou = 0.0
                best_j = -1
                for j, g in enumerate(gts):
                    if gt_used[j]:
                        continue
                    i = _iou(box, g)
                    if i > best_iou:
                        best_iou = i; best_j = j
                is_tp = best_iou >= iou_threshold and best_j >= 0
                if is_tp:
                    gt_used[best_j] = True
                X_list.append(np.asarray(d["features"], dtype=np.float32))
                y_list.append(1 if is_tp else 0)

    X = np.stack(X_list) if X_list else np.zeros((0, 7), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True,
                        help="JSONL produced by FeatureDumpWrapper (run_eval --fusion-features-out).")
    parser.add_argument("--image-list", type=Path, required=True,
                        help="The same val_list.txt-style file used when dumping features. "
                             "Lines must be in the same order as image_index in the JSONL.")
    parser.add_argument("--gt-csv", type=Path, required=True,
                        help="Train GT CSV (image_path,x1,y1,x2,y2,label).")
    parser.add_argument("--output", type=Path, default=Path("models/fusion_mlp.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="BCE positive-class weight. >1 boosts TP recall.")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of dumped detections held out for early stopping.")
    args = parser.parse_args()

    repo_root = _repo_root()
    feat_path = args.features if args.features.is_absolute() else repo_root / args.features
    img_list = args.image_list if args.image_list.is_absolute() else repo_root / args.image_list
    gt_path = args.gt_csv if args.gt_csv.is_absolute() else repo_root / args.gt_csv
    out_path = args.output if args.output.is_absolute() else repo_root / args.output

    image_paths: list[Path] = []
    with open(img_list) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                p = Path(line)
                if not p.is_absolute():
                    p = repo_root / p
                image_paths.append(p)

    gt_by_image = _load_gt(gt_path)
    logger.info("Loaded %d images, %d images with GT.",
                len(image_paths), len(gt_by_image))

    X, y = _build_dataset(feat_path, image_paths, gt_by_image)
    logger.info("Built dataset: %d detections, %d TPs (%.1f%% positive).",
                len(y), int(y.sum()), 100.0 * y.mean() if len(y) else 0.0)
    if len(y) == 0:
        logger.error("No detections in dump. Check --features and --image-list.")
        sys.exit(1)

    # Standardize features (mean/std on the train split). Saved with the MLP.
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(y))
    n_val = max(1, int(args.val_frac * len(y)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    sys.path.insert(0, str(repo_root / "perception"))
    from eval.fusion_mlp import FusionMLP, N_FEATURES

    model = FusionMLP(in_dim=N_FEATURES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    Xv = torch.from_numpy(X_val_n)
    yv = torch.from_numpy(y_val)

    best_val_ap = -1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item(); n_batches += 1
        train_loss /= max(1, n_batches)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, yv, pos_weight=pos_weight).item()
            # Cheap proxy: AUC via score ordering.
            val_probs = torch.sigmoid(val_logits).numpy()
            order = np.argsort(-val_probs)
            tp = (yv.numpy()[order] == 1).cumsum()
            fp = (yv.numpy()[order] == 0).cumsum()
            n_pos = max(1, int(yv.sum()))
            precisions = tp / np.maximum(tp + fp, 1)
            recalls = tp / n_pos
            # np.trapezoid exists in numpy>=2, trapz works in all versions.
            # Use a manual cumulative rectangle sum to stay version-agnostic.
            val_ap = float(np.sum(np.diff(np.concatenate([[0.0], recalls])) * precisions))

        marker = ""
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save({
                "model": model.state_dict(),
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
            }, out_path)
            marker = " ✓ saved"
        logger.info(
            "epoch %3d  train_loss=%.4f  val_loss=%.4f  val_ap=%.4f%s",
            epoch, train_loss, val_loss, val_ap, marker,
        )

    logger.info("Best val_ap = %.4f. Saved to %s", best_val_ap, out_path)
    logger.info("")
    logger.info("Eval with learned fusion:")
    logger.info(
        "  AGROBOT_FORCE_CPU=1 PYTHONPATH=perception "
        "python3 perception/eval/run_eval.py "
        "--val-list data/val_list.txt --gt-csv data/val_gt.csv "
        "--detector sam2_amg --amg-points 28 --siglip --fusion-mlp %s "
        "--metric coco --confidence 0.0 --max-detections 30 --nms-iou 0.5",
        out_path,
    )


if __name__ == "__main__":
    main()
