"""
fusion_mlp.py — Learned late-fusion head over per-detection features (Phase 2.2).

Why learned fusion replaces fixed weights:
  Phase 1.3 fuses dino_sim + siglip_sim + pred_iou with hand-picked weights
  (0.4/0.4/0.2). Those weights are dataset-blind: they ignore the fact that on
  Laboro Tomato, pred_iou correlates with IoU vs GT differently for green vs
  ripe tomatoes (SAM2 is trained on web images, mostly red-ish objects). A
  small MLP trained on train-set detections learns the right per-feature
  weighting AND captures non-linear interactions (e.g. "pred_iou matters more
  when the box is small").

What features the MLP sees:
  1. dino_sim       — DINOv2 contrastive score (already includes negative term)
  2. siglip_sim     — SigLIP image-text score (positive prompts max - negative max)
  3. pred_iou       — SAM2's self-assessed mask quality
  4. mask_area_norm — mask pixel count / (518*518). Small detections are noisier.
  5. circularity    — 4*pi*area / perimeter^2. Round=1, elongated leaves<<1.
  6. color_mean_h   — mean OpenCV Hue (0-180) inside bbox. Tomato hues cluster.
  7. color_sat_mean — mean Saturation. Distinguishes tomatoes from grey background.

  These 7 features are cheap to compute and cover orthogonal failure modes
  (semantic identity, mask shape, geometry, color) so the MLP can learn
  correlations the fixed-weight fusion cannot express.

Usage flow (Phase 2.2):
  1. Dump train-set features:
       run_eval.py --val-list data/train_list.txt --gt-csv data/train_gt.csv \\
                   --siglip --fusion-features-out features_train.jsonl \\
                   --confidence 0.0 --max-detections 1000 --nms-iou 0.0
  2. Train MLP:
       python perception/tools/train_fusion_mlp.py \\
              --features features_train.jsonl --output models/fusion_mlp.pt
  3. Eval with learned fusion:
       run_eval.py --val-list data/val_list.txt --gt-csv data/val_gt.csv \\
                   --siglip --fusion-mlp models/fusion_mlp.pt

This module provides:
  - FEATURE_NAMES:           canonical feature ordering
  - extract_features:        per-detection feature vector (used by dump and runtime)
  - FusionMLP:               the small torch.nn module
  - FusionMLPWrapper:        runtime wrapper that re-scores using a trained MLP

Sprint 4: Phase 2.2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 518

FEATURE_NAMES = (
    "dino_sim",
    "siglip_sim",
    "pred_iou",
    "mask_area_norm",
    "circularity",
    "color_mean_h",
    "color_sat_mean",
)
N_FEATURES = len(FEATURE_NAMES)


class _DetectorLike(Protocol):
    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]: ...


def reconstruct_rgb(preprocessed_chw: np.ndarray) -> np.ndarray:
    rgb_float = (
        preprocessed_chw * _IMAGENET_STD[:, None, None]
        + _IMAGENET_MEAN[:, None, None]
    )
    rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    return np.transpose(rgb_uint8, (1, 2, 0))


def extract_features(det: dict, rgb_hwc: np.ndarray) -> np.ndarray:
    """Compute the 7-dim feature vector for a single detection.

    Detection must carry `dino_sim`, `siglip_sim`, `pred_iou`, `mask`, `box`.
    Geometry features are robust to mask noise (uses cv2.findContours on the
    largest connected component).
    """
    dino_sim = float(det.get("dino_sim", det.get("score", 0.0)))
    siglip_sim = float(det.get("siglip_sim", 0.0))
    pred_iou = float(det.get("pred_iou", 0.0))

    mask = det.get("mask")
    if mask is None or mask.sum() == 0:
        mask_area_norm = 0.0
        circularity = 0.0
    else:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        area = float(mask_u8.sum() / 255)
        mask_area_norm = area / (_INPUT_SIZE * _INPUT_SIZE)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Largest contour = main object; ignores noise speckles.
            cnt = max(contours, key=cv2.contourArea)
            perim = float(cv2.arcLength(cnt, True))
            circularity = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
        else:
            circularity = 0.0

    # HSV color stats inside the bbox.
    x1, y1, x2, y2 = (int(round(v)) for v in det["box"])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(rgb_hwc.shape[1], x2); y2 = min(rgb_hwc.shape[0], y2)
    if x2 > x1 and y2 > y1:
        bgr_crop = cv2.cvtColor(rgb_hwc[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        color_mean_h = float(hsv[:, :, 0].mean())
        color_sat_mean = float(hsv[:, :, 1].mean()) / 255.0
    else:
        color_mean_h = 0.0
        color_sat_mean = 0.0

    return np.array([
        dino_sim, siglip_sim, pred_iou,
        mask_area_norm, circularity,
        color_mean_h / 180.0, color_sat_mean,  # both in [0, 1]
    ], dtype=np.float32)


class FusionMLP(nn.Module):
    """Tiny MLP: 7 -> 32 -> 16 -> 1. Output is a logit; sigmoid for probability.

    Small enough to train on CPU in seconds even with 100k examples; small
    enough at inference that latency overhead vs fixed-weight fusion is zero.
    """

    def __init__(self, in_dim: int = N_FEATURES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def score(self, features: np.ndarray) -> np.ndarray:
        """Convenience: numpy in, numpy probability out."""
        device = next(self.parameters()).device
        x = torch.from_numpy(features.astype(np.float32)).to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.forward(x)
        return torch.sigmoid(logits).cpu().numpy()


class FusionMLPWrapper:
    """Wraps a detector to re-score detections with a trained FusionMLP.

    Args:
        base: a detector that already attaches dino_sim, siglip_sim, pred_iou,
            mask. Compose AFTER SigLIPRescoringWrapper so siglip_sim is set.
        mlp_path: path to a trained FusionMLP state dict.
        nms_iou_threshold, max_detections, confidence_threshold: applied on
            the re-scored ranking, same as in SigLIPRescoringWrapper.
    """

    def __init__(
        self,
        base: _DetectorLike,
        mlp_path: str,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 30,
        confidence_threshold: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self._base = base
        self._nms_iou = nms_iou_threshold
        self._max_detections = max_detections
        self._conf_threshold = confidence_threshold
        self._device = device or torch.device("cpu")

        self._mlp = FusionMLP().to(self._device).eval()
        ckpt = torch.load(mlp_path, map_location=self._device)
        # train_fusion_mlp.py saves {"model": state_dict, "feature_mean": [...],
        # "feature_std": [...]}. Older variant saved a bare state_dict; support both.
        if isinstance(ckpt, dict) and "model" in ckpt:
            self._mlp.load_state_dict(ckpt["model"])
            self._feature_mean = np.asarray(ckpt.get("feature_mean", [0.0] * N_FEATURES),
                                            dtype=np.float32)
            self._feature_std = np.asarray(ckpt.get("feature_std", [1.0] * N_FEATURES),
                                           dtype=np.float32)
        else:
            self._mlp.load_state_dict(ckpt)
            self._feature_mean = np.zeros(N_FEATURES, dtype=np.float32)
            self._feature_std = np.ones(N_FEATURES, dtype=np.float32)
        logger.info("Loaded fusion MLP from %s.", mlp_path)

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        dets = self._base.detect(preprocessed_chw)
        if not dets:
            return dets
        rgb_hwc = reconstruct_rgb(preprocessed_chw)
        feats = np.stack([extract_features(d, rgb_hwc) for d in dets])
        # Apply the same standardization the trainer used. Skipping it leaves
        # the MLP scoring on a different distribution from training -> garbage.
        feats_norm = (feats - self._feature_mean) / self._feature_std
        probs = self._mlp.score(feats_norm)  # (N,) probabilities

        for d, p, f in zip(dets, probs, feats):
            d["fusion_score"] = float(p)
            d["score"] = float(p)  # MLP probability replaces fused score
            d["fusion_features"] = {n: float(v) for n, v in zip(FEATURE_NAMES, f)}

        if self._conf_threshold > 0:
            dets = [d for d in dets if d["score"] >= self._conf_threshold]

        # Cheap NMS replay using the MLP-derived score order.
        from eval.tta import _box_nms
        dets = _box_nms(dets, self._nms_iou)
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets[: self._max_detections]


class FeatureDumpWrapper:
    """Runs the inner detector, computes features per detection, writes JSONL.

    Used by the train-set dump step (Phase 2.2 step 1). Produces input for
    perception/tools/train_fusion_mlp.py. Does NOT change the score (so the
    dump-time NMS reflects the existing detector ordering).

    Each output line: {image_path, detections: [{box, score, dino_sim, ...,
    features: [...]}], image_index: int}.
    """

    def __init__(self, base: _DetectorLike, output_path: Path) -> None:
        self._base = base
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate at start so re-runs do not append.
        self._output_path.write_text("")
        self._idx = 0

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        import json
        dets = self._base.detect(preprocessed_chw)
        rgb_hwc = reconstruct_rgb(preprocessed_chw)
        records = []
        for d in dets:
            f = extract_features(d, rgb_hwc)
            records.append({
                "box": [float(b) for b in d["box"]],
                "score": float(d["score"]),
                "dino_sim": float(d.get("dino_sim", 0.0)),
                "siglip_sim": float(d.get("siglip_sim", 0.0)),
                "pred_iou": float(d.get("pred_iou", 0.0)),
                "features": [float(x) for x in f],
            })
        with open(self._output_path, "a") as f:
            f.write(json.dumps({"image_index": self._idx, "detections": records}) + "\n")
        self._idx += 1
        return dets
