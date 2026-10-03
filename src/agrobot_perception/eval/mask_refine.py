"""
mask_refine.py — Second-pass SAM2 mask refinement for surviving detections.

Why this exists (Phase 2.3):
  SAM2 AMG places point prompts on a fixed grid. For tomatoes whose centre
  falls between grid points, the AMG-generated mask boundary is offset from
  the actual object boundary by up to half a grid cell (~9 px at pts=28).
  That offset costs IoU vs GT — a tomato detection with a 5-pixel boundary
  offset routinely sits at IoU ~0.45 vs its GT box and gets counted as a
  false positive at the IoU=0.5 threshold.

  This wrapper takes each detection that survived the detector's own
  filter+NMS+cap and re-runs SAM2 with the detection's centroid as a single
  positive point prompt. SAM2's mask decoder is symmetric in the prompt — a
  centroid prompt produces a mask centred on the actual object, not on the
  nearest grid cell. We then re-score the refined mask with the same
  coverage-weighted DINOv2 formula and only keep the refinement when it
  improves the score AND has high enough IoU with the original.

Why we re-score:
  Refining without re-scoring would let SAM2 hallucinate a "better" mask of a
  non-tomato object (e.g. a leaf cluster nearby) and silently replace a
  correct detection with a wrong one. Requiring score improvement uses the
  detector's own semantic check as a guardrail.

Architecture coupling:
  This wrapper specifically targets SAM2AMGDetector because it reuses the
  inner detector's already-loaded SAM2 predictor (via base._amg.predictor),
  DINOv2, query embedding, and negative embedding. It does NOT load any
  additional models. Composes with TTAWrapper and SigLIPRescoringWrapper
  (refine inside, TTA outside).

Sprint 4: Phase 2.3.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DINO_PATCH_SIZE = 14
_DINO_INPUT_SIZE = 518
_DINO_GRID = _DINO_INPUT_SIZE // _DINO_PATCH_SIZE  # 37


def _box_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _mask_to_patch_coverage(seg: np.ndarray) -> torch.Tensor:
    """Mirror sam2_amg_detector._mask_to_patch_coverage exactly."""
    seg_f = seg[:_DINO_GRID * _DINO_PATCH_SIZE, :_DINO_GRID * _DINO_PATCH_SIZE].astype(np.float32)
    blocks = seg_f.reshape(_DINO_GRID, _DINO_PATCH_SIZE, _DINO_GRID, _DINO_PATCH_SIZE)
    return torch.from_numpy(blocks.mean(axis=(1, 3)))


class MaskRefineWrapper:
    """Wraps SAM2AMGDetector to add a second SAM2 pass per surviving detection.

    Args:
        base: a SAM2AMGDetector instance (must expose _amg, _dino, _query_embedding,
              _negative_embedding, _negative_weight, _dino_score_weight, _device).
        min_iou_with_original: minimum IoU between refined and original box to
            accept the refinement. Lower allows more drift; higher rejects useful
            refinements that shift the box significantly.
        min_score_delta: minimum (refined_score - original_score) to accept.
            Use 0.0 to accept any improvement; positive values reject ties.
        only_borderline: only refine detections with original score within this
            margin of the confidence threshold. Cheap (skips the obviously-good
            detections) but limits ceiling. Set to a large value to refine all.
    """

    def __init__(
        self,
        base,
        min_iou_with_original: float = 0.5,
        min_score_delta: float = 0.0,
        only_borderline: float = 999.0,
    ) -> None:
        self._base = base
        self._min_iou = min_iou_with_original
        self._min_score_delta = min_score_delta
        self._borderline_margin = only_borderline

        # Validate that base exposes the expected internals.
        for attr in ("_amg", "_dino", "_query_embedding", "_device",
                     "_dino_score_weight", "_conf_threshold"):
            if not hasattr(base, attr):
                raise TypeError(
                    f"MaskRefineWrapper requires a SAM2AMGDetector-like base; "
                    f"missing attribute {attr!r}."
                )

    def _reconstruct_rgb(self, preprocessed_chw: np.ndarray) -> np.ndarray:
        rgb_float = (
            preprocessed_chw * _IMAGENET_STD[:, None, None]
            + _IMAGENET_MEAN[:, None, None]
        )
        rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
        return np.transpose(rgb_uint8, (1, 2, 0))

    def _dino_forward(self, preprocessed_chw: np.ndarray) -> torch.Tensor:
        """Single DINOv2 forward; cached patch_norms reused across all refinements."""
        tensor = torch.from_numpy(preprocessed_chw).unsqueeze(0).to(self._base._device)
        with torch.no_grad():
            features = self._base._dino.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"].squeeze(0).float()
        return F.normalize(patch_tokens, dim=1)

    def _score_mask(self, seg: np.ndarray, patch_norms: torch.Tensor) -> tuple[float, float, float]:
        """Return (tomato_sim, neg_sim, dino_sim) using the base detector's formula."""
        coverage = _mask_to_patch_coverage(seg).reshape(-1).to(self._base._device)
        if coverage.sum() < 1e-6:
            return 0.0, 0.0, 0.0

        q = self._base._query_embedding
        if q.dim() == 1:
            sims = patch_norms @ q
            tomato_sim = float((sims * coverage).sum() / coverage.sum())
        else:
            sims_k = patch_norms @ q.T
            per_proto = (sims_k * coverage.unsqueeze(1)).sum(dim=0) / coverage.sum()
            tomato_sim = float(per_proto.max())

        neg_sim = 0.0
        if (self._base._negative_embedding is not None
                and self._base._negative_weight > 0):
            n = self._base._negative_embedding
            if n.dim() == 1:
                neg_sims = patch_norms @ n
                neg_sim = float((neg_sims * coverage).sum() / coverage.sum())
            else:
                neg_sims_k = patch_norms @ n.T
                neg_sim = float(
                    ((neg_sims_k * coverage.unsqueeze(1)).sum(dim=0) / coverage.sum()).max()
                )
            dino_sim = tomato_sim - self._base._negative_weight * neg_sim
        else:
            dino_sim = tomato_sim

        return tomato_sim, neg_sim, dino_sim

    def _refine_one(
        self,
        rgb_hwc: np.ndarray,
        det: dict,
        patch_norms: torch.Tensor,
        predictor,
    ) -> Optional[dict]:
        """Re-prompt SAM2 with detection centroid; return improved det or None."""
        x1, y1, x2, y2 = det["box"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        try:
            masks_np, sam_scores_np, _ = predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
        except Exception as exc:
            logger.debug("SAM2 refine failed at (%.1f,%.1f): %s", cx, cy, exc)
            return None

        # Choose mask with highest SAM2 predicted_iou.
        best = int(np.argmax(sam_scores_np))
        seg = masks_np[best].astype(bool)
        pred_iou = float(sam_scores_np[best])

        if seg.sum() < 50:  # tiny mask = SAM2 prompt landed off-target
            return None

        rows = np.where(seg.any(axis=1))[0]
        cols = np.where(seg.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return None
        new_box = [float(cols[0]), float(rows[0]), float(cols[-1]), float(rows[-1])]

        # IoU vs original — don't accept refinements that drift to a different object.
        iou = _box_iou(new_box, [x1, y1, x2, y2])
        if iou < self._min_iou:
            return None

        tomato_sim, neg_sim, dino_sim = self._score_mask(seg, patch_norms)
        alpha = self._base._dino_score_weight
        new_score = alpha * dino_sim + (1.0 - alpha) * pred_iou

        if new_score < det.get("dino_sim", det["score"]) + self._min_score_delta:
            # Refinement did not improve the dino-side score; keep original.
            return None

        return {
            "box": new_box,
            "score": new_score,
            "label": det.get("label", "tomato"),
            "mask": seg.astype(np.uint8),
            "tomato_sim": float(tomato_sim),
            "neg_sim": float(neg_sim),
            "dino_sim": float(dino_sim),
            "pred_iou": float(pred_iou),
            "refined": True,
        }

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        dets = self._base.detect(preprocessed_chw)
        if not dets:
            return dets

        # Reuse the AMG's already-loaded image predictor for point-prompt mode.
        # SAM2AutomaticMaskGenerator stores it as self.predictor.
        amg = self._base._amg
        predictor = getattr(amg, "predictor", None)
        if predictor is None:
            logger.warning("Inner detector has no SAM2 predictor; refinement skipped.")
            return dets

        rgb_hwc = self._reconstruct_rgb(preprocessed_chw)
        try:
            predictor.set_image(rgb_hwc)
        except Exception as exc:
            logger.warning("SAM2 set_image() failed in refine: %s", exc)
            return dets

        patch_norms = self._dino_forward(preprocessed_chw)

        out: list[dict] = []
        for det in dets:
            if (det["score"] - self._base._conf_threshold) > self._borderline_margin:
                # Far above threshold → cheap-skip refinement.
                out.append(det)
                continue
            refined = self._refine_one(rgb_hwc, det, patch_norms, predictor)
            out.append(refined if refined is not None else det)

        out.sort(key=lambda d: d["score"], reverse=True)
        return out
