"""
siglip_rescoring.py — Per-detection SigLIP-conditioned re-scoring (Phase 1.3).

Why SigLIP on top of DINOv2:
  DINOv2 patch features are dense and self-supervised — strong on local texture
  and shape, weaker on global object identity. SigLIP is contrastively pretrained
  on web-scale image-text pairs — strong on object identity from a single global
  embedding, weaker on dense localisation. Their failure modes are largely
  decorrelated, so a simple late fusion of the two scores reliably outperforms
  either alone (this is the well-established "ensemble of complementary
  encoders" trick from open-vocabulary detection).

Why a re-scoring wrapper, not a detector rewrite:
  The underlying detector already returns its raw scoring components
  (tomato_sim, neg_sim, dino_sim, pred_iou) per detection. We re-fuse with
  SigLIP cosine into a single float and re-sort. The detector itself stays
  pure DINOv2+SAM2 and remains useful as an ablation baseline (paper table).

Score formula (Phase 1.3 fixed weights — Phase 2.2 will train an MLP on these):
  siglip_sim = max_p cos(siglip_img, prompt_pos_p) - max_n cos(siglip_img, prompt_neg_n)
  score = w_dino * dino_sim + w_siglip * siglip_sim + w_pred_iou * pred_iou

  Default weights (0.4, 0.4, 0.2) follow the plan; tune via CLI in Phase 2.2.

Cost:
  ~30 surviving detections per image × one SigLIP image-encoder forward at
  224x224 ≈ ~300 ms/image on NucBox CPU. Negligible vs the detector's ~18 s.

Sprint 4: Phase 1.3 — drop-in multi-modal fusion gate experiment.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Defaults match the plan's prompt set. Override per-call via the constructor.
_DEFAULT_POSITIVE_PROMPTS = (
    "a photograph of a ripe red tomato on a vine",
    "a photograph of a ripe yellow tomato",
    "a green unripe tomato",
)
_DEFAULT_NEGATIVE_PROMPTS = (
    "a green leaf",
    "a stem",
    "soil",
    "a wooden post",
)
_DEFAULT_MODEL_ID = "google/siglip-base-patch16-224"


class _DetectorLike(Protocol):
    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]: ...


def _select_device() -> torch.device:
    import os
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_tensor(x):
    """Recent transformers versions wrap get_*_features outputs in a
    BaseModelOutputWithPooling instead of returning a raw Tensor. Unwrap by
    preferring `pooler_output`, then `last_hidden_state[:, 0]`, then a
    last-resort attribute scan. Tensors pass through unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
        return x.last_hidden_state[:, 0]
    for attr in ("text_embeds", "image_embeds", "embeds"):
        if hasattr(x, attr) and getattr(x, attr) is not None:
            return getattr(x, attr)
    raise TypeError(f"Cannot extract tensor from SigLIP output of type {type(x).__name__}")


def _box_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Same call signature as sam2_amg_detector._nms for bit-identical behaviour."""
    import cv2
    if len(detections) <= 1 or iou_threshold <= 0:
        return detections
    boxes = np.array([d["box"] for d in detections], dtype=np.float32)
    scores = np.array([d["score"] for d in detections], dtype=np.float32)
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = boxes[:, 0]
    xywh[:, 1] = boxes[:, 1]
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    indices = cv2.dnn.NMSBoxes(
        xywh.tolist(), scores.tolist(),
        score_threshold=0.0, nms_threshold=iou_threshold,
    )
    if len(indices) == 0:
        return detections
    kept = np.asarray(indices).flatten()
    return [detections[int(i)] for i in kept]


class SigLIPRescoringWrapper:
    """Wraps a detector to add SigLIP global-image scoring per detection.

    The underlying detector's `score` is replaced by:
      score = w_dino * dino_sim + w_siglip * siglip_sim + w_pred_iou * pred_iou

    Requires the inner detector to attach raw scoring components per detection
    (`dino_sim`, `pred_iou`). sam2_amg_detector.SAM2AMGDetector does this as of
    Sprint 4 Phase 1.3.

    Args:
        base: any detector with .detect(preprocessed_chw) -> list[dict].
        model_id: HuggingFace SigLIP model. Default `google/siglip-base-patch16-224`
            (~370 MB; first run downloads to ~/.cache/huggingface/).
        positive_prompts: text descriptions of the target object. Cosine to
            these is the SigLIP "positive" similarity (max-over-prompts).
        negative_prompts: text descriptions of distractors (leaves, soil, etc.).
            Cosine to these is the SigLIP "negative" similarity (max-over-prompts).
            siglip_sim = pos_max - neg_max. Set to () to disable the negative term.
        w_dino, w_siglip, w_pred_iou: late-fusion weights. Phase 1.3 default is
            (0.4, 0.4, 0.2); Phase 2.2 will replace this with a trained MLP.
        nms_iou_threshold, max_detections: applied AFTER re-scoring so the
            final ranked set reflects the new fusion, not the original DINOv2 order.
        confidence_threshold: drop detections with new score below this. Use
            0.0 to keep all and let downstream sweep tune.
        min_crop_px: skip SigLIP on crops smaller than this (pixels per side).
            Tiny crops produce uninformative SigLIP embeddings; we keep the
            inner detector's score unchanged for them.
    """

    def __init__(
        self,
        base: _DetectorLike,
        model_id: str = _DEFAULT_MODEL_ID,
        positive_prompts: tuple[str, ...] = _DEFAULT_POSITIVE_PROMPTS,
        negative_prompts: tuple[str, ...] = _DEFAULT_NEGATIVE_PROMPTS,
        w_dino: float = 0.4,
        w_siglip: float = 0.4,
        w_pred_iou: float = 0.2,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 30,
        confidence_threshold: float = 0.0,
        min_crop_px: int = 16,
        device: Optional[torch.device] = None,
    ) -> None:
        self._base = base
        self._w_dino = w_dino
        self._w_siglip = w_siglip
        self._w_pred_iou = w_pred_iou
        self._nms_iou = nms_iou_threshold
        self._max_detections = max_detections
        self._conf_threshold = confidence_threshold
        self._min_crop_px = min_crop_px
        self._device = device or _select_device()
        self._positive_prompts = positive_prompts
        self._negative_prompts = negative_prompts

        logger.info("Loading SigLIP model %s on %s...", model_id, self._device)
        from transformers import AutoModel, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).eval().to(self._device)

        # Pre-encode text prompts once. SigLIP's text branch is heavier than
        # the image branch; doing this per-image would dominate latency.
        with torch.no_grad():
            text_inputs = self._processor(
                text=list(positive_prompts) + list(negative_prompts),
                padding="max_length",
                return_tensors="pt",
            )
            text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}
            text_feats = self._model.get_text_features(**text_inputs)
        self._text_emb = F.normalize(_as_tensor(text_feats), dim=1)  # (n_pos+n_neg, D)
        self._n_pos = len(positive_prompts)
        self._n_neg = len(negative_prompts)
        logger.info(
            "SigLIP cached %d positive + %d negative text embeddings (D=%d).",
            self._n_pos, self._n_neg, self._text_emb.shape[1],
        )

    def _reconstruct_rgb(self, preprocessed_chw: np.ndarray) -> np.ndarray:
        """Invert the ImageNet normalisation to get a uint8 HWC RGB image."""
        rgb_float = (
            preprocessed_chw * _IMAGENET_STD[:, None, None]
            + _IMAGENET_MEAN[:, None, None]
        )
        rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
        return np.transpose(rgb_uint8, (1, 2, 0))  # CHW -> HWC

    def _siglip_scores(
        self,
        rgb_hwc: np.ndarray,
        boxes: list[list[float]],
    ) -> np.ndarray:
        """Compute siglip_sim = pos_max - neg_max for each box's crop."""
        from PIL import Image
        H, W = rgb_hwc.shape[:2]
        crops = []
        valid_idx = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1i = max(0, int(round(x1)))
            y1i = max(0, int(round(y1)))
            x2i = min(W, int(round(x2)))
            y2i = min(H, int(round(y2)))
            if x2i - x1i < self._min_crop_px or y2i - y1i < self._min_crop_px:
                continue
            crops.append(Image.fromarray(rgb_hwc[y1i:y2i, x1i:x2i]))
            valid_idx.append(i)

        sims = np.zeros(len(boxes), dtype=np.float32)
        if not crops:
            return sims

        with torch.no_grad():
            img_inputs = self._processor(images=crops, return_tensors="pt")
            img_inputs = {k: v.to(self._device) for k, v in img_inputs.items()}
            img_feats = self._model.get_image_features(**img_inputs)
        img_norm = F.normalize(_as_tensor(img_feats), dim=1)  # (M, D)

        all_sims = (img_norm @ self._text_emb.T).cpu().numpy()  # (M, n_pos+n_neg)
        pos_max = all_sims[:, : self._n_pos].max(axis=1)
        neg_max = (
            all_sims[:, self._n_pos:].max(axis=1)
            if self._n_neg > 0
            else np.zeros(len(crops), dtype=np.float32)
        )
        contrastive = pos_max - neg_max

        for j, i in enumerate(valid_idx):
            sims[i] = contrastive[j]
        return sims

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        dets = self._base.detect(preprocessed_chw)
        if not dets:
            return dets

        # Defensive: detector must attach raw components. If they're missing,
        # treat the existing score as dino_sim and pred_iou=0 — degrades to a
        # weighted sum of (existing_score, siglip) which is still useful but
        # not the intended Phase 1.3 formula.
        if "dino_sim" not in dets[0]:
            logger.warning(
                "Inner detector does not attach raw scoring components. "
                "SigLIPRescoringWrapper falls back to fusing existing 'score' as dino_sim."
            )
            for d in dets:
                d.setdefault("dino_sim", float(d["score"]))
                d.setdefault("pred_iou", 0.0)

        rgb_hwc = self._reconstruct_rgb(preprocessed_chw)
        boxes = [list(d["box"]) for d in dets]
        siglip_sims = self._siglip_scores(rgb_hwc, boxes)

        for d, s in zip(dets, siglip_sims):
            d["siglip_sim"] = float(s)
            d["score"] = (
                self._w_dino * d["dino_sim"]
                + self._w_siglip * d["siglip_sim"]
                + self._w_pred_iou * d["pred_iou"]
            )

        if self._conf_threshold > 0:
            dets = [d for d in dets if d["score"] >= self._conf_threshold]
            if not dets:
                return []

        dets = _box_nms(dets, self._nms_iou)
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets[: self._max_detections]
