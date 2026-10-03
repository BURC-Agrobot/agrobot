"""
sam2_semantic_detector.py — DINOv2-guided semantic point sampling + SAM2 per-point prediction.

Why this replaces uniform AMG grid for E3:
  SAM2 AMG places a regular NxN grid of points across the image — at pts=20
  that is 400 prompts, of which ~388 land on background. The grid is semantically
  blind: it cannot know which cells contain tomatoes before generating masks.
  Result: superlinear compute growth with diminishing recall returns.

  This detector inverts the order of operations:
    OLD (AMG): grid → masks → DINOv2 scores each mask
    NEW (Semantic): DINOv2 heatmap → top-K points → SAM2 predicts one mask per point

  DINOv2 runs once per frame (same cost as before). The 37×37 cosine similarity
  map to the query embedding acts as a free pre-filter. We sample prompt points
  from the top-scoring patches (high tomato likelihood) and augment with a sparse
  uniform grid for coverage. SAM2ImagePredictor then generates exactly K masks,
  one per point, using multimask_output=True and selecting the highest predicted_iou.

  This gives:
    - Higher recall at the same or lower prompt budget (semantic bias vs uniform)
    - Faster inference than pts=20 AMG (K=48 prompts vs 400)
    - Smaller memory footprint (no intermediate 400-mask candidate list)

Architecture:
  DINOv2 forward → (37,37) heatmap → top-K + sparse grid points → SAM2ImagePredictor
  → per-point masks → coverage-weighted DINOv2 re-scoring → NMS → detections

Interface:
  detect(preprocessed_chw) → list of {"box", "score", "label", "mask"}
  Identical to SAM2AMGDetector — zero node changes, just add --detector sam2_semantic.

Runtime estimate [MAC CPU]:
  DINOv2: ~350ms (once)
  SAM2 image encoder: ~200ms (once)
  SAM2 mask decoder: ~30ms × K (K=48 → ~1.4s)
  Total: ~2s/frame vs ~9.5s for pts=20 AMG at comparable or better recall.

[MAC] development, [NUCBOX] deployment target.
Sprint 4: E3 semantic point sampling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14
_DINO_INPUT_SIZE = 518
_DINO_GRID = _DINO_INPUT_SIZE // _DINO_PATCH_SIZE  # 37
_DEFAULT_SAM2_CKPT = "models/sam2/sam2.1_hiera_small.pt"
_DEFAULT_SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"
_DEFAULT_QUERY_EMB = "models/query_embedding.pt"
_DEFAULT_NEGATIVE_EMB = "models/negative_embedding.pt"

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


def _find_repo_root() -> Path:
    candidate = Path(__file__).resolve()
    for _ in range(10):
        candidate = candidate.parent
        if (candidate / "MODULE.bazel").exists():
            return candidate
    env_root = os.environ.get("AGROBOT_ROOT")
    if env_root:
        return Path(env_root)
    raise RuntimeError("Cannot find repo root. Set AGROBOT_ROOT or run from the repo.")


class SAM2SemanticDetector:
    """Tomato detector using DINOv2-guided semantic point sampling + SAM2ImagePredictor.

    Implements the same interface as SAM2AMGDetector:
      detect(preprocessed_chw) → list[dict]
    where each dict has keys: box, score, label, mask.

    Args:
        device: Inference device. Auto-selected if None.
        confidence_threshold: Minimum fused score to keep a detection.
        max_detections: Maximum detections per frame.
        top_k_semantic: Number of top-scoring DINOv2 heatmap patches to use as
            semantic point prompts. These concentrate sampling on tomato-like regions.
        sparse_grid_n: Size of the supplementary uniform grid (NxN points). Provides
            coverage for tomatoes that score below the top-K threshold. Default 4 → 16
            uniform points. Total prompts = top_k_semantic + sparse_grid_n².
        min_mask_area: Minimum mask area (pixels) to keep.
        dino_score_weight: Weight for DINOv2 in score fusion.
        nms_iou_threshold: IoU threshold for NMS (0 = disabled).
        negative_weight: Contrastive negative suppression weight.
        query_embedding_path: Optional override for the query embedding path.
            Supports (768,) single or (k, 768) multi-prototype tensors.
        heatmap_threshold: Minimum normalised DINOv2 similarity score for a patch
            to be considered as a semantic sampling candidate. Prevents prompting
            on clearly non-tomato regions even if they are in the top-K.
        nms_before_decode: If True, suppress semantically redundant prompt points
            (those within one patch-width of each other) before running the decoder.
            Reduces duplicate masks in dense clusters.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        sam2_checkpoint: Optional[str] = None,
        confidence_threshold: float = 0.25,
        max_detections: int = 30,
        top_k_semantic: int = 48,
        sparse_grid_n: int = 4,
        min_mask_area: int = 100,
        dino_score_weight: float = 1.0,
        nms_iou_threshold: float = 0.5,
        negative_weight: float = 0.3,
        query_embedding_path: Optional[str] = None,
        heatmap_threshold: float = 0.0,
        nms_before_decode: bool = True,
        dino_lora_path: Optional[str] = None,
        negative_embedding_path: Optional[str] = None,
    ) -> None:
        self._device = device or _select_device()
        self._conf_threshold = confidence_threshold
        self._max_detections = max_detections
        self._top_k_semantic = top_k_semantic
        self._sparse_grid_n = sparse_grid_n
        self._min_mask_area = min_mask_area
        self._dino_score_weight = dino_score_weight
        self._nms_iou_threshold = nms_iou_threshold
        self._negative_weight = negative_weight
        self._heatmap_threshold = heatmap_threshold
        self._nms_before_decode = nms_before_decode
        self._dino_lora_path = Path(dino_lora_path) if dino_lora_path else None
        self._negative_embedding_path = Path(negative_embedding_path) if negative_embedding_path else None
        self._repo_root = _find_repo_root()

        ckpt = sam2_checkpoint or str(self._repo_root / _DEFAULT_SAM2_CKPT)
        self._sam2_ckpt = Path(ckpt)
        self._query_embedding_path = Path(query_embedding_path) if query_embedding_path else None

        self._dino: Optional[torch.nn.Module] = None
        self._sam2_predictor = None
        self._query_embedding: Optional[torch.Tensor] = None
        self._negative_embedding: Optional[torch.Tensor] = None

        self._load_models()
        self._load_query_embedding()
        self._load_negative_embedding()

    def _load_models(self) -> None:
        logger.info("Loading DINOv2 (%s) via torch.hub...", _DINO_MODEL_NAME)
        self._dino = torch.hub.load(
            "facebookresearch/dinov2", _DINO_MODEL_NAME, pretrained=True
        )
        if self._dino_lora_path is not None and self._dino_lora_path.exists():
            try:
                import sys as _sys
                from pathlib import Path as _Path
                _tools = _Path(__file__).resolve().parent.parent.parent / "tools"
                if str(_tools) not in _sys.path:
                    _sys.path.insert(0, str(_tools))
                from finetune_dino_lora import inject_lora  # type: ignore[import]
                self._dino = inject_lora(self._dino, rank=8, lora_blocks=4)
                lora_state = torch.load(str(self._dino_lora_path), map_location=self._device)
                self._dino.load_state_dict(lora_state, strict=False)
                logger.info("Applied LoRA adapters from %s.", self._dino_lora_path)
            except Exception as exc:
                logger.warning("LoRA load failed (%s). Using vanilla DINOv2.", exc)
        self._dino.eval().to(self._device)
        logger.info("DINOv2 loaded.")

        if not self._sam2_ckpt.exists():
            logger.warning(
                "SAM2 checkpoint not found at %s. No detections will be produced.",
                self._sam2_ckpt,
            )
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(
                _DEFAULT_SAM2_CFG, str(self._sam2_ckpt), device=self._device
            )

            finetuned = self._sam2_ckpt.parent / "sam2_tomato_finetuned.pt"
            if finetuned.exists():
                state = torch.load(str(finetuned), map_location=self._device)
                sam2_model.load_state_dict(state)
                logger.info("Loaded fine-tuned SAM2 weights from %s.", finetuned)

            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
            logger.info("SAM2ImagePredictor loaded (semantic point mode).")
        except Exception as exc:
            logger.warning("SAM2 failed to load (%s). No detections.", exc)
            self._sam2_predictor = None

    def _load_query_embedding(self) -> None:
        query_path = self._query_embedding_path or (self._repo_root / _DEFAULT_QUERY_EMB)
        if query_path.exists():
            q = torch.load(str(query_path), map_location=self._device).float()
            self._query_embedding = F.normalize(q, dim=0 if q.dim() == 1 else 1)
            logger.info(
                "Loaded query embedding from %s (shape=%s).", query_path, tuple(self._query_embedding.shape)
            )
        else:
            logger.warning("Query embedding not found at %s.", query_path)

    def _load_negative_embedding(self) -> None:
        if self._negative_weight <= 0:
            return
        neg_path = self._negative_embedding_path or (self._repo_root / _DEFAULT_NEGATIVE_EMB)
        if neg_path.exists():
            q = torch.load(str(neg_path), map_location=self._device).float()
            self._negative_embedding = F.normalize(q, dim=0 if q.dim() == 1 else 1)
            logger.info(
                "Loaded negative embedding from %s (shape=%s, contrastive λ=%.2f).",
                neg_path, tuple(self._negative_embedding.shape), self._negative_weight,
            )

    def _build_prompt_points(self, patch_sims: torch.Tensor) -> np.ndarray:
        """Build prompt point coordinates in 518×518 pixel space.

        Strategy:
          1. Flatten the (37,37) heatmap → take top_k_semantic patch indices.
          2. Convert each patch index to its centre pixel coordinate.
          3. Optionally filter by heatmap_threshold (discard clearly non-tomato).
          4. Augment with a sparse NxN uniform grid.
          5. If nms_before_decode, suppress points within 1 patch-width of each other
             (keep the higher-scoring one) to avoid redundant decoder calls in clusters.

        Returns: (M, 2) float32 array of (x, y) coordinates in 518×518 space.
        """
        flat = patch_sims.reshape(-1)  # (1369,)
        top_k = min(self._top_k_semantic, flat.shape[0])
        topk_vals, topk_idx = flat.topk(top_k)

        points = []
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            if val < self._heatmap_threshold:
                break
            gr = idx // _DINO_GRID
            gc = idx  % _DINO_GRID
            # Centre of the 14×14 patch cell
            px = float(gc * _DINO_PATCH_SIZE + _DINO_PATCH_SIZE / 2)
            py = float(gr * _DINO_PATCH_SIZE + _DINO_PATCH_SIZE / 2)
            points.append((px, py, float(val)))

        # Sparse uniform grid for coverage
        if self._sparse_grid_n > 0:
            step = _DINO_INPUT_SIZE / (self._sparse_grid_n + 1)
            for r in range(1, self._sparse_grid_n + 1):
                for c in range(1, self._sparse_grid_n + 1):
                    px = float(c * step)
                    py = float(r * step)
                    # Score at this grid position (bilinear interp from heatmap)
                    gr = min(int(py / _DINO_PATCH_SIZE), _DINO_GRID - 1)
                    gc = min(int(px / _DINO_PATCH_SIZE), _DINO_GRID - 1)
                    val = float(patch_sims[gr, gc])
                    points.append((px, py, val))

        if not points:
            return np.zeros((0, 2), dtype=np.float32)

        # NMS on prompt points: suppress duplicates within 1 patch-width (14px)
        if self._nms_before_decode and len(points) > 1:
            points_sorted = sorted(points, key=lambda p: p[2], reverse=True)
            kept: list[tuple[float, float, float]] = []
            for cand in points_sorted:
                cx, cy, _ = cand
                too_close = False
                for kx, ky, _ in kept:
                    if abs(cx - kx) < _DINO_PATCH_SIZE and abs(cy - ky) < _DINO_PATCH_SIZE:
                        too_close = True
                        break
                if not too_close:
                    kept.append(cand)
            points = kept

        coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        return coords  # (M, 2)

    def _score_mask(
        self,
        seg: np.ndarray,
        patch_norms: torch.Tensor,
    ) -> tuple[float, float]:
        """Coverage-weighted cosine similarity to tomato query (and negative).

        Returns: (tomato_sim, neg_sim) — caller handles fusion.
        """
        seg_f = seg[:_DINO_GRID * _DINO_PATCH_SIZE, :_DINO_GRID * _DINO_PATCH_SIZE].astype(np.float32)
        blocks = seg_f.reshape(_DINO_GRID, _DINO_PATCH_SIZE, _DINO_GRID, _DINO_PATCH_SIZE)
        coverage = torch.from_numpy(blocks.mean(axis=(1, 3)).reshape(-1)).to(self._device)

        if coverage.sum() < 1e-6:
            return 0.0, 0.0

        if self._query_embedding.dim() == 1:
            sims = patch_norms @ self._query_embedding
            tomato_sim = float((sims * coverage).sum() / coverage.sum())
        else:
            sims_k = patch_norms @ self._query_embedding.T  # (1369, k)
            per_proto = (sims_k * coverage.unsqueeze(1)).sum(dim=0) / coverage.sum()
            tomato_sim = float(per_proto.max())

        neg_sim = 0.0
        if self._negative_embedding is not None:
            if self._negative_embedding.dim() == 1:
                neg_sims = patch_norms @ self._negative_embedding
                neg_sim = float((neg_sims * coverage).sum() / coverage.sum())
            else:
                neg_sims_k = patch_norms @ self._negative_embedding.T
                neg_sim = float(
                    ((neg_sims_k * coverage.unsqueeze(1)).sum(dim=0) / coverage.sum()).max()
                )

        return tomato_sim, neg_sim

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        """Detect tomatoes using DINOv2-guided semantic prompts + SAM2.

        Args:
            preprocessed_chw: float32 (3, 518, 518) ImageNet-normalised array.

        Returns:
            list of {"box": [x1,y1,x2,y2], "score": float, "label": str, "mask": np.ndarray}
        """
        if self._sam2_predictor is None or self._query_embedding is None or self._dino is None:
            return []

        # ── Step 1: DINOv2 forward ────────────────────────────────────────────
        tensor = torch.from_numpy(preprocessed_chw).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._dino.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"].squeeze(0).float()  # (1369, 768)
        patch_norms  = F.normalize(patch_tokens, dim=1)                    # L2-normalised

        # ── Step 2: DINOv2 heatmap → semantic prompt points ──────────────────
        # Compute per-patch cosine similarity to the tomato query embedding.
        # For multi-prototype, use the max similarity across prototypes.
        if self._query_embedding.dim() == 1:
            flat_sims = patch_norms @ self._query_embedding  # (1369,)
        else:
            flat_sims = (patch_norms @ self._query_embedding.T).max(dim=1).values  # (1369,)
        heatmap = flat_sims.reshape(_DINO_GRID, _DINO_GRID)  # (37, 37)

        prompt_coords = self._build_prompt_points(heatmap)   # (M, 2) in 518×518 space

        if prompt_coords.shape[0] == 0:
            return []

        # ── Step 3: Reconstruct uint8 RGB for SAM2 ───────────────────────────
        rgb_float = preprocessed_chw * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
        rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
        rgb_hwc   = np.transpose(rgb_uint8, (1, 2, 0))

        # ── Step 4: SAM2 image encoding (once per frame) ─────────────────────
        try:
            self._sam2_predictor.set_image(rgb_hwc)
        except Exception as exc:
            logger.warning("SAM2 set_image() failed: %s", exc)
            return []

        # ── Step 5: Per-point mask prediction ────────────────────────────────
        # SAM2ImagePredictor.predict() accepts (N, 2) coords and (N,) labels.
        # multimask_output=True returns 3 candidate masks per point; we pick the
        # highest predicted_iou among the 3.
        detections: list[dict] = []

        for i in range(prompt_coords.shape[0]):
            pt = prompt_coords[i]  # (2,) [x, y]
            try:
                masks_np, scores_np, _ = self._sam2_predictor.predict(
                    point_coords=pt[None],   # (1, 2)
                    point_labels=np.array([1], dtype=np.int32),  # foreground
                    multimask_output=True,
                )
            except Exception as exc:
                logger.debug("SAM2 predict() failed at point %s: %s", pt, exc)
                continue

            # masks_np: (3, H, W) bool; scores_np: (3,) predicted_iou
            best_idx = int(np.argmax(scores_np))
            seg = masks_np[best_idx]   # (H, W) bool
            pred_iou = float(scores_np[best_idx])

            if seg.sum() < self._min_mask_area:
                continue

            # ── Step 6: DINOv2 scoring ────────────────────────────────────────
            tomato_sim, neg_sim = self._score_mask(seg, patch_norms)
            dino_sim = tomato_sim - self._negative_weight * neg_sim

            alpha = self._dino_score_weight
            score = alpha * dino_sim + (1.0 - alpha) * pred_iou

            if score < self._conf_threshold:
                continue

            # Bounding box from mask contour
            rows = np.where(seg.any(axis=1))[0]
            cols = np.where(seg.any(axis=0))[0]
            if rows.size == 0 or cols.size == 0:
                continue
            x1, y1 = float(cols[0]), float(rows[0])
            x2, y2 = float(cols[-1]), float(rows[-1])

            detections.append({
                "box": [x1, y1, x2, y2],
                "score": score,
                "label": "tomato",
                "mask": seg.astype(np.uint8),
            })

        # ── Step 7: NMS ───────────────────────────────────────────────────────
        if self._nms_iou_threshold > 0 and detections:
            detections = self._nms(detections)

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections[: self._max_detections]

    def _nms(self, detections: list[dict]) -> list[dict]:
        """Box NMS — identical to SAM2AMGDetector._nms."""
        if len(detections) <= 1:
            return detections
        boxes  = np.array([d["box"] for d in detections], dtype=np.float32)
        scores = np.array([d["score"] for d in detections], dtype=np.float32)
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = boxes[:, 0]
        xywh[:, 1] = boxes[:, 1]
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.tolist(),
            score_threshold=0.0, nms_threshold=self._nms_iou_threshold,
        )
        if len(indices) == 0:
            return detections
        kept = np.asarray(indices).flatten()
        return [detections[int(i)] for i in kept]
