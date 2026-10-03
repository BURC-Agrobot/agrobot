"""
sam2_amg_detector.py — SAM2 Automatic Mask Generator + DINOv2 semantic scoring.

Architecture lineage:
  Sprint 1: PlaceholderDetector (colour threshold only)
  Sprint 2: DINOv2 proposals → SAM2 refinement  [abandoned: 14px grid quantisation]
  Sprint 3: SAM2 AMG proposals → DINOv2 scoring  [mAP 0 → 0.023 → 0.170]
  Sprint 4: All of the following improvements applied to this detector.

Sprint 4 improvements (vs Sprint 3 baseline mAP=0.170):
  E1 — Coverage-weighted scoring: hard 0.5 threshold replaced with float coverage
       weights. Boundary patches of circular tomato masks now contribute
       proportionally instead of being silently excluded.
  E2 — Multi-prototype query: k-means (k=4) over training patches produces one
       centroid per ripeness stage (green/yellow/orange-red/occluded). Scoring
       uses max-over-k instead of a single mean that represents no mode well.
  E7 — Quadrant-crop proposals: optional 4-crop multi-scale AMG recovers small/
       distant tomatoes below the full-image grid resolution threshold (26px at
       pts=20). Enable with use_quadrant_crops=True.
  E6 — LoRA DINOv2: optional LoRA adapter weights (from finetune_dino_lora.py)
       shift DINOv2 feature space toward tomato vs background discrimination
       without catastrophic forgetting. Load via dino_lora_path.
  E8 — MIGraphX inference: optional MIGraphX-compiled .mxr replaces the PyTorch
       DINOv2 forward (~350ms→~20ms on NucBox GPU). Load via migraphx_dino_path.
       Blocked on ROCm gfx1151 — see docs/SPRINT3_ROCM_ISSUE.md.

Scoring pipeline (per frame):
  1. SAM2 AMG generates N pixel-precise mask proposals from a uniform grid.
  2. [Optional] 4 quadrant-crop AMGs generate additional proposals for small objects.
  3. DINOv2 runs once → (1369, 768) L2-normalised patch token matrix.
  4. Per mask: coverage-weighted cosine similarity to k prototype vectors → max_k.
  5. Contrastive term: subtract λ × coverage-weighted similarity to negative.
  6. Fuse: α×dino_sim + (1-α)×pred_iou. Filter by confidence_threshold. NMS.

Interface:
  detect(preprocessed_chw) → list of {"box", "score", "label", "mask"}
  Identical to DINOv2SAM2Detector and SAM2SemanticDetector — zero node changes.

Runtime [NUCBOX CPU, pts=20]:
  SAM2 encoder:  ~200 ms (amortised over all masks)
  SAM2 decoder:  ~30 ms × N masks (batched)
  DINOv2:        ~350 ms (once per frame; ~20 ms with MIGraphX GPU)
  Scoring:       ~5 ms
  Total:         ~9–10 s/frame CPU. <300 ms target with MIGraphX (Sprint 4 E8).

[MAC] development, [NUCBOX] deployment.
Sprint 4: E1 coverage scoring, E2 multi-prototype, E7 quadrant crops, E6/E8 GPU path.
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


class SAM2AMGDetector:
    """Tomato detector using SAM2 AMG proposals scored by DINOv2 similarity.

    Implements the same interface as DINOv2SAM2Detector and PlaceholderDetector:
      detect(preprocessed_chw) → list[dict]
    where each dict has keys: box, score, label, mask.

    Args:
        device: Inference device. Auto-selected if None.
        confidence_threshold: Minimum fused score to keep a mask.
        max_detections: Maximum number of detections per frame.
        points_per_side: SAM2 AMG grid density. 8 → 64 masks, 16 → 256 masks.
            Higher = better recall, slower. Use 8 for CPU, 16+ for GPU.
        min_mask_area: Minimum mask area in pixels (518×518 space) to keep.
            Filters out tiny spurious SAM2 masks.
        dino_score_weight: Weight for DINOv2 in fusion. score = α*dino + (1-α)*pred_iou.
            1.0 = DINOv2 only (default; fusion needs per-dataset tuning).
        nms_iou_threshold: IoU threshold for NMS. 0 = disabled (default).
            Enable with 0.5+ to suppress duplicates; tune per-dataset.
        negative_weight: Weight for contrastive negative term. 0 = disabled.
            When negative_embedding.pt exists, score = tomato_sim - λ*negative_sim.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        sam2_checkpoint: Optional[str] = None,
        confidence_threshold: float = 0.3,
        max_detections: int = 20,
        points_per_side: int = 8,
        min_mask_area: int = 100,
        dino_score_weight: float = 1.0,
        nms_iou_threshold: float = 0.0,
        negative_weight: float = 0.3,
        query_embedding_path: Optional[str] = None,
        use_quadrant_crops: bool = False,
        crop_overlap: float = 0.25,
        dino_lora_path: Optional[str] = None,
        migraphx_dino_path: Optional[str] = None,
        negative_embedding_path: Optional[str] = None,
    ) -> None:
        self._device = device or _select_device()
        self._conf_threshold = confidence_threshold
        self._max_detections = max_detections
        self._points_per_side = points_per_side
        self._min_mask_area = min_mask_area
        self._dino_score_weight = dino_score_weight
        self._nms_iou_threshold = nms_iou_threshold
        self._negative_weight = negative_weight
        self._repo_root = _find_repo_root()

        self._use_quadrant_crops = use_quadrant_crops
        self._crop_overlap = crop_overlap
        self._dino_lora_path = Path(dino_lora_path) if dino_lora_path else None
        self._migraphx_dino_path = Path(migraphx_dino_path) if migraphx_dino_path else None
        self._migraphx_prog = None
        self._negative_embedding_path = Path(negative_embedding_path) if negative_embedding_path else None

        ckpt = sam2_checkpoint or str(self._repo_root / _DEFAULT_SAM2_CKPT)
        self._sam2_ckpt = Path(ckpt)
        self._query_embedding_path = Path(query_embedding_path) if query_embedding_path else None

        self._dino: Optional[torch.nn.Module] = None
        self._amg = None
        self._query_embedding: Optional[torch.Tensor] = None
        self._negative_embedding: Optional[torch.Tensor] = None

        self._load_models()
        self._load_query_embedding()
        self._load_negative_embedding()

    def _load_models(self) -> None:
        # ── DINOv2 — MIGraphX path (NucBox GPU) or PyTorch fallback ──────────
        if self._migraphx_dino_path is not None and self._migraphx_dino_path.exists():
            try:
                import migraphx  # type: ignore[import]
                self._migraphx_prog = migraphx.load(str(self._migraphx_dino_path))
                logger.info("Loaded MIGraphX compiled DINOv2 from %s.", self._migraphx_dino_path)
                # Still need PyTorch DINOv2 for query embedding builds; for inference
                # the MIGraphX path short-circuits the PyTorch forward in detect().
            except Exception as exc:
                logger.warning("MIGraphX load failed (%s). Falling back to PyTorch DINOv2.", exc)
                self._migraphx_prog = None

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

        # ── SAM2 AMG ──────────────────────────────────────────────────────────
        if not self._sam2_ckpt.exists():
            logger.warning(
                "SAM2 checkpoint not found at %s. "
                "SAM2 AMG disabled — no detections will be produced.",
                self._sam2_ckpt,
            )
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            sam2_model = build_sam2(
                _DEFAULT_SAM2_CFG, str(self._sam2_ckpt), device=self._device
            )

            # Load fine-tuned decoder weights if present alongside base checkpoint.
            finetuned = self._sam2_ckpt.parent / "sam2_tomato_finetuned.pt"
            if finetuned.exists():
                state = torch.load(str(finetuned), map_location=self._device)
                sam2_model.load_state_dict(state)
                logger.info("Loaded fine-tuned SAM2 weights from %s.", finetuned)

            self._amg = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=self._points_per_side,
                # Keep only confident, stable masks to reduce noise.
                pred_iou_thresh=0.70,
                stability_score_thresh=0.80,
                # Crop-based multi-scale augmentation — disable for speed on CPU.
                crop_n_layers=0,
                # Post-process: remove masks with <min_mask_area pixels.
                min_mask_region_area=self._min_mask_area,
            )
            logger.info(
                "SAM2 AMG loaded (points_per_side=%d).", self._points_per_side
            )
        except Exception as exc:
            logger.warning("SAM2 AMG failed to load (%s). No detections.", exc)
            self._amg = None

    def _load_query_embedding(self) -> None:
        query_path = self._query_embedding_path or (self._repo_root / _DEFAULT_QUERY_EMB)
        if query_path.exists():
            q = torch.load(str(query_path), map_location=self._device).float()
            if q.dim() == 1:
                # Single mean embedding (768,) — normalise and keep as-is
                self._query_embedding = F.normalize(q, dim=0)
            else:
                # Multi-prototype (k, 768) — each row already normalised by builder;
                # re-normalise defensively in case the file was built without it.
                self._query_embedding = F.normalize(q, dim=1)
            logger.info(
                "Loaded query embedding from %s (shape=%s).", query_path, tuple(self._query_embedding.shape)
            )
        else:
            logger.warning(
                "Query embedding not found at %s. "
                "Run perception/tools/build_query_embedding.py first.",
                query_path,
            )

    def _load_negative_embedding(self) -> None:
        if self._negative_weight <= 0:
            return
        neg_path = self._negative_embedding_path or (self._repo_root / _DEFAULT_NEGATIVE_EMB)
        if neg_path.exists():
            q = torch.load(str(neg_path), map_location=self._device).float()
            # dim=1 for (k, 768) multi-prototype tensors; dim=0 for (768,) single vector.
            self._negative_embedding = F.normalize(q, dim=0 if q.dim() == 1 else 1)
            logger.info(
                "Loaded negative embedding from %s (shape=%s, contrastive λ=%.2f).",
                neg_path, tuple(self._negative_embedding.shape), self._negative_weight,
            )
        else:
            logger.debug(
                "Negative embedding not found at %s. Run build_query_embedding.py --output-negative.",
                neg_path,
            )

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        """Detect tomatoes using SAM2 AMG proposals scored by DINOv2.

        Args:
            preprocessed_chw: float32 (3, 518, 518) ImageNet-normalised array
                               from preprocess_for_dino(). Same as DINOv2SAM2Detector.

        Returns:
            list of {"box": [x1,y1,x2,y2], "score": float, "label": str, "mask": np.ndarray}
        """
        if self._amg is None or self._query_embedding is None or self._dino is None:
            return []

        # ── Step 1: Reconstruct uint8 RGB image for SAM2 AMG ─────────────────
        # SAM2 AMG needs a uint8 HWC image, not the normalized CHW tensor.
        rgb_float = preprocessed_chw * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
        rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
        rgb_hwc = np.transpose(rgb_uint8, (1, 2, 0))  # CHW → HWC

        # ── Step 2: DINOv2 forward — patch tokens for scoring ─────────────────
        # Use MIGraphX compiled path on NucBox when available; PyTorch otherwise.
        if self._migraphx_prog is not None:
            try:
                import migraphx  # type: ignore[import]
                inp = preprocessed_chw[None].astype(np.float32)  # (1,3,518,518)
                result = self._migraphx_prog.run({"image": migraphx.argument(inp)})
                # export_dino_onnx.py bakes L2-normalisation into the graph
                patch_norms = torch.from_numpy(np.array(result[0])).squeeze(0).to(self._device)
            except Exception as exc:
                logger.debug("MIGraphX forward failed (%s). Falling back to PyTorch.", exc)
                self._migraphx_prog = None  # disable for subsequent frames
                tensor = torch.from_numpy(preprocessed_chw).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    features = self._dino.forward_features(tensor)
                patch_tokens = features["x_norm_patchtokens"].squeeze(0)
                patch_norms  = F.normalize(patch_tokens, dim=1)
        else:
            tensor = torch.from_numpy(preprocessed_chw).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._dino.forward_features(tensor)
            patch_tokens = features["x_norm_patchtokens"].squeeze(0)  # (1369, 768)
            patch_norms  = F.normalize(patch_tokens, dim=1)            # L2-normalised

        # ── Step 3: SAM2 AMG — generate pixel-precise mask proposals ──────────
        # generate() is already @torch.no_grad internally.
        try:
            masks_data = self._amg.generate(rgb_hwc)
        except Exception as exc:
            logger.warning("SAM2 AMG generate() failed: %s", exc)
            return []

        if not masks_data:
            masks_data = []

        # Optionally augment with quadrant-crop proposals to catch small/distant
        # tomatoes that fall below the full-image grid resolution threshold.
        if self._use_quadrant_crops:
            crop_masks = self._generate_quadrant_masks(rgb_hwc)
            masks_data = masks_data + crop_masks
            logger.debug(
                "AMG: %d full-image + %d quadrant-crop proposals = %d total.",
                len(masks_data) - len(crop_masks), len(crop_masks), len(masks_data),
            )

        if not masks_data:
            return []

        # ── Step 4: Score each mask with DINOv2 patch similarity ──────────────
        detections = []
        for mask_info in masks_data:
            seg = mask_info["segmentation"]   # bool (518, 518)
            bbox_xywh = mask_info["bbox"]     # [x, y, w, h] in 518×518 space

            if seg.sum() < self._min_mask_area:
                continue

            # Downsample mask to the 37×37 DINOv2 patch grid as float coverage
            # weights. Each cell ∈ [0,1] = fraction of the 14×14 block inside
            # the mask. Using float weights (not a hard 0.5 threshold) preserves
            # boundary patches of circular tomatoes that would otherwise be
            # excluded, restoring the full mask footprint in the scoring region.
            coverage = self._mask_to_patch_coverage(seg)  # (37, 37) float32
            weights = coverage.reshape(-1)                 # (1369,) float32

            if weights.sum() < 1e-6:
                continue

            weights = weights.to(self._device)

            # Coverage-weighted cosine similarity.
            # Single prototype (768,):  score = Σ w_i·cos(f_i, q) / Σ w_i
            # Multi-prototype (k, 768): score = max_k [ Σ w_i·cos(f_i, qₖ) / Σ w_i ]
            # The max-over-k picks the ripeness mode the mask best matches, so
            # green tomatoes are no longer penalised against a red-biased centroid.
            if self._query_embedding.dim() == 1:
                sims = patch_norms @ self._query_embedding       # (1369,)
                tomato_sim = float((sims * weights).sum() / weights.sum())
            else:
                sims_k = patch_norms @ self._query_embedding.T   # (1369, k)
                per_proto = (sims_k * weights.unsqueeze(1)).sum(dim=0) / weights.sum()  # (k,)
                tomato_sim = float(per_proto.max())

            # Contrastive: coverage-weighted negative similarity.
            neg_sim = 0.0
            if self._negative_embedding is not None and self._negative_weight > 0:
                if self._negative_embedding.dim() == 1:
                    neg_sims = patch_norms @ self._negative_embedding  # (1369,)
                    neg_sim = float((neg_sims * weights).sum() / weights.sum())
                else:
                    neg_sims_k = patch_norms @ self._negative_embedding.T
                    neg_sim = float(
                        ((neg_sims_k * weights.unsqueeze(1)).sum(dim=0) / weights.sum()).max()
                    )
                dino_sim = tomato_sim - self._negative_weight * neg_sim
            else:
                dino_sim = tomato_sim

            # Fuse with SAM2 predicted_iou when available (mask shape quality).
            pred_iou_raw = mask_info.get("predicted_iou") or mask_info.get("pred_iou")
            pred_iou_val = float(pred_iou_raw) if isinstance(pred_iou_raw, (int, float)) else 0.0
            if isinstance(pred_iou_raw, (int, float)):
                alpha = self._dino_score_weight
                score = alpha * dino_sim + (1.0 - alpha) * pred_iou_val
            else:
                score = dino_sim

            if score < self._conf_threshold:
                continue

            # Box from SAM2's own bbox (pixel-precise, from mask contour).
            x, y, w, h = bbox_xywh
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)

            # Attach raw scoring components so downstream wrappers (TTA, SigLIP
            # rescoring, learned MLP fusion in Phase 2) can re-fuse without
            # recomputing the heavy DINOv2/SAM2 forwards. Existing consumers
            # (visualize.py, metrics.py) read only box/score/label/mask and
            # ignore the extra keys.
            detections.append({
                "box": [x1, y1, x2, y2],
                "score": score,
                "label": "tomato",
                "mask": seg.astype(np.uint8),
                "tomato_sim": float(tomato_sim),
                "neg_sim": float(neg_sim),
                "dino_sim": float(dino_sim),
                "pred_iou": pred_iou_val,
            })

        # NMS to remove overlapping detections for the same tomato.
        if self._nms_iou_threshold > 0 and detections:
            detections = self._nms(detections)

        # Sort by score descending, cap at max_detections.
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections[: self._max_detections]

    def _generate_quadrant_masks(self, rgb_hwc: np.ndarray) -> list[dict]:
        """Run AMG on 4 overlapping quadrant crops and reproject detections.

        Why this helps: at pts=20 (full image 518×518) the grid spacing is
        518/20 = 26px. A tomato smaller than 26px in diameter gets no grid point
        inside it and is never proposed. Splitting the image into quadrants with
        25% overlap and running AMG at the same pts density effectively halves
        the minimum detectable object size without changing the per-image AMG params.

        Each quadrant crop covers 50% of the image in each dimension with
        `crop_overlap` (default 25%) bleed into adjacent quadrants. Detections
        from sub-crops are reprojected to full 518×518 pixel space and
        deduplicated by the caller's NMS.
        """
        H, W = rgb_hwc.shape[:2]  # both 518
        half_h = H // 2
        half_w = W // 2
        overlap_h = int(half_h * self._crop_overlap)
        overlap_w = int(half_w * self._crop_overlap)

        # Four quadrants with overlap bleed
        crop_boxes = [
            (0,              0,              half_w + overlap_w, half_h + overlap_h),  # TL
            (half_w - overlap_w, 0,          W,                  half_h + overlap_h),  # TR
            (0,              half_h - overlap_h, half_w + overlap_w, H),               # BL
            (half_w - overlap_w, half_h - overlap_h, W,          H),                  # BR
        ]

        all_crop_masks: list[dict] = []
        for x1c, y1c, x2c, y2c in crop_boxes:
            crop = rgb_hwc[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            # Resize crop to 518×518 for AMG (maintains point density)
            crop_resized = cv2.resize(crop, (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE))
            crop_h = y2c - y1c
            crop_w = x2c - x1c
            scale_x = crop_w / _DINO_INPUT_SIZE
            scale_y = crop_h / _DINO_INPUT_SIZE

            try:
                crop_masks = self._amg.generate(crop_resized)
            except Exception as exc:
                logger.debug("AMG failed on quadrant crop (%d,%d,%d,%d): %s", x1c, y1c, x2c, y2c, exc)
                continue

            for m in crop_masks:
                seg_crop = m["segmentation"]  # (518, 518) in crop space
                # Reproject segmentation mask to full image 518×518 space
                seg_full = np.zeros((H, W), dtype=bool)
                # Map non-zero pixels from crop-scaled space back to full image
                ys, xs = np.where(seg_crop)
                if xs.size == 0:
                    continue
                # Scale back to crop pixel coords, then offset to full image
                full_xs = np.clip((xs * scale_x + x1c).astype(np.int32), 0, W - 1)
                full_ys = np.clip((ys * scale_y + y1c).astype(np.int32), 0, H - 1)
                seg_full[full_ys, full_xs] = True

                # Recompute bbox in full image space
                cols = np.where(seg_full.any(axis=0))[0]
                rows = np.where(seg_full.any(axis=1))[0]
                if cols.size == 0 or rows.size == 0:
                    continue

                reprojected = dict(m)
                reprojected["segmentation"] = seg_full
                reprojected["bbox"] = [int(cols[0]), int(rows[0]),
                                       int(cols[-1] - cols[0]), int(rows[-1] - rows[0])]
                all_crop_masks.append(reprojected)

        return all_crop_masks

    def _nms(self, detections: list[dict]) -> list[dict]:
        """Apply Non-Maximum Suppression. Keeps highest-scoring per overlap cluster."""
        if len(detections) <= 1:
            return detections
        boxes = np.array([d["box"] for d in detections], dtype=np.float32)
        scores = np.array([d["score"] for d in detections], dtype=np.float32)
        # cv2.dnn.NMSBoxes expects [x, y, w, h]
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = boxes[:, 0]  # x
        xywh[:, 1] = boxes[:, 1]  # y
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=self._nms_iou_threshold,
        )
        if len(indices) == 0:
            return detections
        kept = np.asarray(indices).flatten()
        return [detections[int(i)] for i in kept]

    @staticmethod
    def _mask_to_patch_coverage(seg: np.ndarray) -> torch.Tensor:
        """Downsample a 518×518 bool mask to a (37×37) float coverage map.

        Each cell holds the fraction [0,1] of its 14×14 pixel block that is
        inside the mask. Retaining fractional coverage rather than hard-
        thresholding at 0.5 prevents boundary patches of circular tomato masks
        from being silently excluded — at 37×37 resolution most perimeter
        patches of a round object have 20–50% coverage and would be zeroed by
        a hard threshold, shrinking the effective scoring region to ~60% of the
        mask interior.
        """
        seg_f = seg[:_DINO_GRID * _DINO_PATCH_SIZE, :_DINO_GRID * _DINO_PATCH_SIZE].astype(np.float32)
        blocks = seg_f.reshape(_DINO_GRID, _DINO_PATCH_SIZE, _DINO_GRID, _DINO_PATCH_SIZE)
        patch_coverage = blocks.mean(axis=(1, 3))  # (37, 37), values in [0, 1]
        return torch.from_numpy(patch_coverage)  # float32 tensor
