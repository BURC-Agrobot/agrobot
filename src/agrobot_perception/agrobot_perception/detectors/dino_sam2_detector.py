"""
dino_sam2_detector.py — DINOv2 + SAM2 Zero-Shot Tomato Detector

Target environment : [MAC] CPU/MPS prototype · [EDGE] ROCm (Sprint 3)
Sprint             : 2 (zero-shot prototype)
Sprint 3           : replace CPU torch.hub load with MIGraphX-optimized ONNX graph

Architecture
------------
Zero-shot detection pipeline (no tomato-specific training required):

  1. DINOv2 feature extraction
     - Input : float32 CHW tensor (3, 518, 518) from preprocess_for_dino()
     - Output: patch token grid (37×37 patches, 768-dim each for ViT-B/14)
     - Method: torch.hub loads facebookresearch/dinov2 on first call,
               cached to ~/.cache/torch/hub/

  2. Bounding box proposals via patch similarity
     - Compute cosine similarity of each patch embedding against a
       stored tomato query embedding (mean of red/round patch prototypes)
     - Threshold top-k similar patches → connected components → bounding boxes
     - This is the "zero-shot" step: no labelled tomato data required

  3. SAM2 mask refinement
     - Feed each bounding box as a box prompt to SAM2
     - SAM2 returns a precise pixel-level mask
     - Refine bounding box from mask contour (tighter than the proposal)
     - Score = mean similarity of patches within the refined mask region

  4. Return List[dict] matching PlaceholderDetector interface:
     [{"box": [x1, y1, x2, y2], "score": float, "label": str, "mask": np.ndarray}]

Device strategy
---------------
  Precedence: MPS (Mac host, outside Docker) → CUDA → CPU
  Inside Docker on Mac: always CPU (MPS does not pass through the Linux VM).
  Sprint 3 adds: ROCm via HSA_OVERRIDE_GFX_VERSION=11.5.1 on AMD edge.

Model weights
-------------
  DINOv2 : auto-downloaded by torch.hub to ~/.cache/torch/hub/
  SAM2   : checkpoint must be placed at models/sam2/sam2.1_hiera_small.pt
           Download from: https://github.com/facebookresearch/sam2/releases

Swap point
----------
  Sprint 3: swap torch.hub DINOv2 load for a MIGraphX-compiled ONNX export.
  The detect() interface remains unchanged — only the model loading changes.
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

# Path to SAM2 checkpoint relative to the repo root.
# Resolved at runtime from the AGROBOT_ROOT env var, then falls back to
# a path relative to this file (works inside the bind-mounted container).
_DEFAULT_SAM2_CKPT = "models/sam2/sam2.1_hiera_small.pt"
# Config path is relative to the sam2 package root (not the repo root).
# build_sam2() resolves this via Hydra's pkg://sam2 search path.
_DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
# Data-driven query embedding built by perception/tools/build_query_embedding.py.
# When present, replaces the hardcoded RGB prior below.
_DEFAULT_QUERY_EMBEDDING = "models/query_embedding.pt"

# DINOv2 ViT-B/14: patch_size=14, embed_dim=768, image_size=518 → 37×37 patches.
_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14
_DINO_INPUT_SIZE = 518
_DINO_GRID = _DINO_INPUT_SIZE // _DINO_PATCH_SIZE  # 37

# Tomato appearance prior: tomatoes are round, red-to-orange objects.
# This RGB prototype is used to build the initial query embedding.
# Sprint 2: hardcoded prior. Sprint 3: replace with mean embedding from
# Laboro Tomato training set after fine-tuning.
_TOMATO_RGB_PRIOR = np.array([
    [200, 50, 50],    # ripe red
    [220, 80, 20],    # orange-red
    [180, 40, 40],    # deep red
    [210, 100, 30],   # orange
], dtype=np.float32) / 255.0  # normalise to [0, 1]


def _select_device() -> torch.device:
    """Select the best available inference device.

    MPS is checked first so Mac host runs (outside Docker) use the GPU.
    Inside the Docker container on Mac, MPS is unavailable and CPU is used.

    Set AGROBOT_FORCE_CPU=1 to bypass GPU selection entirely.
    Required on NucBox (gfx1151/Strix Halo) until Sprint 3 MIGraphX path is
    in place — the pre-built ROCm 6.4 PyTorch wheels target discrete RDNA 3
    (gfx1100) and fault on gfx1151 unified memory.
    """
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root (contains MODULE.bazel)."""
    candidate = Path(__file__).resolve()
    for _ in range(10):
        candidate = candidate.parent
        if (candidate / "MODULE.bazel").exists():
            return candidate
    # Fallback: use AGROBOT_ROOT env var if set (always correct inside Docker).
    env_root = os.environ.get("AGROBOT_ROOT")
    if env_root:
        return Path(env_root)
    raise RuntimeError(
        "Cannot find repo root. Set AGROBOT_ROOT env var or run from the repo."
    )


class DINOv2SAM2Detector:
    """Zero-shot tomato detector using DINOv2 patch similarity + SAM2 masking.

    Implements the same interface as PlaceholderDetector so the node requires
    no changes beyond swapping the import.

    Args:
        device: torch.device to run inference on. Defaults to auto-selection.
        sam2_checkpoint: Path to SAM2 .pt checkpoint file.
        confidence_threshold: Minimum patch similarity score to generate a proposal.
        max_detections: Maximum number of tomato detections per frame.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        sam2_checkpoint: Optional[str] = None,
        confidence_threshold: float = 0.3,
        max_detections: int = 20,
        use_sam2: bool = True,
    ) -> None:
        self._device = device or _select_device()
        self._conf_threshold = confidence_threshold
        self._max_detections = max_detections
        self._use_sam2 = use_sam2
        self._repo_root = _find_repo_root()

        self._dino: Optional[torch.nn.Module] = None
        self._sam2_predictor = None
        self._query_embedding: Optional[torch.Tensor] = None

        ckpt_path = sam2_checkpoint or str(
            self._repo_root / _DEFAULT_SAM2_CKPT
        )
        self._sam2_ckpt = Path(ckpt_path)

        self._load_models()
        self._build_query_embedding()

        logger.info(
            "DINOv2SAM2Detector initialized on %s. "
            "SAM2 ckpt: %s. conf_threshold=%.2f",
            self._device,
            self._sam2_ckpt,
            self._conf_threshold,
        )

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load DINOv2 via torch.hub and SAM2 from local checkpoint."""
        logger.info("Loading DINOv2 (%s) via torch.hub...", _DINO_MODEL_NAME)
        # torch.hub downloads to ~/.cache/torch/hub/ on first call.
        # Subsequent calls use the cached download.
        self._dino = torch.hub.load(
            "facebookresearch/dinov2",
            _DINO_MODEL_NAME,
            pretrained=True,
        )
        self._dino.eval()
        self._dino.to(self._device)
        logger.info("DINOv2 loaded.")

        if not self._sam2_ckpt.exists() or not self._use_sam2:
            if not self._use_sam2:
                logger.info("SAM2 disabled (use_sam2=False). DINOv2-only mode.")
            else:
                logger.warning(
                "SAM2 checkpoint not found at %s. "
                "Download sam2.1_hiera_small.pt from "
                "https://github.com/facebookresearch/sam2/releases "
                "and place it at models/sam2/sam2.1_hiera_small.pt. "
                "Detector will run WITHOUT mask refinement until weights are present.",
                self._sam2_ckpt,
            )
            self._sam2_predictor = None
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # build_sam2 resolves cfg via Hydra pkg://sam2 search path —
            # pass only the relative path within the sam2 package, not absolute.
            sam2_model = build_sam2(_DEFAULT_SAM2_CFG, str(self._sam2_ckpt), device=self._device)

            # If a fine-tuned state dict exists alongside the base checkpoint
            # (same path with _finetuned suffix, or explicit override), load it.
            # This keeps the loading interface clean — base weights always load
            # first so the model is valid even if fine-tuning was partial.
            finetuned_path = self._sam2_ckpt.parent / "sam2_tomato_finetuned.pt"
            if finetuned_path.exists():
                state_dict = torch.load(str(finetuned_path), map_location=self._device)
                sam2_model.load_state_dict(state_dict)
                logger.info("Loaded fine-tuned SAM2 weights from %s.", finetuned_path)

            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
            logger.info("SAM2 loaded from %s.", self._sam2_ckpt)
        except Exception as exc:
            logger.warning(
                "SAM2 failed to load (%s). Running without mask refinement.", exc
            )
            self._sam2_predictor = None

    def _build_query_embedding(self) -> None:
        """Build the tomato query embedding.

        Priority:
          1. Load from models/query_embedding.pt if present (built by
             perception/tools/build_query_embedding.py from Laboro Tomato
             training patches). This gives data-driven, colour-agnostic features.
          2. Fall back to the hardcoded red/orange RGB prior (zero-shot baseline).

        The file-based embedding replaces the prior permanently once generated.
        Sprint 3: recompute after SAM2 fine-tuning for further improvement.
        """
        query_path = self._repo_root / _DEFAULT_QUERY_EMBEDDING
        if query_path.exists():
            query = torch.load(str(query_path), map_location=self._device)
            self._query_embedding = F.normalize(query.float(), dim=0)
            logger.info("Loaded data-driven query embedding from %s.", query_path)
            return

        logger.info(
            "No query embedding at %s. Using hardcoded RGB prior. "
            "Run perception/tools/build_query_embedding.py to build a better one.",
            query_path,
        )
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        embeddings = []
        with torch.no_grad():
            for rgb in _TOMATO_RGB_PRIOR:
                # Build a 518×518 patch filled with the tomato colour.
                patch = np.full(
                    (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE, 3), rgb, dtype=np.float32
                )
                patch = (patch - mean) / std
                tensor = torch.from_numpy(
                    np.transpose(patch, (2, 0, 1))
                ).unsqueeze(0).to(self._device)

                # Extract CLS token as the colour's global embedding.
                features = self._dino.forward_features(tensor)
                cls_token = features["x_norm_clstoken"]  # (1, 768)
                embeddings.append(cls_token)

        # Mean of colour priors → (768,) query vector, L2-normalised.
        query = torch.cat(embeddings, dim=0).mean(dim=0)
        self._query_embedding = F.normalize(query, dim=0)
        logger.debug("Tomato query embedding built. shape=%s", self._query_embedding.shape)

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        """Detect tomatoes in a preprocessed frame.

        Args:
            preprocessed_chw: float32 array (3, H, W) from preprocess_for_dino().
                               Values are ImageNet-normalized, range ~[-2.5, 2.5].

        Returns:
            List of dicts: [{"box": [x1,y1,x2,y2], "score": float,
                             "label": str, "mask": np.ndarray | None}]
            Empty list if no tomatoes detected above confidence_threshold.
        """
        if self._dino is None or self._query_embedding is None:
            return []

        tensor = torch.from_numpy(preprocessed_chw).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._dino.forward_features(tensor)

        # patch_tokens: (1, N_patches, embed_dim) where N_patches = 37*37 = 1369
        patch_tokens = features["x_norm_patchtokens"]  # (1, 1369, 768)
        patch_tokens = patch_tokens.squeeze(0)          # (1369, 768)

        # Cosine similarity between each patch and the tomato query.
        patch_norms = F.normalize(patch_tokens, dim=1)  # (1369, 768)
        similarity = (patch_norms @ self._query_embedding).cpu().numpy()  # (1369,)

        # Reshape to spatial grid (37, 37).
        sim_map = similarity.reshape(_DINO_GRID, _DINO_GRID)

        proposals = self._similarity_map_to_boxes(sim_map)
        if not proposals:
            return []

        # Scale proposals from DINOv2 grid coords (37×37) to image coords (518×518).
        scale = _DINO_PATCH_SIZE
        scaled_proposals = []
        for box, score in proposals:
            gx1, gy1, gx2, gy2 = box
            scaled_proposals.append((
                [gx1 * scale, gy1 * scale, gx2 * scale, gy2 * scale],
                score,
            ))

        if self._sam2_predictor is not None:
            detections = self._refine_with_sam2(
                preprocessed_chw, scaled_proposals
            )
        else:
            detections = [
                {"box": box, "score": float(score), "label": "tomato", "mask": None}
                for box, score in scaled_proposals
            ]

        return detections[: self._max_detections]

    def _similarity_map_to_boxes(
        self, sim_map: np.ndarray
    ) -> list[tuple[list[int], float]]:
        """Convert a (37, 37) similarity map to bounding box proposals.

        Thresholds the map, finds connected components, and returns one box
        per component with score = mean similarity of patches in that region.

        Returns:
            List of ([gx1, gy1, gx2, gy2], score) in grid coordinates.
        """
        binary = (sim_map >= self._conf_threshold).astype(np.uint8)
        if binary.sum() == 0:
            return []

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        proposals = []
        for label_id in range(1, n_labels):  # skip background (0)
            mask = labels == label_id
            component_scores = sim_map[mask]
            mean_score = float(component_scores.mean())

            if mean_score < self._conf_threshold:
                continue

            # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA
            x, y, w, h = (
                stats[label_id, cv2.CC_STAT_LEFT],
                stats[label_id, cv2.CC_STAT_TOP],
                stats[label_id, cv2.CC_STAT_WIDTH],
                stats[label_id, cv2.CC_STAT_HEIGHT],
            )
            proposals.append(([x, y, x + w, y + h], mean_score))

        proposals.sort(key=lambda p: p[1], reverse=True)
        return proposals

    def _refine_with_sam2(
        self,
        preprocessed_chw: np.ndarray,
        proposals: list[tuple[list[float], float]],
    ) -> list[dict]:
        """Use SAM2 to convert bounding box proposals into pixel-level masks.

        SAM2 takes the original image + a box prompt and produces a tight mask.
        We refine the bounding box from the mask contour (tighter than the
        patch-grid proposal) and recompute the score from the mask region.

        Args:
            preprocessed_chw: The same (3, H, W) float32 array passed to detect().
            proposals: List of ([x1,y1,x2,y2], score) in image pixel coordinates.

        Returns:
            List of detection dicts with refined boxes and masks.
        """
        # SAM2 needs the original uint8 RGB image.
        # We reverse the ImageNet normalization to recover approximate pixel values.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        rgb_float = preprocessed_chw * std + mean
        rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
        # CHW → HWC for SAM2
        rgb_hwc = np.transpose(rgb_uint8, (1, 2, 0))

        self._sam2_predictor.set_image(rgb_hwc)

        detections = []
        for box_coords, proposal_score in proposals:
            box_np = np.array(box_coords, dtype=np.float32)

            masks, scores, _ = self._sam2_predictor.predict(
                box=box_np,
                multimask_output=False,
            )

            if masks is None or len(masks) == 0:
                detections.append({
                    "box": [int(v) for v in box_coords],
                    "score": float(proposal_score),
                    "label": "tomato",
                    "mask": None,
                })
                continue

            best_mask = masks[0].astype(np.uint8)  # (H, W) binary

            # Refine bounding box from mask contour.
            contours, _ = cv2.findContours(
                best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                all_points = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                refined_box = [x, y, x + w, y + h]
            else:
                refined_box = [int(v) for v in box_coords]

            # SAM2's iou_predictions score is reliable — use it directly.
            sam_score = float(scores[0]) if scores is not None else float(proposal_score)

            detections.append({
                "box": refined_box,
                "score": sam_score,
                "label": "tomato",
                "mask": best_mask,
            })

        return detections
