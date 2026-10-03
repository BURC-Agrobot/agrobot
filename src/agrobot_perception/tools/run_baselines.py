#!/usr/bin/env python3
"""
run_baselines.py — Evaluate baseline detectors on the Laboro Tomato val split.

Phase 3.3 deliverables for the paper comparison table:
  --model yolov8           supervised single-stage (full Laboro train labels)
  --model grounding_dino   open-vocab zero-shot (HuggingFace IDEA-Research)
  --model owl_vit_v2       open-vocab zero-shot (HuggingFace google/owlv2)

All three produce detections in 518x518 letterboxed pixel space (the same
coordinate system val_gt.csv uses), so the resulting numbers are directly
comparable to perception/eval/run_eval.py + perception/eval/metrics_coco.py.

Usage [NUCBOX CPU]:
  # YOLOv8 (after train_yolo_baseline.py finishes)
  python3 perception/tools/run_baselines.py \\
    --model yolov8 --weights runs/detect/yolov8n_laboro/weights/best.pt \\
    --val-list data/val_list.txt --gt-csv data/val_gt.csv --metric coco

  # Grounding-DINO (HF auto-download ~1 GB on first run)
  python3 perception/tools/run_baselines.py \\
    --model grounding_dino --prompt "tomato." \\
    --val-list data/val_list.txt --gt-csv data/val_gt.csv --metric coco

  # OWL-ViT v2
  python3 perception/tools/run_baselines.py \\
    --model owl_vit_v2 --prompt "a photo of a tomato" \\
    --val-list data/val_list.txt --gt-csv data/val_gt.csv --metric coco

Sprint 4: Phase 3.3.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_INPUT_SIZE = 518


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _setup_path(repo_root: Path) -> None:
    perception_dir = repo_root / "perception"
    if str(perception_dir) not in sys.path:
        sys.path.insert(0, str(perception_dir))


def _letterbox_with_meta(bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(rgb, (new_w, new_h))
    canvas = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
    pad_x = (_INPUT_SIZE - new_w) // 2
    pad_y = (_INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _orig_box_to_518(
    x1: float, y1: float, x2: float, y2: float,
    scale: float, pad_x: int, pad_y: int,
) -> tuple[float, float, float, float]:
    return (x1 * scale + pad_x, y1 * scale + pad_y,
            x2 * scale + pad_x, y2 * scale + pad_y)


# ── YOLOv8 ────────────────────────────────────────────────────────────────────

def _detect_yolov8(weights: Path, val_paths: list[Path], conf: float) -> list[tuple[Path, list[dict]]]:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out: list[tuple[Path, list[dict]]] = []
    for img_path in val_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]
        scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
        new_w = int(orig_w * scale); new_h = int(orig_h * scale)
        pad_x = (_INPUT_SIZE - new_w) // 2
        pad_y = (_INPUT_SIZE - new_h) // 2

        results = model.predict(source=str(img_path), conf=conf, verbose=False)
        dets: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), sc in zip(xyxy, scores):
                bx = _orig_box_to_518(float(x1), float(y1), float(x2), float(y2),
                                      scale, pad_x, pad_y)
                dets.append({"box": list(bx), "score": float(sc), "label": "tomato"})
        out.append((img_path, dets))
    return out


# ── Grounding-DINO ────────────────────────────────────────────────────────────

def _detect_grounding_dino(prompt: str, val_paths: list[Path], conf: float
                           ) -> list[tuple[Path, list[dict]]]:
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    out: list[tuple[Path, list[dict]]] = []
    for img_path in val_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]
        scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
        new_w = int(orig_w * scale); new_h = int(orig_h * scale)
        pad_x = (_INPUT_SIZE - new_w) // 2
        pad_y = (_INPUT_SIZE - new_h) // 2

        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = proc(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=conf,
            text_threshold=conf,
            target_sizes=[image.size[::-1]],
        )[0]

        dets = []
        for box, score in zip(results["boxes"].cpu().numpy(),
                              results["scores"].cpu().numpy()):
            x1, y1, x2, y2 = box.tolist()
            bx = _orig_box_to_518(x1, y1, x2, y2, scale, pad_x, pad_y)
            dets.append({"box": list(bx), "score": float(score), "label": "tomato"})
        out.append((img_path, dets))
    return out


# ── OWL-ViT v2 ────────────────────────────────────────────────────────────────

def _detect_owl_vit_v2(prompt: str, val_paths: list[Path], conf: float
                       ) -> list[tuple[Path, list[dict]]]:
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image

    model_id = "google/owlv2-base-patch16-ensemble"
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    proc = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()

    out: list[tuple[Path, list[dict]]] = []
    for img_path in val_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        orig_h, orig_w = bgr.shape[:2]
        scale = min(_INPUT_SIZE / orig_w, _INPUT_SIZE / orig_h)
        new_w = int(orig_w * scale); new_h = int(orig_h * scale)
        pad_x = (_INPUT_SIZE - new_w) // 2
        pad_y = (_INPUT_SIZE - new_h) // 2

        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inputs = proc(text=[[prompt]], images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = proc.post_process_grounded_object_detection(
            outputs=outputs, threshold=conf, target_sizes=target_sizes,
        )[0]
        dets = []
        for box, score in zip(results["boxes"].cpu().numpy(),
                              results["scores"].cpu().numpy()):
            x1, y1, x2, y2 = box.tolist()
            bx = _orig_box_to_518(x1, y1, x2, y2, scale, pad_x, pad_y)
            dets.append({"box": list(bx), "score": float(score), "label": "tomato"})
        out.append((img_path, dets))
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolov8", "grounding_dino", "owl_vit_v2"],
                        required=True)
    parser.add_argument("--weights", type=Path, default=None,
                        help="[yolov8] path to a trained .pt checkpoint.")
    parser.add_argument("--prompt", type=str, default="tomato",
                        help="[grounding_dino, owl_vit_v2] open-vocab prompt.")
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--metric", choices=["mAP50", "coco"], default="coco")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    repo_root = _repo_root()
    _setup_path(repo_root)
    os.chdir(repo_root)

    val_list_path = args.val_list if args.val_list.is_absolute() else repo_root / args.val_list
    gt_path = args.gt_csv if args.gt_csv.is_absolute() else repo_root / args.gt_csv

    val_paths: list[Path] = []
    with open(val_list_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                p = Path(line)
                val_paths.append(p if p.is_absolute() else repo_root / p)
    if args.max_images > 0:
        val_paths = val_paths[: args.max_images]

    logger.info("Running %s on %d images...", args.model, len(val_paths))
    t0 = time.perf_counter()
    if args.model == "yolov8":
        if args.weights is None:
            logger.error("--weights required for yolov8")
            sys.exit(1)
        weights = args.weights if args.weights.is_absolute() else repo_root / args.weights
        all_dets = _detect_yolov8(weights, val_paths, args.confidence)
    elif args.model == "grounding_dino":
        all_dets = _detect_grounding_dino(args.prompt, val_paths, args.confidence)
    else:
        all_dets = _detect_owl_vit_v2(args.prompt, val_paths, args.confidence)
    elapsed = time.perf_counter() - t0
    logger.info("Inference done in %.1f s (%.1f s/image).", elapsed, elapsed / max(1, len(val_paths)))

    gt_by_image = _load_gt(gt_path)
    from eval.metrics import compute_ap_iou_threshold
    ap, prec, rec = compute_ap_iou_threshold(all_dets, gt_by_image, iou_threshold=0.5)
    print("── mAP@0.5 (legacy) ──")
    print(f"  mAP:        {ap:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")

    if args.metric == "coco":
        from eval.metrics_coco import compute_coco_metrics, format_coco_table
        cm = compute_coco_metrics(all_dets, gt_by_image)
        print(format_coco_table(cm))


if __name__ == "__main__":
    main()
