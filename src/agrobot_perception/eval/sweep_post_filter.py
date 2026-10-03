#!/usr/bin/env python3
"""
sweep_post_filter.py — Cheap post-hoc sweep over (confidence, nms_iou, max_detections).

Why this tool exists:
  The detector's heavy compute (SAM2 AMG + DINOv2 forward + per-mask scoring)
  is independent of the post-filter knobs (--confidence, --nms-iou,
  --max-detections). A naive sweep over 12 combinations would re-run inference
  12 times — ~10h on NucBox CPU. Instead we run inference once with no filter
  (`--confidence 0 --nms-iou 0 --max-detections 1000 --detections-jsonl ...`)
  and replay the post-filter chain in this tool in seconds.

Replays the EXACT post-filter chain from sam2_amg_detector.detect():
  1. score >= confidence
  2. NMS at nms_iou_threshold (cv2.dnn.NMSBoxes — same call as in the detector)
  3. sort by score descending
  4. cap at max_detections

Then computes mAP@0.5 + precision/recall using the same metrics module the live
eval uses (perception/eval/metrics.py), so results are bit-identical to a full
re-run had we paid the inference cost.

Usage (from repo root):
  # Step 1: dump raw detections once (~50 min on NucBox CPU)
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \\
    python3 perception/eval/run_eval.py \\
      --val-list data/val_list.txt --gt-csv data/val_gt.csv \\
      --detector sam2_amg --amg-points 28 \\
      --confidence 0.0 --nms-iou 0.0 --max-detections 1000 \\
      --dino-weight 0.7 \\
      --query-embedding models/query_embedding_k4.pt \\
      --negative-embedding models/negative_embedding.pt \\
      --negative-weight 1.0 \\
      --detections-jsonl eval_reports/sweep_p1_1/raw_detections.jsonl

  # Step 2: sweep post-filter combinations (~30 sec)
  PYTHONPATH=perception python3 perception/eval/sweep_post_filter.py \\
      --detections-jsonl eval_reports/sweep_p1_1/raw_detections.jsonl \\
      --gt-csv data/val_gt.csv \\
      --confidence 0.20 0.25 0.30 0.35 0.40 \\
      --nms-iou 0.5 \\
      --max-detections 30 45 60 \\
      --output-csv eval_reports/sweep_p1_1/sweep_results.csv

Output: a sorted markdown table of (conf, nms, max_det) → (mAP, prec, rec, n_dets).

Sprint 4: Phase 1.1 — find the free-lunch confidence/cap headroom on the S4.12 config.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "perception" / "agrobot_perception").exists():
        env_root = os.environ.get("AGROBOT_ROOT")
        if env_root:
            return Path(env_root)
        raise RuntimeError("Cannot find repo root. Set AGROBOT_ROOT.")
    return root


def _setup_path(repo_root: Path) -> None:
    perception_dir = repo_root / "perception"
    if str(perception_dir) not in sys.path:
        sys.path.insert(0, str(perception_dir))


def _nms(detections: list[dict], nms_iou_threshold: float) -> list[dict]:
    """Replay sam2_amg_detector._nms exactly: cv2.dnn.NMSBoxes with score_threshold=0."""
    if len(detections) <= 1 or nms_iou_threshold <= 0:
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
        score_threshold=0.0, nms_threshold=nms_iou_threshold,
    )
    if len(indices) == 0:
        return detections
    kept = np.asarray(indices).flatten()
    return [detections[int(i)] for i in kept]


def _apply_post_filter(
    raw_dets: list[dict],
    confidence: float,
    nms_iou: float,
    max_detections: int,
) -> list[dict]:
    """Mirror the exact filter chain in sam2_amg_detector.detect() lines 409–429."""
    filtered = [d for d in raw_dets if d["score"] >= confidence]
    if not filtered:
        return []
    filtered = _nms(filtered, nms_iou)
    filtered.sort(key=lambda d: d["score"], reverse=True)
    return filtered[:max_detections]


def _load_dumped_detections(path: Path) -> list[tuple[Path, list[dict]]]:
    """Load JSONL of {image_path, detections}. Returns (image_path, dets) per line."""
    out: list[tuple[Path, list[dict]]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append((Path(obj["image_path"]), obj["detections"]))
    return out


def _load_gt(gt_csv: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Same loader run_eval uses, keyed by resolved absolute path string."""
    gt: dict[str, list[tuple[float, float, float, float]]] = {}
    with open(gt_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get("image_path", "").strip()
            if not img_path:
                continue
            try:
                box = (float(row["x1"]), float(row["y1"]),
                       float(row["x2"]), float(row["y2"]))
            except KeyError:
                continue
            key = str(Path(img_path).resolve())
            gt.setdefault(key, []).append(box)
    return gt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cheap post-hoc sweep over (conf, nms_iou, max_detections) "
                    "against pre-dumped raw detections."
    )
    parser.add_argument("--detections-jsonl", type=Path, required=True,
                        help="Raw detections produced by run_eval.py --detections-jsonl.")
    parser.add_argument("--gt-csv", type=Path, required=True,
                        help="Same val_gt.csv used by run_eval.py.")
    parser.add_argument("--confidence", type=float, nargs="+",
                        default=[0.20, 0.25, 0.30, 0.35, 0.40],
                        help="Confidence thresholds to sweep.")
    parser.add_argument("--nms-iou", type=float, nargs="+", default=[0.5],
                        help="NMS IoU thresholds to sweep.")
    parser.add_argument("--max-detections", type=int, nargs="+", default=[30, 45, 60],
                        help="Per-image detection caps to sweep.")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Optional CSV dump of all sweep results.")
    args = parser.parse_args()

    repo_root = _repo_root()
    _setup_path(repo_root)
    os.chdir(repo_root)

    from eval.metrics import compute_ap_iou_threshold

    det_path = args.detections_jsonl if args.detections_jsonl.is_absolute() \
        else repo_root / args.detections_jsonl
    gt_path = args.gt_csv if args.gt_csv.is_absolute() else repo_root / args.gt_csv

    if not det_path.exists():
        print(f"ERROR: detections file not found: {det_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading raw detections from {det_path}...")
    raw = _load_dumped_detections(det_path)
    total_raw = sum(len(d) for _, d in raw)
    print(f"  {len(raw)} images, {total_raw} raw detections "
          f"({total_raw / max(1, len(raw)):.1f} per image)")

    print(f"Loading ground truth from {gt_path}...")
    gt_by_image = _load_gt(gt_path)
    total_gt = sum(len(v) for v in gt_by_image.values())
    print(f"  {len(gt_by_image)} images with GT, {total_gt} total boxes")
    print()

    # Cartesian sweep over the three knobs.
    results = []
    for conf in args.confidence:
        for nms in args.nms_iou:
            for max_det in args.max_detections:
                filtered_all: list[tuple[Path, list[dict]]] = []
                for img_path, dets in raw:
                    filtered_all.append(
                        (img_path, _apply_post_filter(dets, conf, nms, max_det))
                    )
                ap, prec, rec = compute_ap_iou_threshold(
                    filtered_all, gt_by_image, iou_threshold=0.5,
                )
                n_dets = sum(len(d) for _, d in filtered_all)
                results.append({
                    "confidence": conf, "nms_iou": nms, "max_detections": max_det,
                    "mAP": ap, "precision": prec, "recall": rec,
                    "n_detections": n_dets,
                })

    results.sort(key=lambda r: r["mAP"], reverse=True)

    print("── Sweep results (sorted by mAP@0.5 desc) ──")
    print(f"{'rank':>4}  {'conf':>5}  {'nms':>5}  {'max':>4}  "
          f"{'mAP':>7}  {'prec':>7}  {'rec':>7}  {'n_det':>6}")
    print("─" * 64)
    for rank, r in enumerate(results, 1):
        marker = "  <-- best" if rank == 1 else ""
        print(f"{rank:>4}  {r['confidence']:>5.2f}  {r['nms_iou']:>5.2f}  "
              f"{r['max_detections']:>4d}  {r['mAP']:>7.4f}  "
              f"{r['precision']:>7.4f}  {r['recall']:>7.4f}  "
              f"{r['n_detections']:>6d}{marker}")

    if args.output_csv:
        out_path = args.output_csv if args.output_csv.is_absolute() \
            else repo_root / args.output_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
