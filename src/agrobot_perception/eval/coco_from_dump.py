#!/usr/bin/env python3
"""
coco_from_dump.py — Compute COCO metrics from a raw-detections JSONL dump
with post-filter (confidence, nms_iou, max_detections). Lets us produce
apples-to-apples COCO AP numbers for configurations that were only dumped
under legacy mAP (Phase 1.1 baseline).

Usage:
  PYTHONPATH=perception python3 perception/eval/coco_from_dump.py \\
    --detections-jsonl eval_reports/sweep_p1_1/raw_detections.jsonl \\
    --gt-csv data/val_gt.csv \\
    --confidence 0.35 --nms-iou 0.50 --max-detections 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-jsonl", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--nms-iou", type=float, required=True)
    parser.add_argument("--max-detections", type=int, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "perception"))

    from eval.sweep_post_filter import _apply_post_filter, _load_dumped_detections, _load_gt
    from eval.metrics_coco import compute_coco_metrics, format_coco_table
    from eval.metrics import compute_ap_iou_threshold

    raw = _load_dumped_detections(
        args.detections_jsonl if args.detections_jsonl.is_absolute()
        else repo_root / args.detections_jsonl
    )
    gt = _load_gt(
        args.gt_csv if args.gt_csv.is_absolute() else repo_root / args.gt_csv
    )

    filt = [
        (p, _apply_post_filter(d, args.confidence, args.nms_iou, args.max_detections))
        for p, d in raw
    ]
    print(f"After filter: {sum(len(d) for _, d in filt)} detections across {len(filt)} images")

    legacy_ap, legacy_p, legacy_r = compute_ap_iou_threshold(filt, gt, iou_threshold=0.5)
    print(f"── Legacy mAP@0.5 ──\n  mAP={legacy_ap:.4f}  P={legacy_p:.4f}  R={legacy_r:.4f}")

    coco = compute_coco_metrics(filt, gt)
    print(format_coco_table(coco))


if __name__ == "__main__":
    main()
