#!/usr/bin/env python3
"""
train_yolo_baseline.py — Supervised YOLOv8n baseline on Laboro Tomato (Phase 3.3).

Why this exists for the paper:
  YOLOv8n is the de facto "small supervised single-stage detector" baseline.
  Without it the foundation-model numbers cannot be contextualised — reviewers
  will ask "what does fully supervised get?" and the answer must be a single
  number from the same val split with the same metric.

What this does NOT touch:
  No model surgery, no exotic augmentation, no extended training schedule.
  We use the ultralytics defaults so the comparison is the well-known YOLOv8n
  baseline anybody can reproduce, not a tuned competitor.

YAML written automatically. The standard ultralytics layout:
  data/Laboro-Tomato/
    train/images/
    train/labels/
    val/images/
    val/labels/
  is exactly what the dataset already has, so no file moves needed.

Sprint 4: Phase 3.3.

Usage [NUCBOX CPU, ~6h for 100 epochs at 643 train images]:
  pip install ultralytics --break-system-packages   # one-time
  AGROBOT_FORCE_CPU=1 PYTHONPATH=perception \\
    python3 perception/tools/train_yolo_baseline.py --epochs 100

Then evaluate on the same val_list:
  python3 perception/tools/run_baselines.py \\
    --model yolov8 --weights runs/detect/yolov8n_laboro/weights/best.pt \\
    --val-list data/val_list.txt --gt-csv data/val_gt.csv --metric coco
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _write_dataset_yaml(repo_root: Path) -> Path:
    """The Laboro-Tomato YOLO labels include 6 classes (b/l x ripe/half/green).
    We collapse to single-class for paper comparability with our zero-shot
    pipeline — the comparison is always 'tomato vs not tomato'.

    To do the collapse without rewriting label files, we override the class
    map: train.yaml lists 1 class and ultralytics ignores the original
    integer class IDs as long as the labels exist.

    Caveat: ultralytics actually reads the integer class. To collapse cleanly
    we write a sibling label dir (train_labels_1cls/, val_labels_1cls/) with
    the class id forced to 0. This is idempotent and cheap.
    """
    out_dir = repo_root / "data" / "Laboro-Tomato"
    for split in ("train", "val"):
        src = out_dir / split / "labels"
        dst = out_dir / split / "labels_1cls"
        dst.mkdir(parents=True, exist_ok=True)
        for label_file in src.glob("*.txt"):
            new_lines = []
            for line in label_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # Collapse class id to 0; keep the rest of the line.
                new_lines.append(" ".join(["0"] + parts[1:]))
            (dst / label_file.name).write_text("\n".join(new_lines) + "\n")

    # YOLO needs symlinks (or moved data) so that for each images/<x>.jpg,
    # there is a labels/<x>.txt sibling. Easiest hack: write a yaml that points
    # to symlinked label dirs. We create labels_1cls and tell ultralytics where
    # to find them via 'train' / 'val' image roots and relative label dirs.
    # ultralytics resolves labels by replacing 'images' -> 'labels' in the path.
    # So we ALSO need to provide an alt image root that has labels next to it.
    # Easiest: symlink images dir alongside labels_1cls and call it images.
    # We do that here.
    for split in ("train", "val"):
        alt_root = out_dir / split / "yolo_1cls"
        alt_root.mkdir(parents=True, exist_ok=True)
        img_link = alt_root / "images"
        lbl_link = alt_root / "labels"
        if not img_link.exists():
            img_link.symlink_to(out_dir / split / "images", target_is_directory=True)
        if not lbl_link.exists():
            lbl_link.symlink_to(out_dir / split / "labels_1cls", target_is_directory=True)

    yaml_path = repo_root / "data" / "Laboro-Tomato" / "yolo_1cls.yaml"
    yaml_dict = {
        "path": str(out_dir),
        "train": str(out_dir / "train" / "yolo_1cls" / "images"),
        "val":   str(out_dir / "val"   / "yolo_1cls" / "images"),
        "names": {0: "tomato"},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_dict))
    logger.info("Wrote dataset YAML: %s", yaml_path)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO input size. 640 is the YOLOv8n default.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="yolov8n_laboro")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu / 0 / mps. Use cpu on NucBox until ROCm unblocked.")
    args = parser.parse_args()

    repo_root = _repo_root()
    os.chdir(repo_root)
    yaml_path = _write_dataset_yaml(repo_root)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run:")
        logger.error("  pip install ultralytics --break-system-packages")
        sys.exit(1)

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=2,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
