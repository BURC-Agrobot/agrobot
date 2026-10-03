#!/usr/bin/env python3
"""
run_eval.py — Run the tomato detector on a validation set and report metrics.

Sprint 2: mAP@0.5 (if ground truth provided), latency mean/p99, and per-image
detection counts. Used for ablation (DINOv2-only vs DINOv2+SAM2) and REPRODUCE.md.

Usage (from repo root):
  PYTHONPATH=perception python perception/eval/run_eval.py --val-list data/val_list.txt
  PYTHONPATH=perception python perception/eval/run_eval.py --val-list data/val_list.txt --gt-csv data/val_gt.csv
  PYTHONPATH=perception python perception/eval/run_eval.py --val-list data/val_list.txt --detector dino_only

Inside Docker:
  cd /workspace && PYTHONPATH=perception python perception/eval/run_eval.py --val-list data/val_list.txt

With visualizations (HTML report + annotated images):
  ... --visualize-dir eval_reports/run_001

Requires: AGROBOT_ROOT or run from repo root so models/sam2/ and torch.hub resolve.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "perception" / "agrobot_perception").exists():
        env_root = os.environ.get("AGROBOT_ROOT")
        if env_root:
            return Path(env_root)
        raise RuntimeError(
            "Cannot find repo root. Set AGROBOT_ROOT or run from AgrobotV2/."
        )
    return root


def _setup_path(repo_root: Path) -> None:
    perception_dir = repo_root / "perception"
    if str(perception_dir) not in sys.path:
        sys.path.insert(0, str(perception_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detector on val set; report latency and optionally mAP@0.5."
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        required=True,
        help="Path to val_list.txt (one image path per line, relative to repo root).",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="Optional: val_gt.csv with columns image_path,x1,y1,x2,y2,label for mAP.",
    )
    parser.add_argument(
        "--detector",
        choices=["dino_sam2", "dino_only", "sam2_amg", "sam2_semantic"],
        default="dino_sam2",
        help=(
            "dino_sam2 = DINOv2 proposals + SAM2 refinement (default); "
            "dino_only = DINOv2 proposals only; "
            "sam2_amg = SAM2 AMG proposals scored by DINOv2 (architecturally correct); "
            "sam2_semantic = DINOv2 heatmap → semantic point prompts → SAM2ImagePredictor (E3)."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default 0.3).",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=20,
        help=(
            "Maximum detections returned per frame (default 20). "
            "Raise to 30–40 on dense datasets like Laboro Tomato where some images "
            "contain 20+ tomatoes — the hard cap silently limits recall."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional: write per-image detections to CSV.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=None,
        help="Optional: path to a fine-tuned SAM2 checkpoint. "
             "Overrides the default models/sam2/sam2.1_hiera_small.pt.",
    )
    parser.add_argument(
        "--amg-points",
        type=int,
        default=8,
        help="SAM2 AMG points_per_side (default 8 → 64 proposals). "
             "16 → 256 proposals, better recall, ~4x slower.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup frames before timing (default 2).",
    )
    parser.add_argument(
        "--dino-weight",
        type=float,
        default=1.0,
        help="[sam2_amg] DINOv2 weight in score fusion. 1.0 = DINOv2 only (default).",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.0,
        help="[sam2_amg] NMS IoU threshold. 0 = disabled (default). 0.5 = suppress overlaps.",
    )
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=0.3,
        help="[sam2_amg] Contrastive negative weight. 0 = disabled. Requires negative_embedding.pt.",
    )
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        default=None,
        help="Save annotated images and HTML report to this directory. Creates index.html + images/.",
    )
    parser.add_argument(
        "--query-embedding",
        type=Path,
        default=None,
        help=(
            "[sam2_amg/sam2_semantic] Override path to query (positive) embedding .pt file. "
            "Supports (768,) single-prototype and (k, 768) multi-prototype tensors. "
            "Default: models/query_embedding.pt."
        ),
    )
    parser.add_argument(
        "--negative-embedding",
        type=Path,
        default=None,
        help=(
            "[sam2_amg/sam2_semantic] Override path to negative embedding .pt file. "
            "Use this to A/B between models/negative_embedding.pt (background mean) "
            "and models/hard_negative_embedding.pt (FP-mined, from mine_hard_negatives.py). "
            "Default: models/negative_embedding.pt."
        ),
    )
    parser.add_argument(
        "--migraphx-dino",
        type=Path,
        default=None,
        help=(
            "[sam2_amg/sam2_semantic] Path to MIGraphX compiled DINOv2 .mxr file. "
            "[NUCBOX] only. Requires ROCm + MIGraphX (blocked: see docs/SPRINT3_ROCM_ISSUE.md). "
            "When present, replaces PyTorch DINOv2 forward with MIGraphX inference (~20ms vs ~350ms)."
        ),
    )
    parser.add_argument(
        "--dino-lora-path",
        type=Path,
        default=None,
        help=(
            "[sam2_amg/sam2_semantic] Path to LoRA adapter weights from finetune_dino_lora.py. "
            "When provided, DINOv2 is loaded with LoRA adapters before inference. "
            "Must be paired with a query embedding built with the same LoRA model."
        ),
    )
    parser.add_argument(
        "--amg-crops",
        action="store_true",
        default=False,
        help=(
            "[sam2_amg] Enable quadrant-crop multi-scale proposals. "
            "Runs AMG on 4 overlapping quadrant crops and reprojects to full image. "
            "Improves recall for small/distant tomatoes below the full-image grid resolution. "
            "~2x latency increase."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=48,
        help="[sam2_semantic] Number of top-scoring DINOv2 heatmap patches to use as semantic prompts.",
    )
    parser.add_argument(
        "--fusion-features-out",
        type=Path,
        default=None,
        help=(
            "Phase 2.2 step 1: dump per-detection fusion features to this JSONL "
            "file (one line per image). Pair with --siglip and a low --confidence "
            "to get a dense feature dump suitable for training a fusion MLP via "
            "perception/tools/train_fusion_mlp.py."
        ),
    )
    parser.add_argument(
        "--fusion-mlp",
        type=Path,
        default=None,
        help=(
            "Phase 2.2 step 3: load a trained FusionMLP checkpoint and re-score "
            "every detection with it. Replaces the SigLIP fixed-weight fusion. "
            "Compose with --siglip to populate siglip_sim before MLP scoring."
        ),
    )
    parser.add_argument(
        "--mlp-confidence",
        type=float,
        default=None,
        help=(
            "[--fusion-mlp] Confidence threshold applied AFTER MLP re-scoring "
            "(0–1, MLP probability). Separate from --confidence which gates the "
            "inner SAM2+DINOv2 detector before the MLP sees detections. "
            "Use --confidence 0.0 --mlp-confidence 0.40 to let all SAM2 proposals "
            "reach the MLP, then keep only those the MLP scores ≥ 0.40. "
            "This is the correct way to threshold the P2.2 fusion pipeline. "
            "Defaults to --confidence value when --fusion-mlp is not set, and to "
            "0.0 (no gate) when --fusion-mlp is set but --mlp-confidence is omitted."
        ),
    )
    parser.add_argument(
        "--mask-refine",
        action="store_true",
        default=False,
        help=(
            "Enable second-pass SAM2 mask refinement (Phase 2.3): re-prompt "
            "SAM2 with each surviving detection's centroid and replace the "
            "mask if the refined version scores higher AND has IoU>=0.5 with "
            "the original. Adds ~30 ms per surviving detection. Expected mAP "
            "gain on borderline-IoU cases: +0.005-0.020."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["mAP50", "coco"],
        default="mAP50",
        help=(
            "Metric to report. mAP50 = single-IoU 0.5 (legacy, sprint-tracking). "
            "coco = COCO mAP@[.5:.95] + AP50 + AP75 + AP_small/medium/large "
            "(paper-grade; required for any peer-reviewed publication)."
        ),
    )
    parser.add_argument(
        "--siglip",
        action="store_true",
        default=False,
        help=(
            "Enable SigLIP per-detection re-scoring (Phase 1.3). "
            "Wraps the detector to crop each detection's bbox, run SigLIP "
            "image encoder + cached text embeddings, fuse with DINOv2 scores. "
            "Default fusion: 0.4*dino + 0.4*siglip + 0.2*pred_iou."
        ),
    )
    parser.add_argument(
        "--siglip-model", type=str, default="google/siglip-base-patch16-224",
        help="HuggingFace SigLIP model id. Default: google/siglip-base-patch16-224 (~370 MB).",
    )
    parser.add_argument(
        "--siglip-w-dino", type=float, default=0.4,
        help="[--siglip] DINO weight in late fusion. Default 0.4.",
    )
    parser.add_argument(
        "--siglip-w-siglip", type=float, default=0.4,
        help="[--siglip] SigLIP weight in late fusion. Default 0.4.",
    )
    parser.add_argument(
        "--siglip-w-pred-iou", type=float, default=0.2,
        help="[--siglip] SAM2 pred_iou weight in late fusion. Default 0.2.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help=(
            "Enable test-time augmentation (Phase 1.2): wrap the detector to also "
            "run on a horizontally-flipped input and NMS-merge the results. "
            "Compose with --amg-crops for crop-augmentation as well. "
            "Roughly doubles per-frame latency. Expected mAP gain: +0.015-0.030."
        ),
    )
    parser.add_argument(
        "--detections-jsonl",
        type=Path,
        default=None,
        help=(
            "Dump raw detections (box, score, label) per image to a JSONL file. "
            "Pair with --confidence 0 --nms-iou 0 --max-detections 1000 to capture "
            "the full pre-filter detection set so a post-hoc sweep over (conf, nms, max_det) "
            "can be replayed cheaply via perception/eval/sweep_post_filter.py."
        ),
    )
    parser.add_argument(
        "--sparse-grid",
        type=int,
        default=4,
        help="[sam2_semantic] Supplementary uniform grid size NxN (default 4 → 16 extra points).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    _setup_path(repo_root)
    os.chdir(repo_root)

    from agrobot_perception.utils.image_utils import preprocess_for_dino
    import cv2

    def _select_device():
        import os as _os, torch as _torch
        if _os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
            return _torch.device("cpu")
        if _torch.backends.mps.is_available():
            return _torch.device("mps")
        if _torch.cuda.is_available():
            return _torch.device("cuda")
        return _torch.device("cpu")

    # Build val image paths

    val_list_path = args.val_list if args.val_list.is_absolute() else repo_root / args.val_list
    if not val_list_path.exists():
        print(f"ERROR: Val list not found: {val_list_path}", file=sys.stderr)
        sys.exit(1)

    image_paths = []
    with open(val_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = repo_root / p
            image_paths.append(p)

    if not image_paths:
        print("No image paths in val list. Add paths to data/val_list.txt.", file=sys.stderr)
        sys.exit(1)

    # Load detector
    sam2_ckpt = None
    if args.sam2_checkpoint:
        sam2_ckpt = str(
            args.sam2_checkpoint if args.sam2_checkpoint.is_absolute()
            else repo_root / args.sam2_checkpoint
        )

    def _abs_str(p: Path) -> str:
        return str(p if p.is_absolute() else repo_root / p)

    query_emb_path    = _abs_str(args.query_embedding)    if args.query_embedding    else None
    neg_emb_path      = _abs_str(args.negative_embedding) if args.negative_embedding else None
    lora_path         = _abs_str(args.dino_lora_path)     if args.dino_lora_path     else None
    migraphx_dino_path = _abs_str(args.migraphx_dino)     if args.migraphx_dino     else None

    if args.detector == "sam2_amg":
        from agrobot_perception.detectors.sam2_amg_detector import SAM2AMGDetector
        detector = SAM2AMGDetector(
            device=_select_device(),
            sam2_checkpoint=sam2_ckpt,
            confidence_threshold=args.confidence,
            max_detections=args.max_detections,
            points_per_side=args.amg_points,
            dino_score_weight=args.dino_weight,
            nms_iou_threshold=args.nms_iou,
            negative_weight=args.negative_weight,
            query_embedding_path=query_emb_path,
            negative_embedding_path=neg_emb_path,
            use_quadrant_crops=args.amg_crops,
            dino_lora_path=lora_path,
            migraphx_dino_path=migraphx_dino_path,
        )
    elif args.detector == "sam2_semantic":
        from agrobot_perception.detectors.sam2_semantic_detector import SAM2SemanticDetector
        detector = SAM2SemanticDetector(
            device=_select_device(),
            sam2_checkpoint=sam2_ckpt,
            confidence_threshold=args.confidence,
            max_detections=args.max_detections,
            top_k_semantic=args.top_k,
            sparse_grid_n=args.sparse_grid,
            dino_score_weight=args.dino_weight,
            nms_iou_threshold=args.nms_iou if args.nms_iou > 0 else 0.5,
            negative_weight=args.negative_weight,
            query_embedding_path=query_emb_path,
            negative_embedding_path=neg_emb_path,
            dino_lora_path=lora_path,
        )
    else:
        from agrobot_perception.detectors.dino_sam2_detector import DINOv2SAM2Detector
        use_sam2 = args.detector == "dino_sam2"
        detector = DINOv2SAM2Detector(
            device=_select_device(),
            confidence_threshold=args.confidence,
            sam2_checkpoint=sam2_ckpt,
            use_sam2=use_sam2,
        )
    input_size = (518, 518)

    # Wrapper composition order (innermost → outermost):
    #   base detector  →  mask-refine  →  SigLIP re-score  →  TTA
    # Rationale:
    #   - mask-refine improves IoU and re-scores with the same DINOv2 formula;
    #     must happen before SigLIP because SigLIP scores the cropped *bbox*
    #     and we want SigLIP to see the refined bbox.
    #   - SigLIP re-scores with a richer fusion; must be inside TTA so each
    #     TTA pass's NMS keeps the best of *fused* scores, not raw DINOv2.
    if args.mask_refine:
        from eval.mask_refine import MaskRefineWrapper
        detector = MaskRefineWrapper(detector)
        print("  Mask refine enabled (centroid re-prompt + score guardrail)")

    # When --fusion-mlp is active, the SigLIP wrapper must NOT gate by confidence —
    # that gate must be applied by the MLP wrapper so it fires on MLP probabilities,
    # not on the intermediate DINOv2+SigLIP blended score. Passing confidence_threshold=0
    # to SigLIP here lets every proposal through to the MLP.
    siglip_conf = 0.0 if args.fusion_mlp else args.confidence
    if args.siglip:
        from eval.siglip_rescoring import SigLIPRescoringWrapper
        detector = SigLIPRescoringWrapper(
            detector,
            model_id=args.siglip_model,
            w_dino=args.siglip_w_dino,
            w_siglip=args.siglip_w_siglip,
            w_pred_iou=args.siglip_w_pred_iou,
            nms_iou_threshold=args.nms_iou if args.nms_iou > 0 else 0.5,
            max_detections=args.max_detections,
            confidence_threshold=siglip_conf,
        )
        print(f"  SigLIP enabled (w_dino={args.siglip_w_dino}, "
              f"w_siglip={args.siglip_w_siglip}, w_pred_iou={args.siglip_w_pred_iou})")

    if args.fusion_mlp:
        # --mlp-confidence gates on MLP probability (0–1).
        # Falls back to --confidence if --mlp-confidence was not explicitly set,
        # and to 0.0 (no gate) if neither was set. This lets users write:
        #   --confidence 0.0 --mlp-confidence 0.40
        # to let all SAM2 proposals reach the MLP, then threshold on MLP output.
        mlp_conf = args.mlp_confidence if args.mlp_confidence is not None else args.confidence
        from eval.fusion_mlp import FusionMLPWrapper
        detector = FusionMLPWrapper(
            detector,
            mlp_path=str(args.fusion_mlp if args.fusion_mlp.is_absolute()
                         else repo_root / args.fusion_mlp),
            nms_iou_threshold=args.nms_iou if args.nms_iou > 0 else 0.5,
            max_detections=args.max_detections,
            confidence_threshold=mlp_conf,
        )
        print(f"  Fusion MLP loaded: {args.fusion_mlp}  (mlp-confidence={mlp_conf:.2f})")

    if args.fusion_features_out:
        # Wrap last so the dump captures whatever the upstream chain produced.
        from eval.fusion_mlp import FeatureDumpWrapper
        out_p = args.fusion_features_out if args.fusion_features_out.is_absolute() \
            else repo_root / args.fusion_features_out
        detector = FeatureDumpWrapper(detector, output_path=out_p)
        print(f"  Feature dump enabled: {out_p}")

    if args.tta:
        from eval.tta import TTAWrapper
        detector = TTAWrapper(
            detector,
            nms_iou_threshold=args.nms_iou if args.nms_iou > 0 else 0.5,
            max_detections=args.max_detections,
            do_hflip=True,
        )
        print(f"  TTA enabled (hflip; final NMS@{args.nms_iou or 0.5}, max={args.max_detections})")

    # Load ground truth if provided
    gt_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
    if args.gt_csv:
        gt_path = args.gt_csv if args.gt_csv.is_absolute() else repo_root / args.gt_csv
        if gt_path.exists():
            with open(gt_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = row.get("image_path", "").strip()
                    if not img_path:
                        continue
                    try:
                        x1, y1, x2, y2 = (
                            float(row["x1"]), float(row["y1"]),
                            float(row["x2"]), float(row["y2"]),
                        )
                    except KeyError:
                        continue
                    key = str(Path(img_path).resolve())
                    gt_by_image.setdefault(key, []).append((x1, y1, x2, y2))

    # Run inference and collect latencies + detections
    latencies_ms = []
    all_detections: list[tuple[Path, list[dict]]] = []
    total = len(image_paths)

    print(f"Evaluating {total} images ({args.detector}, conf={args.confidence})...")
    print()

    for i, img_path in enumerate(image_paths):
        if not img_path.exists():
            print(f"Skip missing: {img_path}", file=sys.stderr)
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"Skip unreadable: {img_path}", file=sys.stderr)
            continue

        preprocessed = preprocess_for_dino(bgr, input_size=input_size)

        if i < args.warmup:
            detector.detect(preprocessed)
            pct = 100 * (i + 1) / total
            bar_w = 40
            filled = int(bar_w * (i + 1) / total)
            bar = "=" * filled + "-" * (bar_w - filled)
            sys.stdout.write(f"\r  [{bar}] {i + 1}/{total} ({pct:.0f}%) warmup    ")
            sys.stdout.flush()
            continue

        t0 = time.perf_counter()
        dets = detector.detect(preprocessed)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000
        latencies_ms.append(lat_ms)
        all_detections.append((img_path, dets))

        # Progress: bar, count, detections, avg latency, ETA
        pct = 100 * (i + 1) / total
        bar_w = 40
        filled = int(bar_w * (i + 1) / total)
        bar = "=" * filled + "-" * (bar_w - filled)
        avg_s = sum(latencies_ms) / len(latencies_ms) / 1000
        remaining = total - i - 1
        eta_s = remaining * avg_s if latencies_ms else 0
        eta_str = f"{eta_s / 60:.1f}m" if eta_s >= 60 else f"{eta_s:.0f}s"
        sys.stdout.write(
            f"\r  [{bar}] {i + 1}/{total} ({pct:.0f}%) | "
            f"{len(dets)} det | avg {avg_s:.1f}s | ETA {eta_str}    "
        )
        sys.stdout.flush()

    print()  # Newline after progress bar

    if not latencies_ms:
        print("No frames timed. Add valid image paths to val_list.txt.", file=sys.stderr)
        sys.exit(1)

    # Latency stats
    latencies_ms.sort()
    n = len(latencies_ms)
    mean_ms = sum(latencies_ms) / n
    p99_ms = latencies_ms[int(0.99 * n)] if n >= 100 else latencies_ms[-1]

    print("── Latency ──")
    print(f"  Frames: {n}")
    print(f"  Mean:   {mean_ms:.2f} ms")
    print(f"  p99:    {p99_ms:.2f} ms")
    print(f"  Detector: {args.detector}")

    # mAP if we have GT — legacy mAP50 always computed for sprint comparability;
    # COCO suite computed additionally when --metric coco is set.
    ap, prec, rec = 0.0, 0.0, 0.0
    coco_metrics: dict[str, float] = {}
    if gt_by_image:
        from eval.metrics import compute_ap_iou_threshold
        ap, prec, rec = compute_ap_iou_threshold(all_detections, gt_by_image, iou_threshold=0.5)
        print("── mAP@0.5 (legacy) ──")
        print(f"  mAP:        {ap:.4f}")
        print(f"  Precision:  {prec:.4f}")
        print(f"  Recall:     {rec:.4f}")
        if args.metric == "coco":
            from eval.metrics_coco import compute_coco_metrics, format_coco_table
            coco_metrics = compute_coco_metrics(all_detections, gt_by_image)
            print(format_coco_table(coco_metrics))
    else:
        total_dets = sum(len(d) for _, d in all_detections)
        print("── Detections ──")
        print(f"  Total: {total_dets} (no GT → no mAP)")

    # Visualization
    if args.visualize_dir:
        vis_dir = args.visualize_dir if args.visualize_dir.is_absolute() else repo_root / args.visualize_dir
        from eval.visualize import generate_html_report
        vis_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"mAP": ap, "precision": prec, "recall": rec, "mean_ms": mean_ms, "p99_ms": p99_ms, "n_frames": n}
        config = {
            "detector": args.detector,
            "confidence": args.confidence,
            "amg_points": args.amg_points,
            "negative_weight": args.negative_weight,
            "nms_iou": args.nms_iou,
            "top_k": args.top_k,
            "sparse_grid": args.sparse_grid,
        }
        html_path = generate_html_report(
            vis_dir, all_detections, gt_by_image, metrics, config, max_images=80,
        )
        print(f"  Visualizations: {html_path}")

    if args.detections_jsonl:
        # JSONL is the right format here: one line per image, append-friendly,
        # streaming-readable. Each line carries the image's resolved path so the
        # downstream sweep can join against gt_by_image without re-reading val_list.
        import json
        out_path = args.detections_jsonl if args.detections_jsonl.is_absolute() \
            else repo_root / args.detections_jsonl
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for img_path, dets in all_detections:
                payload = {
                    "image_path": str(Path(img_path).resolve()),
                    "detections": [
                        {
                            "box": [float(b) for b in d["box"]],
                            "score": float(d["score"]),
                            "label": d.get("label", "tomato"),
                        }
                        for d in dets
                    ],
                }
                f.write(json.dumps(payload) + "\n")
        print(f"  Wrote raw detections: {out_path} ({len(all_detections)} images)")

    if args.output_csv:
        out_path = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "num_detections", "latency_ms"])
            for (img_path, dets), lat in zip(all_detections, latencies_ms):
                w.writerow([str(img_path), len(dets), f"{lat:.2f}"])
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
