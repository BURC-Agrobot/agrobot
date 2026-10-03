"""
visualize.py — Professional eval visualizations: annotated images + HTML report.

Generates per-image overlays (GT, TP, FP boxes) and an HTML index with metrics.
Used by run_eval.py when --visualize-dir is set.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np


_INPUT_SIZE = 518
_GT_COLOR = (34, 197, 94)      # BGR green
_TP_COLOR = (34, 211, 238)     # BGR cyan
_FP_COLOR = (34, 34, 255)      # BGR red
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1


def _iou_box(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _match_detections_to_gt(
    dets: list[dict],
    gt_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[dict, bool]]:
    """Return (det, is_tp) for each detection, using greedy IoU matching."""
    sorted_dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    gt_matched = [False] * len(gt_boxes)
    result = []
    for d in sorted_dets:
        box = (float(d["box"][0]), float(d["box"][1]), float(d["box"][2]), float(d["box"][3]))
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = _iou_box(box, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        is_tp = best_iou >= iou_threshold and best_j >= 0
        if is_tp:
            gt_matched[best_j] = True
        result.append((d, is_tp))
    return result


def _draw_box(
    img: np.ndarray,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, _BOX_THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        img, label, (x1 + 2, y1 - 4),
        _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICKNESS, cv2.LINE_AA,
    )


def _draw_mask_overlay(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.3) -> None:
    overlay = img.copy()
    mask_bgr = np.zeros_like(overlay)
    mask_bgr[:] = color
    overlay[mask > 0] = (
        alpha * mask_bgr[mask > 0] + (1 - alpha) * overlay[mask > 0]
    ).astype(np.uint8)
    np.copyto(img, overlay, where=(mask > 0)[:, :, np.newaxis])


def draw_eval_image(
    bgr: np.ndarray,
    dets: list[dict],
    gt_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
    show_masks: bool = True,
) -> np.ndarray:
    """Draw GT and detections (TP/FP) on letterboxed 518×518 image.

    Args:
        bgr: Original BGR image.
        dets: Detections from detector (box, score, label, mask).
        gt_boxes: Ground truth boxes in 518×518 space.
        iou_threshold: IoU threshold for TP assignment.
        show_masks: If True, overlay TP masks semi-transparently.

    Returns:
        Annotated BGR image (518×518).
    """
    from agrobot_perception.utils.image_utils import resize_with_aspect

    letterboxed = resize_with_aspect(bgr, (_INPUT_SIZE, _INPUT_SIZE))
    vis = letterboxed.copy()

    for (x1, y1, x2, y2) in gt_boxes:
        _draw_box(vis, (x1, y1, x2, y2), _GT_COLOR, "GT")

    matched = _match_detections_to_gt(dets, gt_boxes, iou_threshold)
    for d, is_tp in matched:
        color = _TP_COLOR if is_tp else _FP_COLOR
        tag = "TP" if is_tp else "FP"
        _draw_box(vis, tuple(d["box"]), color, f"{tag} {d['score']:.2f}")
        if show_masks and "mask" in d and d["mask"] is not None and is_tp:
            mask = d["mask"]
            if mask.shape != (vis.shape[0], vis.shape[1]):
                mask = cv2.resize(
                    mask.astype(np.uint8), (vis.shape[1], vis.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            _draw_mask_overlay(vis, mask, _TP_COLOR, alpha=0.25)

    return vis


def _image_to_base64_png(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def generate_html_report(
    output_dir: Path,
    all_detections: list[tuple[Path, list[dict]]],
    gt_by_image: dict[str, list[tuple[float, float, float, float]]],
    metrics: dict,
    config: dict,
    max_images: int = 80,
) -> Path:
    """Generate HTML report with embedded images and metrics.

    Args:
        output_dir: Directory to write report and images.
        all_detections: (img_path, dets) from run_eval.
        gt_by_image: GT boxes keyed by resolved image path.
        metrics: {"mAP": float, "precision": float, "recall": float,
                  "mean_ms": float, "p99_ms": float, "n_frames": int}
        config: {"detector": str, "confidence": float, ...}
        max_images: Cap images in report (avoid huge HTML).

    Returns:
        Path to index.html.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    html_path = output_dir / "index.html"
    items = []
    for i, (img_path, dets) in enumerate(all_detections):
        if i >= max_images:
            break
        key = str(Path(img_path).resolve())
        gt_boxes = gt_by_image.get(key, [])
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        vis = draw_eval_image(bgr, dets, gt_boxes)
        out_name = f"{img_path.stem}_{i:04d}.png"
        out_path = images_dir / out_name
        cv2.imwrite(str(out_path), vis)
        rel_path = f"images/{out_name}"
        items.append({
            "rel_path": rel_path,
            "stem": img_path.stem,
            "n_det": len(dets),
            "n_gt": len(gt_boxes),
        })

    mAP = metrics.get("mAP", 0.0)
    prec = metrics.get("precision", 0.0)
    rec = metrics.get("recall", 0.0)
    mean_ms = metrics.get("mean_ms", 0.0)
    p99_ms = metrics.get("p99_ms", 0.0)
    n_frames = metrics.get("n_frames", 0)
    detector = config.get("detector", "unknown")
    conf = config.get("confidence", 0.0)
    amg_pts = config.get("amg_points", 0)
    neg_w = config.get("negative_weight", 0.0)
    nms = config.get("nms_iou", 0.0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agrobot TOM v2 — Eval Report</title>
  <style>
    :root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --success: #3fb950; --danger: #f85149; }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; padding: 1.5rem; }}
    h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }}
    .metric {{ background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem; }}
    .metric .label {{ font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric .value {{ font-size: 1.25rem; font-weight: 600; color: var(--accent); }}
    .metric .value.success {{ color: var(--success); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; height: auto; display: block; }}
    .card .meta {{ padding: 0.5rem 0.75rem; font-size: 0.8rem; color: var(--muted); }}
    .legend {{ display: flex; gap: 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: var(--muted); }}
    .legend span {{ display: flex; align-items: center; gap: 0.35rem; }}
    .legend .swatch {{ width: 12px; height: 12px; border-radius: 2px; }}
    .swatch-gt {{ background: #22c55e; }}
    .swatch-tp {{ background: #22d3ee; }}
    .swatch-fp {{ background: #ef4444; }}
  </style>
</head>
<body>
  <h1>Agrobot TOM v2 — Eval Report</h1>
  <div class="metrics">
    <div class="metric"><div class="label">mAP@0.5</div><div class="value success">{mAP:.4f}</div></div>
    <div class="metric"><div class="label">Precision</div><div class="value">{prec:.4f}</div></div>
    <div class="metric"><div class="label">Recall</div><div class="value">{rec:.4f}</div></div>
    <div class="metric"><div class="label">Mean (ms)</div><div class="value">{mean_ms:.0f}</div></div>
    <div class="metric"><div class="label">p99 (ms)</div><div class="value">{p99_ms:.0f}</div></div>
    <div class="metric"><div class="label">Frames</div><div class="value">{n_frames}</div></div>
    <div class="metric"><div class="label">Detector</div><div class="value">{detector}</div></div>
    <div class="metric"><div class="label">Config</div><div class="value" style="font-size:0.75rem">conf={conf} pts={amg_pts} neg={neg_w} nms={nms}</div></div>
  </div>
  <div class="legend">
    <span><span class="swatch swatch-gt"></span> GT</span>
    <span><span class="swatch swatch-tp"></span> TP (IoU≥0.5)</span>
    <span><span class="swatch swatch-fp"></span> FP</span>
  </div>
  <div class="grid">
"""

    for item in items:
        html += f"""    <div class="card">
      <img src="{item["rel_path"]}" alt="{item["stem"]}" loading="lazy">
      <div class="meta">{item["stem"]} — {item["n_det"]} det, {item["n_gt"]} GT</div>
    </div>
"""

    html += """  </div>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    return html_path
