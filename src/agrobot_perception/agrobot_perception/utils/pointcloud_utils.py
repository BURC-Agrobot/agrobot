"""
pointcloud_utils.py — 3D geometry utilities for tomato spatial localization.

Purpose: Pure-function library for point-cloud clipping, RANSAC sphere fitting,
         and color crop extraction. No ROS types — all I/O is numpy arrays or
         Python scalars, making these trivially testable off-robot.

Target environment: [NUCBOX] runtime; [MAC] for tests and offline analysis.
Sprint: 4 — tomato_spatial_node (NODE 2 in the picking pipeline).

Data flow (caller is TomatoSpatialNode):
  PointCloud2 (parsed to Nx3 by caller)
      │
      ▼
  clip_points_to_bbox()   ← project 3D→2D, keep points inside 518×518 bbox
      │
      ▼
  fit_sphere_ransac()     ← algebraic LS on 4-point subsamples, refine on inliers
      │
      ▼
  compute_extents()       ← axis-aligned AABB of inlier cluster
      │
      ▼
  crop_color_to_bbox()    ← reverse letterbox to cut JPEG from color frame
"""

from __future__ import annotations

import numpy as np


# ─── Clipping ─────────────────────────────────────────────────────────────────

def clip_points_to_bbox(
    points: np.ndarray,
    bbox_518: tuple[float, float, float, float],
    fx: float,
    fy: float,
    cx_cam: float,
    cy_cam: float,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Filter an Nx3 point cloud to points that project inside a 518×518 bbox.

    Detection boxes from tomato_detector_node are in 518×518 letterboxed space
    (preprocess_for_dino() convention). This function projects each 3D point
    to the original image frame and maps it into that same 518×518 space to
    decide membership — no lossy bbox inversion needed.

    Args:
        points:           (N, 3) float32 XYZ in camera frame (metres).
        bbox_518:         (x1, y1, x2, y2) in 518×518 letterboxed coordinates.
        fx, fy:           Focal lengths from color CameraInfo (pixels).
        cx_cam, cy_cam:   Principal point from color CameraInfo (pixels).
        orig_w, orig_h:   Original color image size from CameraInfo.

    Returns:
        (M, 3) float32 array, M ≤ N — points whose 2D projection falls inside bbox.
    """
    if len(points) == 0:
        return points

    _INPUT = 518
    scale = min(_INPUT / orig_w, _INPUT / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (_INPUT - new_w) // 2
    pad_y = (_INPUT - new_h) // 2

    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    valid = Z > 0.0

    # Avoid division by zero on invalid points by substituting Z=1 (result masked out).
    safe_Z = np.where(valid, Z, 1.0)
    u_orig = np.where(valid, X / safe_Z * fx + cx_cam, np.nan)
    v_orig = np.where(valid, Y / safe_Z * fy + cy_cam, np.nan)

    u_518 = u_orig * scale + pad_x
    v_518 = v_orig * scale + pad_y

    x1, y1, x2, y2 = bbox_518
    mask = (
        valid
        & np.isfinite(u_518)
        & np.isfinite(v_518)
        & (u_518 >= x1)
        & (u_518 <= x2)
        & (v_518 >= y1)
        & (v_518 <= y2)
    )
    return points[mask]


# ─── Depth Pre-Filter ─────────────────────────────────────────────────────────

def filter_cluster_by_depth(
    points: np.ndarray,
    depth_window_m: float = 0.10,
) -> np.ndarray:
    """Remove background points from a bbox cluster before sphere fitting.

    The tomato is always the nearest surface inside its detection bbox. Background
    (wall, shelf, leaves behind the tomato) sits at a higher Z. We keep only points
    within depth_window_m of the cluster's minimum Z — that window spans the full
    visible arc of a tomato (diameter ≤ 15 cm) while cutting the background tail.

    This is more robust than percentile filtering because it is invariant to the
    fraction of background points in the cluster.

    Args:
        points:         (N, 3) float32 cluster, already clipped to bbox.
        depth_window_m: Max Z distance from the nearest point to keep (metres).
                        0.10 m (10 cm) comfortably spans a tomato at any depth.

    Returns:
        Filtered (M, 3) array, or original if filter leaves fewer than 4 points.
    """
    if len(points) == 0:
        return points
    z_min = float(points[:, 2].min())
    mask = points[:, 2] <= z_min + depth_window_m
    filtered = points[mask]
    return filtered if len(filtered) >= 4 else points


# ─── Sphere Fitting ────────────────────────────────────────────────────────────

def fit_sphere_algebraic(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a sphere to 3D points using algebraic least squares.

    Linearizes (x-cx)^2 + (y-cy)^2 + (z-cz)^2 = r^2 into:
        2·cx·x + 2·cy·y + 2·cz·z + (r^2 - ||c||^2) = ||p||^2
    which is the linear system A @ [cx, cy, cz, d]^T = B solved with lstsq.

    Args:
        points: (N, 3) array, N ≥ 4.

    Returns:
        center: (3,) float32 in camera frame (metres).
        radius: float in metres.

    Raises:
        ValueError: if N < 4.
    """
    n = len(points)
    if n < 4:
        raise ValueError(f"Sphere fit requires ≥ 4 points; got {n}.")

    pts = points.astype(np.float64)
    A = np.column_stack([2.0 * pts, np.ones(n)])   # (N, 4)
    B = (pts ** 2).sum(axis=1)                      # (N,)

    result, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    cxf, cyf, czf = result[:3]
    d = result[3]  # = r^2 - ||center||^2
    r_sq = d + cxf ** 2 + cyf ** 2 + czf ** 2

    center = np.array([cxf, cyf, czf], dtype=np.float32)
    radius = float(np.sqrt(max(r_sq, 0.0)))
    return center, radius


def fit_sphere_ransac(
    points: np.ndarray,
    n_iterations: int = 60,
    inlier_dist: float = 0.015,
    min_inliers: int = 10,
    max_radius_m: float = 0.12,
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC sphere fitting — robust to depth noise at cluster boundaries.

    Random 4-point subsamples drive algebraic fits; inliers are points within
    inlier_dist of the sphere surface. Final fit is refined on all consensus
    inliers, which improves centre accuracy vs. the 4-point hypothesis alone.

    Tomatoes are 2–12 cm radius; fits outside [0.005, max_radius_m] are
    rejected as degenerate (e.g. three collinear points).

    Args:
        points:        (N, 3) point cloud cluster, already clipped to bbox.
        n_iterations:  RANSAC budget.
        inlier_dist:   Max |dist_to_surface| to count as inlier (metres).
        min_inliers:   Minimum consensus size to accept a hypothesis.
        max_radius_m:  Upper sanity bound on tomato radius.

    Returns:
        center:  (3,) array — sphere centroid in camera frame (metres).
        radius:  float — sphere radius in metres.
        inliers: (M, 3) array — consensus inlier points used for final fit.
    """
    if len(points) < max(4, min_inliers):
        if len(points) >= 4:
            c, r = fit_sphere_algebraic(points)
        else:
            c, r = points.mean(axis=0).astype(np.float32), 0.03
        return c, r, points

    rng = np.random.default_rng(42)  # fixed seed — deterministic run-to-run behaviour
    best_center = points.mean(axis=0).astype(np.float32)
    best_radius = 0.03
    best_count = 0
    best_mask = np.ones(len(points), dtype=bool)

    for _ in range(n_iterations):
        idx = rng.choice(len(points), size=4, replace=False)
        try:
            c, r = fit_sphere_algebraic(points[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue

        if r < 0.005 or r > max_radius_m:
            continue

        dist = np.abs(np.linalg.norm(points - c, axis=1) - r)
        mask = dist < inlier_dist
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask = mask
            try:
                best_center, best_radius = fit_sphere_algebraic(points[mask])
            except (np.linalg.LinAlgError, ValueError):
                best_center, best_radius = c, r

    if best_count < min_inliers:
        # Algebraic fallback: RANSAC never found a stable consensus.
        # The fallback fit is unconstrained — cap it so callers get a physically
        # plausible radius even when the cluster contains background clutter.
        try:
            best_center, best_radius = fit_sphere_algebraic(points)
        except (np.linalg.LinAlgError, ValueError):
            best_center = points.mean(axis=0).astype(np.float32)
            best_radius = 0.03
        best_radius = float(np.clip(best_radius, 0.005, max_radius_m))
        best_mask = np.ones(len(points), dtype=bool)

    return best_center, best_radius, points[best_mask]


# ─── Extents ───────────────────────────────────────────────────────────────────

def compute_extents(points: np.ndarray) -> tuple[float, float, float]:
    """Axis-aligned bounding box extents of a point cluster.

    Returns (width, height, depth) in metres corresponding to X, Y, Z ranges.
    The arm planner uses these for grasp aperture sizing.
    """
    if len(points) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(points[:, 0].max() - points[:, 0].min()),
        float(points[:, 1].max() - points[:, 1].min()),
        float(points[:, 2].max() - points[:, 2].min()),
    )


# ─── Color Crop ────────────────────────────────────────────────────────────────

def crop_color_to_bbox(
    bgr: np.ndarray,
    bbox_518: tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Crop the original color frame to the region corresponding to a 518×518 bbox.

    Reverses the letterbox transform from preprocess_for_dino() so that the
    crop is expressed in original camera resolution rather than model resolution.
    This gives the arm planner and Qwen-VL a full-resolution JPEG of each tomato.

    Args:
        bgr:      (H, W, 3) BGR image at original camera resolution.
        bbox_518: (x1, y1, x2, y2) in 518×518 letterboxed space.
        orig_w, orig_h: Original image dimensions (pixels).

    Returns:
        Cropped BGR image — full resolution, may be smaller if bbox clips the border.
    """
    _INPUT = 518
    scale = min(_INPUT / orig_w, _INPUT / orig_h)
    pad_x = (_INPUT - int(orig_w * scale)) // 2
    pad_y = (_INPUT - int(orig_h * scale)) // 2

    x1_518, y1_518, x2_518, y2_518 = bbox_518
    x1 = max(0, int((x1_518 - pad_x) / scale))
    y1 = max(0, int((y1_518 - pad_y) / scale))
    x2 = min(orig_w, int((x2_518 - pad_x) / scale))
    y2 = min(orig_h, int((y2_518 - pad_y) / scale))

    if x2 <= x1 or y2 <= y1:
        return bgr

    return bgr[y1:y2, x1:x2]
