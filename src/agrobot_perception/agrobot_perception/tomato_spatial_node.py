"""
tomato_spatial_node.py — Tomato 3D Spatial Localization ROS 2 Node

Architecture
------------
NODE 2 in the Agrobot TOM v2 picking pipeline. Consumes 2D detections from
tomato_detector_node and the aligned PointCloud2 from the RealSense driver.
For each detection bbox it clips the point cloud, fits a sphere via RANSAC
algebraic least squares, and publishes rich 3D localization data for the arm
planner (Dani) and Qwen-VL reasoner (Sprint 4).

Why a separate node from tomato_detector_node?
  - The detector runs at ~0.05 Hz on CPU (SAM2 AMG is slow). Point cloud
    processing is fast (<50 ms). Coupling them would stall the spatial pipeline
    for 17s per frame.
  - Dani's arm planner needs a versioned, stable JSON interface independent of
    which perception model is currently active.
  - Qwen-VL (Sprint 4) subscribes to /agrobot/tomato_spatial directly and reads
    clipped_image; it doesn't need to know anything about SAM2 or DINOv2.

Data Flow
---------
  /camera/.../depth/color/points (PointCloud2)  ──┐
  /camera/.../color/image_raw    (Image)          ├─ cached latest
  /camera/.../color/camera_info  (CameraInfo)   ──┘ (used when detections arrive)
      │
  /agrobot/detections (Detection2DArray)
      │
      ▼
  [parse PointCloud2 → Nx3 numpy array]
      │
      ▼  (for each detection bbox)
  [clip_points_to_bbox]  ← project 3D→2D in 518×518 space, keep inside bbox
      │
      ▼
  [fit_sphere_ransac]    ← RANSAC algebraic LS, refine on inliers
      │
      ▼
  [compute_extents]      ← axis-aligned AABB of inlier cluster
      │
      ▼
  /agrobot/tomato_spatial (std_msgs/String — JSON array)
  /agrobot/spatial_debug_image  (sensor_msgs/Image — optional)

Topics
------
  Subscribed:
    <detections_topic>     vision_msgs/Detection2DArray  2D bboxes from detector
    <pointcloud_topic>     sensor_msgs/PointCloud2       Aligned XYZ+RGB cloud
    <color_image_topic>    sensor_msgs/Image             Color frame for JPEG crops
    <camera_info_topic>    sensor_msgs/CameraInfo        Intrinsics + image size

  Published:
    /agrobot/tomato_spatial        std_msgs/String  JSON array (see schema below)
    /agrobot/spatial_debug_image   sensor_msgs/Image  (when publish_debug_image=true)

JSON Schema (one element per tomato)
--------------------------------------
  {
    "tomato_id":    int,
    "centroid":     {"x": float, "y": float, "z": float},   ← camera frame, metres
    "sphere":       {"radius": float, "width": float,
                     "height": float, "depth": float},       ← metres
    "confidence":   float,                                   ← from detector score
    "clipped_image": str | null                              ← base64 JPEG crop
  }

Parameters
----------
  pointcloud_topic      string  PointCloud2 topic (default: /camera/camera/depth/color/points)
  color_image_topic     string  Color image topic (default: /camera/camera/color/image_raw)
  camera_info_topic     string  CameraInfo topic (default: /camera/camera/color/camera_info)
  detections_topic      string  2D detections topic (default: /agrobot/detections)
  min_cluster_points    int     Min points in a cluster to attempt sphere fit (default: 15)
  ransac_iterations     int     RANSAC iterations (default: 60)
  ransac_inlier_dist_m  float   Max surface residual counted as inlier, metres (default: 0.015)
  publish_debug_image   bool    Publish annotated debug image (default: True)
  max_cloud_age_s       float   Warn if cached cloud is older than this, seconds (default: 2.0)
"""

from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header, String
from vision_msgs.msg import Detection2DArray

try:
    from cv_bridge import CvBridge
except ImportError:
    raise ImportError(
        "cv_bridge not found. Are you running inside the ROS 2 Docker container? "
        "Run: docker compose -f deployment/compose/docker-compose.yml run --rm dev bash"
    )

try:
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:
    raise ImportError(
        "sensor_msgs_py not found. "
        "Install it inside the container: apt-get install ros-jazzy-sensor-msgs-py"
    )

from agrobot_perception.utils.pointcloud_utils import (
    clip_points_to_bbox,
    compute_extents,
    crop_color_to_bbox,
    filter_cluster_by_depth,
    fit_sphere_ransac,
)


# ─── QoS Profiles ─────────────────────────────────────────────────────────────
# Camera sensor topics publish with Best Effort + Volatile. Using Reliable QoS
# would cause a QoS mismatch warning and the subscription would receive nothing.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class TomatoSpatialNode(Node):
    """ROS 2 node that localizes detected tomatoes in 3D using sphere fitting."""

    def __init__(self) -> None:
        super().__init__("tomato_spatial")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter(
            "pointcloud_topic", "/camera/camera/depth/color/points"
        )
        self.declare_parameter(
            "color_image_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter("detections_topic", "/agrobot/detections")
        # Minimum cluster size: too few points mean the bbox clipped a very small
        # patch (e.g. a partially occluded tomato) — sphere fit would be unreliable.
        self.declare_parameter("min_cluster_points", 15)
        self.declare_parameter("ransac_iterations", 60)
        # 1.5 cm inlier threshold matches typical RealSense D456 depth noise at 1 m.
        self.declare_parameter("ransac_inlier_dist_m", 0.015)
        self.declare_parameter("publish_debug_image", True)
        # The detector takes ~17s per frame on CPU; the point cloud should still
        # be recent relative to wall clock (camera publishes at 15–30 Hz).
        self.declare_parameter("max_cloud_age_s", 2.0)

        pc_topic: str = self.get_parameter("pointcloud_topic").value
        color_topic: str = self.get_parameter("color_image_topic").value
        info_topic: str = self.get_parameter("camera_info_topic").value
        det_topic: str = self.get_parameter("detections_topic").value
        self._min_pts: int = self.get_parameter("min_cluster_points").value
        self._ransac_iters: int = self.get_parameter("ransac_iterations").value
        self._ransac_dist: float = self.get_parameter("ransac_inlier_dist_m").value
        self._pub_debug: bool = self.get_parameter("publish_debug_image").value
        self._max_cloud_age: float = self.get_parameter("max_cloud_age_s").value

        # ── State ─────────────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._latest_cloud: PointCloud2 | None = None
        self._cloud_recv_time: float = 0.0
        self._latest_bgr: np.ndarray | None = None
        self._color_orig_wh: tuple[int, int] | None = None
        # Camera intrinsics: populated on first CameraInfo message and then stable.
        self._cam_info: dict | None = None

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            PointCloud2, pc_topic, self._cloud_callback, SENSOR_QOS
        )
        self.create_subscription(
            Image, color_topic, self._color_callback, SENSOR_QOS
        )
        # Reliable QoS for CameraInfo — it publishes infrequently and we need it.
        self.create_subscription(CameraInfo, info_topic, self._info_callback, 10)
        self.create_subscription(
            Detection2DArray, det_topic, self._detections_callback, 10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._spatial_pub = self.create_publisher(
            String, "/agrobot/tomato_spatial", 10
        )
        if self._pub_debug:
            self._debug_pub = self.create_publisher(
                Image, "/agrobot/spatial_debug_image", 1
            )

        self.get_logger().info(
            f"TomatoSpatialNode initialized. "
            f"detections='{det_topic}' pointcloud='{pc_topic}'"
        )

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _cloud_callback(self, msg: PointCloud2) -> None:
        self._latest_cloud = msg
        self._cloud_recv_time = time.monotonic()

    def _color_callback(self, msg: Image) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._latest_bgr = bgr
            self._color_orig_wh = (bgr.shape[1], bgr.shape[0])
        except Exception as exc:
            self.get_logger().error(f"cv_bridge color conversion failed: {exc}")

    def _info_callback(self, msg: CameraInfo) -> None:
        # Intrinsics are constant for a given camera session; only store once.
        if self._cam_info is not None:
            return
        K = msg.k
        if len(K) < 6:
            return
        self._cam_info = {
            "fx": float(K[0]),
            "fy": float(K[4]),
            "cx": float(K[2]),
            "cy": float(K[5]),
            "width": int(msg.width),
            "height": int(msg.height),
        }
        self.get_logger().info(
            f"Camera intrinsics cached: "
            f"fx={self._cam_info['fx']:.1f} fy={self._cam_info['fy']:.1f} "
            f"cx={self._cam_info['cx']:.1f} cy={self._cam_info['cy']:.1f} "
            f"res={self._cam_info['width']}×{self._cam_info['height']}"
        )

    def _detections_callback(self, msg: Detection2DArray) -> None:
        """Main processing callback — triggered by each Detection2DArray from NODE 1."""
        if self._latest_cloud is None:
            self.get_logger().warn(
                "Detections received but no point cloud cached yet. "
                "Is pointcloud_topic publishing?",
                throttle_duration_sec=5.0,
            )
            return

        if self._cam_info is None:
            self.get_logger().warn(
                "Detections received but CameraInfo not yet cached. "
                "Waiting for first message on camera_info_topic.",
                throttle_duration_sec=5.0,
            )
            return

        cloud_age = time.monotonic() - self._cloud_recv_time
        if cloud_age > self._max_cloud_age:
            self.get_logger().warn(
                f"Cached point cloud is {cloud_age:.1f}s old "
                f"(threshold={self._max_cloud_age}s). "
                "3D estimates may not match the current scene frame.",
                throttle_duration_sec=10.0,
            )

        # Parse the full cloud once and share the numpy array across all detections.
        # read_points() returns a structured numpy array with named fields ('x','y','z')
        # rather than a plain (N,3) array — itemsize=20 because the PointCloud2 wire
        # format includes padding bytes even when only xyz fields are requested.
        # We extract each field by name before stacking into a plain float32 array.
        try:
            raw_struct = np.array(
                list(
                    pc2.read_points(
                        self._latest_cloud,
                        field_names=("x", "y", "z"),
                        skip_nans=True,
                    )
                )
            )
        except Exception as exc:
            self.get_logger().error(f"PointCloud2 parsing failed: {exc}")
            return

        if raw_struct.size == 0:
            self.get_logger().warn(
                "Empty point cloud — nothing to fit.", throttle_duration_sec=5.0
            )
            return

        all_points = np.column_stack(
            [raw_struct["x"], raw_struct["y"], raw_struct["z"]]
        ).astype(np.float32)  # (N, 3)
        # Hard depth bounds: drop points behind the camera or beyond arm reach.
        all_points = all_points[
            (all_points[:, 2] > 0.05) & (all_points[:, 2] < 5.0)
        ]

        ci = self._cam_info
        orig_w, orig_h = ci["width"], ci["height"]
        results: list[dict] = []

        for i, det in enumerate(msg.detections):
            # Reconstruct the (x1, y1, x2, y2) bbox from Detection2D center + size.
            # All coordinates are in 518×518 letterboxed space (preprocess_for_dino).
            cxd = det.bbox.center.position.x
            cyd = det.bbox.center.position.y
            sw = det.bbox.size_x
            sh = det.bbox.size_y
            bbox_518 = (cxd - sw / 2.0, cyd - sh / 2.0, cxd + sw / 2.0, cyd + sh / 2.0)

            cluster = clip_points_to_bbox(
                all_points,
                bbox_518,
                ci["fx"],
                ci["fy"],
                ci["cx"],
                ci["cy"],
                orig_w,
                orig_h,
            )

            if len(cluster) < self._min_pts:
                self.get_logger().debug(
                    f"Tomato {i}: {len(cluster)} points in cluster "
                    f"(min={self._min_pts}) — skipping (likely occluded or near edge)."
                )
                continue

            # Strip background clutter before fitting — at close range (<0.6 m)
            # the bbox clip includes wall/surface points at higher Z that inflate
            # the sphere radius. Keep only the near-depth portion of the cluster.
            cluster = filter_cluster_by_depth(cluster)

            center, radius, inliers = fit_sphere_ransac(
                cluster,
                n_iterations=self._ransac_iters,
                inlier_dist=self._ransac_dist,
            )

            # Tomatoes are 2–12 cm radius (up to 24cm diameter covers beefsteak).
            # A fit outside this range means residual background clutter or a
            # false-positive detection with no spherical structure.
            if not (0.015 <= radius <= 0.12):
                self.get_logger().debug(
                    f"Tomato {i}: radius={radius:.3f}m outside [0.015, 0.075] — "
                    "sphere fit invalid, skipping."
                )
                continue

            width_m, height_m, depth_m = compute_extents(inliers)

            score = (
                float(det.results[0].hypothesis.score) if det.results else 0.0
            )

            jpeg_b64: str | None = None
            if self._latest_bgr is not None:
                crop = crop_color_to_bbox(
                    self._latest_bgr, bbox_518, orig_w, orig_h
                )
                ok, buf = cv2.imencode(
                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if ok:
                    jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            results.append(
                {
                    "tomato_id": i,
                    "centroid": {
                        "x": round(float(center[0]), 4),
                        "y": round(float(center[1]), 4),
                        "z": round(float(center[2]), 4),
                    },
                    "sphere": {
                        "radius": round(radius, 4),
                        "width": round(width_m, 4),
                        "height": round(height_m, 4),
                        "depth": round(depth_m, 4),
                    },
                    "confidence": round(score, 4),
                    "clipped_image": jpeg_b64,
                }
            )

        out_msg = String()
        out_msg.data = json.dumps(results)
        self._spatial_pub.publish(out_msg)

        self.get_logger().info(
            f"Published {len(results)}/{len(msg.detections)} tomato spatial estimates."
        )

        if self._pub_debug and self._latest_bgr is not None and results:
            self._publish_debug(results, msg.header, orig_w, orig_h)

    # ── Debug publisher ────────────────────────────────────────────────────────

    def _publish_debug(
        self,
        results: list[dict],
        header: Header,
        orig_w: int,
        orig_h: int,
    ) -> None:
        """Project fitted sphere centroids back to the color image for Foxglove."""
        vis = self._latest_bgr.copy()
        ci = self._cam_info
        if ci is None:
            return

        fx, fy = ci["fx"], ci["fy"]
        cx_cam, cy_cam = ci["cx"], ci["cy"]

        for r in results:
            cx3 = r["centroid"]["x"]
            cy3 = r["centroid"]["y"]
            cz3 = r["centroid"]["z"]
            if cz3 <= 0:
                continue

            u = int(cx3 / cz3 * fx + cx_cam)
            v = int(cy3 / cz3 * fy + cy_cam)
            # Project sphere radius (metres) to pixels at the centroid depth.
            radius_px = max(2, int(r["sphere"]["radius"] / cz3 * fx))

            if 0 <= u < orig_w and 0 <= v < orig_h:
                cv2.circle(vis, (u, v), radius_px, (0, 200, 255), 2)
                cv2.circle(vis, (u, v), 4, (0, 200, 255), -1)
                label = (
                    f"z={cz3:.2f}m  r={r['sphere']['radius'] * 100:.1f}cm"
                )
                cv2.putText(
                    vis,
                    label,
                    (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

        debug_msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        debug_msg.header = header
        self._debug_pub.publish(debug_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TomatoSpatialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
