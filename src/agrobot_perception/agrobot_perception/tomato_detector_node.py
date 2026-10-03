"""
tomato_detector_node.py — Tomato Detection ROS 2 Node

Architecture
------------
This node is the central perception pipeline entry point for Agrobot TOM v2.
It follows the "thin node, thick library" pattern:
  - The ROS 2 node itself (this file) is kept minimal: subscribe, preprocess,
    call detector, publish. No business logic here.
  - All image processing lives in `utils/image_utils.py`.
  - The detector is a pluggable backend (Placeholder → DINOv2+SAM2 in Sprint 2).

Data Flow
---------
  /camera/image_raw (sensor_msgs/Image)
      │
      ▼
  [cv_bridge] → BGR np.ndarray
      │
      ▼
  [preprocess_for_dino] → float32 CHW tensor
      │
      ▼
  [DINOv2SAM2Detector.detect()]   ← Sprint 3: swap for MIGraphX ONNX on ROCm
      │
      ▼
  /agrobot/detections (vision_msgs/Detection2DArray)
  /agrobot/debug_image (sensor_msgs/Image)       ← for Foxglove visualization

Topics
------
  Subscribed:
    /camera/image_raw                         sensor_msgs/Image   Raw camera frames
    <depth_topic>                             sensor_msgs/Image   Aligned depth (optional)
    <depth_camera_info_topic>                 sensor_msgs/CameraInfo (optional)

  Published:
    /agrobot/detections                       vision_msgs/Detection2DArray
    /agrobot/detections_3d                    vision_msgs/Detection3DArray (when depth enabled)
    /agrobot/safe_to_pick                     std_msgs/Bool  False = no pick this cycle
    /agrobot/debug_image                      sensor_msgs/Image (debug only)

Parameters
----------
  confidence_threshold    float   Minimum detection confidence (default: 0.5)
  input_width             int     Model input width in pixels (default: 518)
  input_height            int     Model input height in pixels (default: 518)
  publish_debug_image     bool    Whether to publish annotated debug frames (default: True)
  depth_topic             string  Optional. Aligned depth for 3D (default: "" = disabled)
  depth_camera_info_topic string  Optional. CameraInfo for depth (default: "")
  watchdog_timeout_ms     int     Publish empty detections if no frame within this window.
                                  0 = disabled. (default: 2000)
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Header
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    Detection3DArray,
    Detection3D,
    ObjectHypothesisWithPose,
)
from geometry_msgs.msg import Point

import cv2
import numpy as np

try:
    from cv_bridge import CvBridge
except ImportError:
    raise ImportError(
        "cv_bridge not found. Are you running inside the ROS 2 Docker container? "
        "Run: docker compose -f deployment/compose/docker-compose.yml run --rm dev bash"
    )

from agrobot_perception.utils.image_utils import (
    preprocess_for_dino,
    draw_detection_overlay,
)
from agrobot_perception.detectors.sam2_amg_detector import (
    SAM2AMGDetector,
    _select_device,
)


# ─── QoS Profiles ─────────────────────────────────────────────────────────────
# Camera topics from real sensors use "Best Effort" reliability — they drop
# frames rather than queue them. Using Reliable QoS here would cause a
# QoS compatibility mismatch warning and the subscription would receive nothing.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class PlaceholderDetector:
    """Stub detector — returns zero detections.

    Retained as a documented fallback. To revert to the stub:
      self._detector = PlaceholderDetector()

    Returns:
        List of dicts: [{"box": [x1,y1,x2,y2], "score": float, "label": str}]
    """

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        return []


class TomatoDetectorNode(Node):
    """ROS 2 node that detects tomatoes in camera frames."""

    def __init__(self) -> None:
        super().__init__("tomato_detector")

        # ── Parameters ────────────────────────────────────────────────────────
        # Declare all parameters with defaults. Users can override via:
        #   ros2 run agrobot_perception tomato_detector --ros-args -p confidence_threshold:=0.7
        # Or in a launch file (see launch/perception.launch.py).
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("input_width", 518)
        self.declare_parameter("input_height", 518)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("depth_camera_info_topic", "")
        # Watchdog: if no camera frame arrives within this window, publish
        # empty detections and safe_to_pick=False (FM-1 in FAILURE_MODES.md).
        # Set to 0 to disable the watchdog entirely.
        # Watchdog timeout must exceed worst-case inference time (~17s on CPU).
        # Default 60s prevents false-positives during SAM2+DINOv2 processing.
        self.declare_parameter("watchdog_timeout_ms", 60000)
        # SAM2AMGDetector parameters — GPU ablation best config (2026-05-21).
        # Override at launch: ros2 launch ... amg_points_per_side:=32
        self.declare_parameter("amg_points_per_side", 32)
        self.declare_parameter("max_detections", 30)
        self.declare_parameter("nms_iou_threshold", 0.4)
        self.declare_parameter("dino_score_weight", 0.7)
        self.declare_parameter("negative_weight", 1.0)
        self.declare_parameter("query_embedding_path", "models/query_embedding_k4.pt")
        self.declare_parameter("negative_embedding_path", "models/negative_embedding.pt")
        self.declare_parameter("sam2_checkpoint", "")
        # SigLIP + Fusion MLP pipeline (eval-matched, full accuracy).
        # Set siglip_enabled:=false to fall back to DINOv2-only scoring.
        self.declare_parameter("siglip_enabled", True)
        self.declare_parameter("siglip_model", "google/siglip-base-patch16-224")
        self.declare_parameter("fusion_mlp_path", "models/fusion_mlp.pt")
        self.declare_parameter("mlp_confidence_threshold", 0.40)
        self.declare_parameter("siglip_dino_weight", 0.4)
        self.declare_parameter("siglip_weight", 0.4)
        self.declare_parameter("siglip_pred_iou_weight", 0.2)
        # Pipe-separated overrides for SigLIP text prompts. Empty = built-in defaults.
        # In scenes with green foliage but only ripe red tomatoes, remove the
        # "green unripe tomato" positive prompt — it co-fires with leaves.
        self.declare_parameter("siglip_positive_prompts", "")
        self.declare_parameter("siglip_negative_prompts", "")
        # Optional red-color post-filter (FM-7: leaf-wall false-positives).
        # When enabled, detections whose bbox crop contains <min_red_fraction
        # red-saturated pixels are dropped. The MLP's HSV features alone do not
        # strongly bias against green in single-ripe scenes — this is a hard prior.
        self.declare_parameter("color_filter_enabled", False)
        self.declare_parameter("color_min_red_fraction", 0.15)
        self.declare_parameter("color_red_hue_max", 12)        # OpenCV H in [0, 180]
        self.declare_parameter("color_red_hue_min", 168)
        self.declare_parameter("color_red_min_saturation", 90)
        self.declare_parameter("color_red_min_value", 60)

        self._conf_threshold = self.get_parameter("confidence_threshold").value
        self._input_size = (
            self.get_parameter("input_width").value,
            self.get_parameter("input_height").value,
        )
        self._publish_debug = self.get_parameter("publish_debug_image").value
        self._depth_topic = self.get_parameter("depth_topic").value
        self._depth_info_topic = self.get_parameter("depth_camera_info_topic").value
        self._depth_image: np.ndarray | None = None
        self._depth_K: tuple[float, float, float, float] | None = None
        self._watchdog_timeout_ms: int = self.get_parameter("watchdog_timeout_ms").value
        self._last_frame_time: float = 0.0

        self._color_filter_enabled: bool = bool(self.get_parameter("color_filter_enabled").value)
        self._color_min_red_fraction: float = float(self.get_parameter("color_min_red_fraction").value)
        self._color_red_hue_max: int = int(self.get_parameter("color_red_hue_max").value)
        self._color_red_hue_min: int = int(self.get_parameter("color_red_hue_min").value)
        self._color_red_min_sat: int = int(self.get_parameter("color_red_min_saturation").value)
        self._color_red_min_val: int = int(self.get_parameter("color_red_min_value").value)

        _query_emb = self.get_parameter("query_embedding_path").value or None
        _neg_emb = self.get_parameter("negative_embedding_path").value or None
        _sam2_ckpt = self.get_parameter("sam2_checkpoint").value or None

        # ── Core Components ───────────────────────────────────────────────────
        self._bridge = CvBridge()
        _device = _select_device()
        _siglip_enabled = self.get_parameter("siglip_enabled").value
        _nms = self.get_parameter("nms_iou_threshold").value
        _max_det = self.get_parameter("max_detections").value

        # When the SigLIP+MLP pipeline is active, confidence_threshold=0.0 lets
        # all SAM2 proposals reach the MLP; the MLP's own threshold gates the
        # final output. When DINOv2-only, the raw DINOv2 confidence_threshold
        # is the gate.
        _base_conf = 0.0 if _siglip_enabled else self._conf_threshold

        self._detector = SAM2AMGDetector(
            device=_device,
            confidence_threshold=_base_conf,
            points_per_side=self.get_parameter("amg_points_per_side").value,
            max_detections=_max_det,
            nms_iou_threshold=_nms,
            dino_score_weight=self.get_parameter("dino_score_weight").value,
            negative_weight=self.get_parameter("negative_weight").value,
            query_embedding_path=_query_emb,
            negative_embedding_path=_neg_emb,
            sam2_checkpoint=_sam2_ckpt,
        )

        if _siglip_enabled:
            import os
            _mlp_path = self.get_parameter("fusion_mlp_path").value
            if not os.path.isabs(_mlp_path):
                _mlp_path = os.path.join("/workspace", _mlp_path)
            try:
                from eval.siglip_rescoring import SigLIPRescoringWrapper
                from eval.fusion_mlp import FusionMLPWrapper
                if not os.path.exists(_mlp_path):
                    raise FileNotFoundError(f"fusion_mlp not found: {_mlp_path}")
                _siglip_model = self.get_parameter("siglip_model").value
                _mlp_conf = self.get_parameter("mlp_confidence_threshold").value
                self.get_logger().info(
                    f"Loading SigLIP model {_siglip_model} (this takes ~30s)..."
                )
                _pos_raw = self.get_parameter("siglip_positive_prompts").value or ""
                _neg_raw = self.get_parameter("siglip_negative_prompts").value or ""
                _pos_prompts = tuple(p.strip() for p in _pos_raw.split("|") if p.strip())
                _neg_prompts = tuple(p.strip() for p in _neg_raw.split("|") if p.strip())
                # Build kwargs so SigLIPRescoringWrapper's built-in defaults stay
                # the source of truth when no override is supplied. Passing an
                # empty tuple here would silently disable the prompt set.
                _siglip_kwargs = {}
                if _pos_prompts:
                    _siglip_kwargs["positive_prompts"] = _pos_prompts
                if _neg_prompts:
                    _siglip_kwargs["negative_prompts"] = _neg_prompts
                self._detector = SigLIPRescoringWrapper(
                    base=self._detector,
                    model_id=_siglip_model,
                    w_dino=self.get_parameter("siglip_dino_weight").value,
                    w_siglip=self.get_parameter("siglip_weight").value,
                    w_pred_iou=self.get_parameter("siglip_pred_iou_weight").value,
                    nms_iou_threshold=_nms,
                    max_detections=_max_det,
                    confidence_threshold=0.0,
                    device=_device,
                    **_siglip_kwargs,
                )
                if _pos_prompts or _neg_prompts:
                    self.get_logger().info(
                        f"SigLIP prompt overrides — pos={list(_pos_prompts) or 'defaults'}, "
                        f"neg={list(_neg_prompts) or 'defaults'}"
                    )
                self._detector = FusionMLPWrapper(
                    base=self._detector,
                    mlp_path=_mlp_path,
                    nms_iou_threshold=_nms,
                    max_detections=_max_det,
                    confidence_threshold=_mlp_conf,
                    device=_device,
                )
                self.get_logger().info(
                    f"SigLIP + Fusion MLP pipeline active "
                    f"(mlp_conf={_mlp_conf:.2f}, "
                    f"dino={self.get_parameter('siglip_dino_weight').value:.1f}, "
                    f"siglip={self.get_parameter('siglip_weight').value:.1f}, "
                    f"pred_iou={self.get_parameter('siglip_pred_iou_weight').value:.1f})"
                )
            except Exception as exc:
                self.get_logger().warning(
                    f"SigLIP+MLP pipeline unavailable ({exc}). "
                    f"Falling back to DINOv2-only with "
                    f"confidence_threshold={self._conf_threshold:.2f}."
                )
                # Re-create base detector with the raw confidence threshold so
                # DINOv2-only mode gates correctly.
                self._detector = SAM2AMGDetector(
                    device=_device,
                    confidence_threshold=self._conf_threshold,
                    points_per_side=self.get_parameter("amg_points_per_side").value,
                    max_detections=_max_det,
                    nms_iou_threshold=_nms,
                    dino_score_weight=self.get_parameter("dino_score_weight").value,
                    negative_weight=self.get_parameter("negative_weight").value,
                    query_embedding_path=_query_emb,
                    negative_embedding_path=_neg_emb,
                    sam2_checkpoint=_sam2_ckpt,
                )

        # ── Subscribers ───────────────────────────────────────────────────────
        self._image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_callback,
            SENSOR_QOS,
        )

        if self._depth_topic and self._depth_info_topic:
            self._depth_sub = self.create_subscription(
                Image, self._depth_topic, self._depth_callback, SENSOR_QOS
            )
            self._depth_info_sub = self.create_subscription(
                CameraInfo,
                self._depth_info_topic,
                self._depth_info_callback,
                10,
            )
            self._detections_3d_pub = self.create_publisher(
                Detection3DArray, "/agrobot/detections_3d", 10
            )

        # ── Publishers ────────────────────────────────────────────────────────
        self._detections_pub = self.create_publisher(
            Detection2DArray,
            "/agrobot/detections",
            10,
        )
        # safe_to_pick: False when no detections or watchdog triggered.
        # The arm planner subscribes here to gate pick attempts.
        self._safe_to_pick_pub = self.create_publisher(
            Bool,
            "/agrobot/safe_to_pick",
            10,
        )

        if self._publish_debug:
            self._debug_pub = self.create_publisher(
                Image,
                "/agrobot/debug_image",
                1,
            )

        # ── Watchdog timer ────────────────────────────────────────────────────
        if self._watchdog_timeout_ms > 0:
            self._watchdog_timer = self.create_timer(
                self._watchdog_timeout_ms / 1000.0,
                self._watchdog_callback,
            )
            self.get_logger().info(
                f"Watchdog enabled: safe_to_pick=False if no frame "
                f"in {self._watchdog_timeout_ms} ms."
            )

        if self._color_filter_enabled:
            self.get_logger().info(
                f"Red-color post-filter ENABLED: min_red_fraction="
                f"{self._color_min_red_fraction:.2f}, "
                f"hue ∈ [0,{self._color_red_hue_max}]∪[{self._color_red_hue_min},180], "
                f"S≥{self._color_red_min_sat}, V≥{self._color_red_min_val}."
            )

        self.get_logger().info(
            f"TomatoDetectorNode initialized. "
            f"conf_threshold={self._conf_threshold}, "
            f"input_size={self._input_size}"
        )

    def _watchdog_callback(self) -> None:
        """Fires if no camera frame has arrived within watchdog_timeout_ms.

        Publishes empty detections + safe_to_pick=False so the planner gets
        an explicit signal rather than silence. See FM-1 in docs/FAILURE_MODES.md.
        """
        if self._last_frame_time == 0.0:
            # Node just started, no frame ever received yet — don't alarm.
            return

        elapsed_ms = (time.monotonic() - self._last_frame_time) * 1000.0
        if elapsed_ms >= self._watchdog_timeout_ms:
            self.get_logger().warn(
                f"No camera frame for {elapsed_ms:.0f} ms "
                f"(timeout={self._watchdog_timeout_ms} ms). "
                "Publishing empty detections — safe_to_pick=False. "
                "Check /camera/image_raw and the RealSense driver.",
                throttle_duration_sec=1.0,
            )
            self._publish_detections([], Header())
            self._publish_safe_to_pick(False)

    def _publish_safe_to_pick(self, safe: bool) -> None:
        msg = Bool()
        msg.data = safe
        self._safe_to_pick_pub.publish(msg)

    def _bbox_redness(self, bgr_frame: np.ndarray, bbox_518: list[float]) -> float:
        """Return the fraction [0,1] of bbox pixels that look saturated-red.

        The detector returns boxes in 518×518 letterboxed space. This helper
        un-letterboxes back to the native BGR frame, then masks pixels whose
        OpenCV hue lies in the red band (red wraps H=0/180) with high enough
        saturation and value to exclude shadow noise. Cheap (~µs per box) and
        operates on raw camera pixels, not the ImageNet-normalised tensor — so
        the colour stats are not corrupted by ViT preprocessing.
        """
        h, w = bgr_frame.shape[:2]
        iw, ih = self._input_size
        scale = min(iw / w, ih / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_x = (iw - new_w) // 2
        pad_y = (ih - new_h) // 2

        x1_518, y1_518, x2_518, y2_518 = bbox_518
        x1 = max(0, int((x1_518 - pad_x) / scale))
        y1 = max(0, int((y1_518 - pad_y) / scale))
        x2 = min(w, int((x2_518 - pad_x) / scale))
        y2 = min(h, int((y2_518 - pad_y) / scale))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        crop = bgr_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]
        red_mask = (
            ((h_chan <= self._color_red_hue_max) | (h_chan >= self._color_red_hue_min))
            & (s_chan >= self._color_red_min_sat)
            & (v_chan >= self._color_red_min_val)
        )
        return float(red_mask.mean())

    def _image_callback(self, msg: Image) -> None:
        """Called for every incoming camera frame."""
        self._last_frame_time = time.monotonic()

        try:
            bgr_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        preprocessed = preprocess_for_dino(bgr_frame, input_size=self._input_size)
        raw_detections = self._detector.detect(preprocessed)
        detections = [d for d in raw_detections if d["score"] >= self._conf_threshold]

        if self._color_filter_enabled and detections:
            n_before = len(detections)
            scored = [
                (d, self._bbox_redness(bgr_frame, d["box"])) for d in detections
            ]
            detections = [
                d for d, red in scored if red >= self._color_min_red_fraction
            ]
            if n_before > 0 and not detections:
                # Surface the best red-score seen so operators can tune the threshold
                # without re-running with debug logging enabled.
                best = max(scored, key=lambda kv: kv[1])
                self.get_logger().info(
                    f"Color filter dropped all {n_before} detections "
                    f"(best red_fraction={best[1]:.3f} < "
                    f"{self._color_min_red_fraction:.3f}).",
                    throttle_duration_sec=5.0,
                )
            elif n_before > 0:
                self.get_logger().debug(
                    f"Color filter kept {len(detections)}/{n_before} detections."
                )

        self._publish_detections(detections, msg.header)
        # Explicit safe_to_pick signal every frame — planner doesn't need to
        # infer from detection count; it reads this directly.
        self._publish_safe_to_pick(len(detections) > 0)

        if self._depth_image is not None and self._depth_K is not None and detections:
            self._publish_detections_3d(
                detections, msg.header, bgr_frame.shape, self._input_size
            )

        if self._publish_debug:
            self._publish_debug_image(bgr_frame, detections, msg.header)

    def _publish_detections(self, detections: list[dict], header: Header) -> None:
        """Build and publish a vision_msgs/Detection2DArray message."""
        array_msg = Detection2DArray()
        array_msg.header = header

        for det in detections:
            detection = Detection2D()
            detection.header = header

            # Bounding box center + size (vision_msgs convention).
            x1, y1, x2, y2 = det["box"]
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)

            # Hypothesis: class label + confidence.
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det["label"]
            hypothesis.hypothesis.score = float(det["score"])
            detection.results.append(hypothesis)

            array_msg.detections.append(detection)

        self._detections_pub.publish(array_msg)

        if detections:
            self.get_logger().debug(f"Published {len(detections)} detection(s).")

    def _publish_debug_image(
        self,
        bgr_frame: np.ndarray,
        detections: list[dict],
        header: Header,
    ) -> None:
        """Annotate the frame with bounding boxes and publish for visualization."""
        annotated = bgr_frame.copy()
        if detections:
            draw_detection_overlay(
                annotated,
                boxes=[d["box"] for d in detections],
                labels=[d["label"] for d in detections],
                scores=[d["score"] for d in detections],
            )
        debug_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = header
        self._debug_pub.publish(debug_msg)

    def _depth_callback(self, msg: Image) -> None:
        try:
            self._depth_image = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception:
            self._depth_image = None

    def _depth_info_callback(self, msg: CameraInfo) -> None:
        K = msg.k
        if len(K) >= 6:
            self._depth_K = (float(K[0]), float(K[4]), float(K[2]), float(K[5]))

    def _publish_detections_3d(
        self,
        detections: list[dict],
        header: Header,
        color_shape: tuple,
        input_size: tuple[int, int],
    ) -> None:
        if self._depth_image is None or self._depth_K is None:
            return
        fx, fy, cx, cy = self._depth_K
        h, w = color_shape[:2]
        iw, ih = input_size
        scale = min(iw / w, ih / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_x = (iw - new_w) // 2
        pad_y = (ih - new_h) // 2

        array_3d = Detection3DArray()
        array_3d.header = header

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx_518 = (x1 + x2) / 2.0
            cy_518 = (y1 + y2) / 2.0
            u_orig = (cx_518 - pad_x) / scale
            v_orig = (cy_518 - pad_y) / scale
            u_int = int(round(u_orig))
            v_int = int(round(v_orig))
            if u_int < 0 or u_int >= self._depth_image.shape[1] or v_int < 0 or v_int >= self._depth_image.shape[0]:
                continue
            z = float(self._depth_image[v_int, u_int])
            if z <= 0 or not np.isfinite(z):
                continue
            x_cam = (u_orig - cx) * z / fx
            y_cam = (v_orig - cy) * z / fy

            d3 = Detection3D()
            d3.header = header
            d3.bbox.center.position.x = x_cam
            d3.bbox.center.position.y = y_cam
            d3.bbox.center.position.z = z
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det["label"]
            hyp.hypothesis.score = float(det["score"])
            d3.results.append(hyp)
            array_3d.detections.append(d3)

        if array_3d.detections:
            self._detections_3d_pub.publish(array_3d)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TomatoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
