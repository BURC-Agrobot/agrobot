"""
perception.launch.py — Launch file for the Agrobot perception stack.

ROS 2 launch files are Python scripts. They describe which nodes to start,
with what parameters, remappings, and in what namespace.

Run inside the container (GPU enabled — uses agrobot-tom-v2/rocm-gpu:latest):
    ros2 launch agrobot_perception perception.launch.py
    ros2 launch agrobot_perception perception.launch.py mlp_confidence_threshold:=0.5

Production config (P2.2 — current best, see REPRODUCE.md):
    ros2 launch agrobot_perception perception.launch.py \\
      depth_topic:=/camera/camera/depth/image_rect_raw \\
      depth_camera_info_topic:=/camera/camera/depth/camera_info

Why a launch file vs `ros2 run`?
- Starts multiple nodes in one command (detector + future spatial node + planner).
- Centralized parameter configuration (no long --ros-args chains).
- Handles node lifecycle, respawn policies, and namespace isolation cleanly.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ── Camera topics ──────────────────────────────────────────────────────────
    camera_topic_arg = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera/camera/color/image_raw",
        description="Color image topic from RealSense driver (sensor_msgs/Image).",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="",
        description=(
            "Aligned depth image topic (sensor_msgs/Image). "
            "Empty = 3D output disabled. "
            "Typical value: /camera/camera/depth/image_rect_raw"
        ),
    )
    depth_camera_info_arg = DeclareLaunchArgument(
        "depth_camera_info_topic",
        default_value="",
        description=(
            "CameraInfo for depth (sensor_msgs/CameraInfo). "
            "Typical value: /camera/camera/depth/camera_info"
        ),
    )

    # ── Detection thresholds ───────────────────────────────────────────────────
    conf_threshold_arg = DeclareLaunchArgument(
        "confidence_threshold",
        default_value="0.0",
        description=(
            "Pre-MLP DINOv2 score gate [0.0, 1.0]. P2.2 best: 0.0 — "
            "pass all SAM2 proposals to the MLP; mlp_confidence_threshold gates instead."
        ),
    )
    nms_iou_arg = DeclareLaunchArgument(
        "nms_iou_threshold",
        default_value="0.4",
        description="IoU threshold for box NMS. P2.2 best: 0.40.",
    )

    # ── SAM2 AMG parameters ────────────────────────────────────────────────────
    amg_points_arg = DeclareLaunchArgument(
        "amg_points_per_side",
        default_value="48",
        description=(
            "SAM2 AMG grid density. 48 → 2304 proposals. "
            "Demo-validated best for this scene/camera setup: "
            "denser coverage catches small/distant targets like the wall tomato. "
            "~16-18s/frame GPU. "
            "Use 28 if latency becomes a bottleneck."
        ),
    )
    max_detections_arg = DeclareLaunchArgument(
        "max_detections",
        default_value="30",
        description="Max tomatoes reported per frame. S4.12 best: 30.",
    )

    # ── DINOv2 scoring parameters ──────────────────────────────────────────────
    dino_weight_arg = DeclareLaunchArgument(
        "dino_score_weight",
        default_value="0.7",
        description=(
            "α in: score = α·dino_sim + (1-α)·pred_iou. "
            "S4.12 best: 0.7. 1.0 = pure DINOv2, 0.0 = pure SAM2 quality."
        ),
    )
    neg_weight_arg = DeclareLaunchArgument(
        "negative_weight",
        default_value="1.0",
        description="λ for contrastive negative suppression. S4.12 best: 1.0.",
    )

    # ── Embedding paths ────────────────────────────────────────────────────────
    query_emb_arg = DeclareLaunchArgument(
        "query_embedding_path",
        default_value="models/query_embedding_k4.pt",
        description="Path to k=4 prototype query embedding (built by build_query_embedding.py).",
    )
    neg_emb_arg = DeclareLaunchArgument(
        "negative_embedding_path",
        default_value="models/negative_embedding.pt",
        description="Path to background-mean negative embedding.",
    )
    sam2_ckpt_arg = DeclareLaunchArgument(
        "sam2_checkpoint",
        default_value="",
        description=(
            "Path to SAM2 checkpoint. Empty = auto-detect models/sam2/. "
            "Use models/sam2/sam2_tomato_finetuned.pt for point-prompt fine-tuned version."
        ),
    )

    # ── SigLIP + Fusion MLP pipeline ──────────────────────────────────────────
    siglip_enabled_arg = DeclareLaunchArgument(
        "siglip_enabled",
        default_value="true",
        description="Enable SigLIP + Fusion MLP rescoring. Set false for DINOv2-only fallback.",
    )
    siglip_model_arg = DeclareLaunchArgument(
        "siglip_model",
        default_value="google/siglip-base-patch16-224",
        description="HuggingFace SigLIP model id. Cached after first download.",
    )
    fusion_mlp_path_arg = DeclareLaunchArgument(
        "fusion_mlp_path",
        default_value="models/fusion_mlp.pt",
        description="Path to trained FusionMLP weights (relative to /workspace).",
    )
    mlp_conf_arg = DeclareLaunchArgument(
        "mlp_confidence_threshold",
        default_value="0.45",
        description=(
            "MLP output probability threshold [0,1]. GPU eval best: 0.40 (high precision). "
            "Live default: 0.45 — strict precision setting for demos. Real ripe fruit "
            "consistently scores 0.70+; this rejects 0.25–0.40 marginal proposals that "
            "produced ghost tracks in foliage scenes. Drop to 0.30 if real fruit gets "
            "missed; drop to 0.25 only for small/distant/occluded tomatoes."
        ),
    )

    # ── Debug ──────────────────────────────────────────────────────────────────
    publish_debug_arg = DeclareLaunchArgument(
        "publish_debug_image",
        default_value="true",
        description="Publish annotated debug image on /agrobot/debug_image (Foxglove).",
    )

    # ── Spatial node topics ────────────────────────────────────────────────────
    pointcloud_topic_arg = DeclareLaunchArgument(
        "pointcloud_topic",
        default_value="/camera/camera/depth/color/points",
        description=(
            "PointCloud2 topic from RealSense (requires pointcloud.enable:=true). "
            "Typical: /camera/camera/depth/color/points"
        ),
    )
    camera_info_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/camera/color/camera_info",
        description="CameraInfo for the color camera (used by tomato_spatial for intrinsics).",
    )

    # ── Tomato Detector Node (Node 1) ──────────────────────────────────────────
    tomato_detector_node = Node(
        package="agrobot_perception",
        executable="tomato_detector",
        name="tomato_detector",
        namespace="agrobot",
        output="screen",
        remappings=[
            ("/camera/image_raw", LaunchConfiguration("camera_topic")),
        ],
        parameters=[
            {
                "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
                "input_width": 518,
                "input_height": 518,
                "depth_topic": LaunchConfiguration("depth_topic"),
                "depth_camera_info_topic": LaunchConfiguration("depth_camera_info_topic"),
                "amg_points_per_side": LaunchConfiguration("amg_points_per_side"),
                "max_detections": LaunchConfiguration("max_detections"),
                "nms_iou_threshold": LaunchConfiguration("nms_iou_threshold"),
                "dino_score_weight": LaunchConfiguration("dino_score_weight"),
                "negative_weight": LaunchConfiguration("negative_weight"),
                "query_embedding_path": LaunchConfiguration("query_embedding_path"),
                "negative_embedding_path": LaunchConfiguration("negative_embedding_path"),
                "sam2_checkpoint": LaunchConfiguration("sam2_checkpoint"),
                "siglip_enabled": LaunchConfiguration("siglip_enabled"),
                "siglip_model": LaunchConfiguration("siglip_model"),
                "fusion_mlp_path": LaunchConfiguration("fusion_mlp_path"),
                "mlp_confidence_threshold": LaunchConfiguration("mlp_confidence_threshold"),
            }
        ],
    )

    # ── Tomato Spatial Node (Node 2) ───────────────────────────────────────────
    # Consumes detections from Node 1 + PointCloud2 from the RealSense driver.
    # Fits a sphere to each bbox cluster and publishes /agrobot/tomato_spatial.
    # Enabled only when pointcloud_topic is non-empty (same pattern as depth 3D).
    tomato_spatial_node = Node(
        package="agrobot_perception",
        executable="tomato_spatial",
        name="tomato_spatial",
        namespace="agrobot",
        output="screen",
        parameters=[
            {
                "pointcloud_topic": LaunchConfiguration("pointcloud_topic"),
                "color_image_topic": LaunchConfiguration("camera_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "detections_topic": "/agrobot/detections",
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
            }
        ],
    )

    # ── Qwen-VL Pick Policy Args ───────────────────────────────────────────────
    pick_policy_arg = DeclareLaunchArgument(
        "pick_policy",
        default_value="ripe_first",
        description=(
            "VLM pick policy: ripe_first | closest_first | largest_first. "
            "ripe_first: prefers red/ripe tomatoes. "
            "closest_first: picks min-z (nearest to camera). "
            "largest_first: picks largest by radius."
        ),
    )
    qwen_model_arg = DeclareLaunchArgument(
        "qwen_model_path",
        default_value="Qwen/Qwen2.5-VL-3B-Instruct",
        description=(
            "HuggingFace model ID or local path. "
            "Set to models/qwen_vl/ after first download to avoid re-fetching."
        ),
    )

    # ── Tomato Tracker Node (NODE 2b) ──────────────────────────────────────────
    # Consumes /agrobot/tomato_spatial, assigns persistent IDs across frames,
    # EMA-smooths centroids, and publishes /agrobot/tomato_tracks.
    # Qwen-VL and the arm planner subscribe to tomato_tracks, not tomato_spatial.
    tomato_tracker_node = Node(
        package="agrobot_perception",
        executable="tomato_tracker",
        name="tomato_tracker",
        namespace="agrobot",
        output="screen",
        parameters=[
            {
                "spatial_topic": "/agrobot/tomato_spatial",
                "tracks_topic": "/agrobot/tomato_tracks",
                "match_threshold_m": 0.08,
                # 5 missed cycles ≈ 30 s at the ~6 s/frame detector cadence.
                # Sweet spot for demos: long enough to survive a brief miss
                # (avoids ID-thrashing on a still tomato), short enough that a
                # moved/removed tomato clears within half a minute and a new one
                # can take the next persistent_id cleanly.
                "max_missed_frames": 5,
                "smoothing_alpha": 0.4,
            }
        ],
    )

    # ── Qwen-VL Node (NODE 3) ──────────────────────────────────────────────────
    # Subscribes to /agrobot/tomato_tracks. Loads Qwen2.5-VL-3B in a background
    # thread (~30s). Until the model is ready it logs a warning and does nothing.
    # Falls back to heuristic (closest tomato) if transformers is not installed.
    qwen_vl_node = Node(
        package="agrobot_perception",
        executable="qwen_vl",
        name="qwen_vl",
        namespace="agrobot",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("qwen_model_path"),
                "pick_policy": LaunchConfiguration("pick_policy"),
                "tracks_topic": "/agrobot/tomato_tracks",
                "min_smoothed_age": 3,
                "max_new_tokens": 64,
            }
        ],
    )

    return LaunchDescription(
        [
            camera_topic_arg,
            depth_topic_arg,
            depth_camera_info_arg,
            pointcloud_topic_arg,
            camera_info_arg,
            conf_threshold_arg,
            nms_iou_arg,
            amg_points_arg,
            max_detections_arg,
            dino_weight_arg,
            neg_weight_arg,
            query_emb_arg,
            neg_emb_arg,
            sam2_ckpt_arg,
            siglip_enabled_arg,
            siglip_model_arg,
            fusion_mlp_path_arg,
            mlp_conf_arg,
            publish_debug_arg,
            pick_policy_arg,
            qwen_model_arg,
            tomato_detector_node,
            tomato_spatial_node,
            tomato_tracker_node,
            qwen_vl_node,
        ]
    )
