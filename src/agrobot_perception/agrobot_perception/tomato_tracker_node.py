"""
tomato_tracker_node.py — Persistent Tomato Track Management (NODE 2b)

Architecture
------------
Sits between tomato_spatial_node (NODE 2) and the arm planner / Qwen-VL (NODE 3/4).
Consumes the per-frame JSON from /agrobot/tomato_spatial and maintains a registry
of persistent tracks across frames so that:

  1. Each physical tomato keeps the same ID across detection cycles.
     Without this, tomato_id resets to 0,1,2 every frame (detection order).
     With tracking, persistent_id=3 refers to the same physical tomato
     across 10 consecutive frames until it is picked or leaves the scene.

  2. Centroid noise is suppressed via exponential moving average (EMA).
     Sphere-fit jitter of ±3 cm at 0.5 m depth is averaged away over
     3–4 frames, giving the arm planner a stable pick target.

  3. Picked tomatoes are removed from the active list.
     The arm planner publishes to /agrobot/mark_picked with the persistent_id.
     That track is then suppressed from /agrobot/tomato_tracks so Qwen-VL and
     the planner don't re-select an empty location on the next cycle.

  4. Camera motion is compensated before matching.
     For a rail-mounted camera doing step-and-shoot scanning, every tomato in
     camera-frame coordinates appears to move by −Δ_camera between cycles. The
     arm controller publishes its actual displacement on /agrobot/camera_motion
     (geometry_msgs/Vector3 in camera optical frame) and the tracker accumulates
     it. Before Hungarian matching, every existing track centroid is shifted by
     −Δ_accumulated — putting it in the predicted location for the new frame.
     This restores the diagonal-dominant cost matrix that Hungarian needs to
     work, so persistent_ids survive the entire rail scan instead of being
     destroyed on every move. The accumulator resets after each match cycle.

Algorithm (Hungarian bipartite matching in 3D):
  Build cost matrix C[i][j] = Euclidean distance between track i and detection j.
  Set C[i][j] = INF (1e9) when distance exceeds match_threshold_m — these pairs
  are forbidden, equivalent to edge pre-filtering in bipartite matching literature.
  Run scipy.optimize.linear_sum_assignment(C) to find the globally optimal
  one-to-one assignment minimising total cost (O(n³), trivial for n < 15).
  Accept matched pair (i, j) only if C[i][j] < threshold (rejects INF pairs).
  Unmatched detections → new tracks. Unmatched tracks → increment missed counter.

  Why Hungarian over greedy nearest-neighbour:
  Greedy is order-dependent — it claims the locally best match first, which
  can steal a track from a better global assignment. Hungarian is optimal:
  it minimises the sum of all matched distances simultaneously, preventing
  ID-swaps when tomatoes are close together (e.g. 5+ tomatoes on a vine).

Data Flow
---------
  /agrobot/tomato_spatial (String JSON)
      │
      ▼
  TomatoTrackerNode._spatial_callback()
      │  nearest-neighbour match + EMA update
      ▼
  /agrobot/tomato_tracks (String JSON)
      │  same schema as /agrobot/tomato_spatial, with extra fields:
      │    persistent_id  int    stable ID across frames
      │    age            int    how many frames this track has been active
      │    smoothed       bool   True once EMA has 3+ observations
      ▼
  Qwen-VL node / Arm planner (NODE 3 / 4)

Topics
------
  Subscribed:
    /agrobot/tomato_spatial   std_msgs/String         Per-frame sphere-fit JSON
    /agrobot/mark_picked      std_msgs/String         JSON {"persistent_id": N}
                                                      Published by arm planner after pick
    /agrobot/camera_motion    geometry_msgs/Vector3   Camera displacement in camera
                                                      optical frame since last publish.
                                                      Publish AFTER each rail move,
                                                      BEFORE the next spatial frame.
                                                      Multiple publishes between frames
                                                      are summed (vector accumulation).
    /agrobot/reset_tracker    std_msgs/String         Wipe all tracks + restart IDs (any payload)

  Published:
    /agrobot/tomato_tracks    std_msgs/String         Tracked + smoothed JSON array

Parameters
----------
  match_threshold_m    float   Max centroid distance to match a track (default: 0.08 m)
  max_missed_frames    int     Drop track after this many missed frames (default: 3)
  smoothing_alpha      float   EMA weight for new observation (default: 0.4)
                               Lower = smoother but more lag. 0.4 converges in ~3 frames.
  spatial_topic        string  Input topic (default: /agrobot/tomato_spatial)
  tracks_topic         string  Output topic (default: /agrobot/tomato_tracks)
"""

from __future__ import annotations

import json
import math
import threading

import numpy as np
from scipy.optimize import linear_sum_assignment

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import String


# ─── Track Registry ────────────────────────────────────────────────────────────

class Track:
    """One persistent tomato track.

    Centroid is maintained as a smoothed EMA estimate. The most recent
    clipped_image JPEG is carried through for Qwen-VL consumption.
    """

    def __init__(
        self,
        persistent_id: int,
        tomato: dict,
        alpha: float,
    ) -> None:
        self.persistent_id = persistent_id
        self.centroid = dict(tomato["centroid"])       # {"x", "y", "z"}
        self.sphere = dict(tomato["sphere"])
        self.confidence = tomato["confidence"]
        self.clipped_image = tomato.get("clipped_image")
        self.missed_frames = 0
        self.age = 1
        self._alpha = alpha
        self._obs = 1                                  # observations so far

    def predict_motion(self, dx: float, dy: float, dz: float) -> None:
        """Shift the track centroid by -(dx, dy, dz) to compensate for camera motion.

        Sign convention: if the camera moved by +Δ in camera optical frame, every
        static tomato in that frame appears at its previous position − Δ. We pre-shift
        the track centroid to where we expect to see the tomato in the new frame,
        so Hungarian matching can find the diagonal-dominant solution.

        Pure translation only — the camera is assumed to keep its orientation between
        captures (true for a linear rail). Rotational motion would need a full rigid
        transform of the centroid vector.
        """
        self.centroid["x"] -= dx
        self.centroid["y"] -= dy
        self.centroid["z"] -= dz

    def update(self, tomato: dict) -> None:
        """Apply EMA update from a new matched observation."""
        alpha = self._alpha
        c_new = tomato["centroid"]
        # EMA: new_est = alpha * observation + (1 - alpha) * old_est
        self.centroid["x"] = alpha * c_new["x"] + (1 - alpha) * self.centroid["x"]
        self.centroid["y"] = alpha * c_new["y"] + (1 - alpha) * self.centroid["y"]
        self.centroid["z"] = alpha * c_new["z"] + (1 - alpha) * self.centroid["z"]

        # Sphere extents: update with same EMA so radius converges
        s_new = tomato["sphere"]
        for key in ("radius", "width", "height", "depth"):
            self.sphere[key] = (
                alpha * s_new[key] + (1 - alpha) * self.sphere[key]
            )

        self.confidence = tomato["confidence"]
        # Always keep the freshest JPEG for Qwen-VL
        if tomato.get("clipped_image"):
            self.clipped_image = tomato["clipped_image"]

        self.missed_frames = 0
        self.age += 1
        self._obs += 1

    def to_dict(self) -> dict:
        return {
            "persistent_id": self.persistent_id,
            "tomato_id": self.persistent_id,       # alias so consumers see a stable id
            "centroid": {k: round(v, 4) for k, v in self.centroid.items()},
            "sphere": {k: round(v, 4) for k, v in self.sphere.items()},
            "confidence": round(self.confidence, 4),
            "clipped_image": self.clipped_image,
            "age": self.age,
            "smoothed": self._obs >= 3,
        }


def _euclidean(a: dict, b: dict) -> float:
    """3D Euclidean distance between two centroid dicts {x, y, z}."""
    return math.sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


# ─── Node ──────────────────────────────────────────────────────────────────────

class TomatoTrackerNode(Node):
    """ROS 2 node that assigns persistent IDs to tomato detections across frames."""

    def __init__(self) -> None:
        super().__init__("tomato_tracker")

        # ── Parameters ────────────────────────────────────────────────────────
        # 8 cm threshold: wider than ±3 cm sphere-fit jitter but tighter than
        # typical tomato-to-tomato spacing (~10–15 cm on a vine).
        self.declare_parameter("match_threshold_m", 0.08)
        # 3 missed frames ≈ 51 s on CPU. A picked tomato is gone within 1 cycle.
        self.declare_parameter("max_missed_frames", 3)
        # alpha=0.4: new observation weighted 40%, history 60%. Converges in ~4 frames.
        self.declare_parameter("smoothing_alpha", 0.4)
        self.declare_parameter("spatial_topic", "/agrobot/tomato_spatial")
        self.declare_parameter("tracks_topic", "/agrobot/tomato_tracks")

        self._threshold: float = self.get_parameter("match_threshold_m").value
        self._max_missed: int = self.get_parameter("max_missed_frames").value
        self._alpha: float = self.get_parameter("smoothing_alpha").value
        spatial_topic: str = self.get_parameter("spatial_topic").value
        tracks_topic: str = self.get_parameter("tracks_topic").value

        # ── State ─────────────────────────────────────────────────────────────
        # tracks: persistent_id → Track
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 0
        self._frame: int = 0
        # picked_ids: suppressed from output until track is naturally lost
        self._picked_ids: set[int] = set()

        # Camera motion accumulator (camera optical frame, metres). Holds the
        # cumulative displacement since the last spatial frame. Protected by a
        # lock because motion and spatial messages arrive on different threads
        # in MultiThreadedExecutor setups; under the default single-threaded
        # executor the lock is essentially free.
        self._motion_lock = threading.Lock()
        self._pending_motion: list[float] = [0.0, 0.0, 0.0]

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            String, spatial_topic, self._spatial_callback, 10
        )
        # Arm planner publishes {"persistent_id": N} here after a successful pick.
        self.create_subscription(
            String, "/agrobot/mark_picked", self._mark_picked_callback, 10
        )
        # Dashboard reset button publishes here to wipe all tracks and restart IDs from 0.
        self.create_subscription(
            String, "/agrobot/reset_tracker", self._reset_callback, 10
        )
        # Arm controller publishes Vector3 camera displacement (camera optical frame,
        # metres) after each rail move. Multiple publishes between spatial frames are
        # summed. On the next spatial callback the accumulator is consumed and reset.
        self.create_subscription(
            Vector3, "/agrobot/camera_motion", self._camera_motion_callback, 10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._tracks_pub = self.create_publisher(String, tracks_topic, 10)

        self.get_logger().info(
            f"TomatoTrackerNode initialized. "
            f"threshold={self._threshold*100:.0f}cm  "
            f"max_missed={self._max_missed}  "
            f"alpha={self._alpha}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _reset_callback(self, msg: String) -> None:
        """Wipe all tracks and restart the persistent ID counter from 0."""
        self._tracks.clear()
        self._picked_ids.clear()
        self._next_id = 0
        self._frame = 0
        with self._motion_lock:
            self._pending_motion = [0.0, 0.0, 0.0]
        self.get_logger().info("Tracker reset — all tracks cleared, IDs restart from 0.")

    def _camera_motion_callback(self, msg: Vector3) -> None:
        """Accumulate camera displacement since the last spatial callback."""
        with self._motion_lock:
            self._pending_motion[0] += float(msg.x)
            self._pending_motion[1] += float(msg.y)
            self._pending_motion[2] += float(msg.z)

    def _mark_picked_callback(self, msg: String) -> None:
        """Arm planner calls this to suppress a track after picking."""
        try:
            data = json.loads(msg.data)
            pid = int(data["persistent_id"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self.get_logger().error(f"mark_picked parse error: {exc}")
            return
        self._picked_ids.add(pid)
        # Also immediately remove from active registry so it stops publishing.
        self._tracks.pop(pid, None)
        self.get_logger().info(f"Tomato persistent_id={pid} marked as picked — suppressed.")

    def _spatial_callback(self, msg: String) -> None:
        """Process one frame of sphere-fit detections and update the registry."""
        try:
            tomatoes: list[dict] = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"JSON parse error: {exc}")
            return

        self._frame += 1

        # ── Step 0: prediction — compensate accumulated camera motion ──────
        # Drain the motion accumulator atomically and shift every track by −Δ
        # so its centroid sits where we expect to observe it in this frame.
        # 1 mm threshold suppresses noise-level zero-motion publishes.
        with self._motion_lock:
            dx, dy, dz = self._pending_motion
            self._pending_motion = [0.0, 0.0, 0.0]
        motion_norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if motion_norm > 1e-3 and self._tracks:
            for track in self._tracks.values():
                track.predict_motion(dx, dy, dz)
            self.get_logger().info(
                f"Applied camera motion compensation: "
                f"Δ=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m (|Δ|={motion_norm*100:.1f} cm) "
                f"to {len(self._tracks)} track(s) before matching."
            )

        # ── Step 1: Hungarian bipartite matching ───────────────────────────
        # Only consider non-picked active tracks as candidates.
        active_tracks = [
            (pid, t) for pid, t in self._tracks.items()
            if pid not in self._picked_ids
        ]
        n_tracks = len(active_tracks)
        n_dets = len(tomatoes)

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        if n_tracks > 0 and n_dets > 0:
            # Build cost matrix C[i][j] = 3D distance(track_i, detection_j).
            # Pairs beyond match_threshold_m are set to INF — the Hungarian
            # solver will only assign them if no feasible alternative exists,
            # and we reject any such pair in the post-filter below.
            _INF = 1e9
            C = np.full((n_tracks, n_dets), fill_value=_INF)
            for i, (pid, track) in enumerate(active_tracks):
                for j, tomato in enumerate(tomatoes):
                    d = _euclidean(track.centroid, tomato["centroid"])
                    if d < self._threshold:
                        C[i, j] = d

            # scipy.optimize.linear_sum_assignment: O(n³), globally optimal.
            row_ind, col_ind = linear_sum_assignment(C)

            for i, j in zip(row_ind, col_ind):
                if C[i, j] >= self._threshold:
                    # INF pair — both track and detection remain unmatched.
                    continue
                pid, track = active_tracks[i]
                track.update(tomatoes[j])
                matched_track_ids.add(pid)
                matched_det_indices.add(j)

        # Unmatched detections → new tracks.
        for j, tomato in enumerate(tomatoes):
            if j not in matched_det_indices:
                new_pid = self._next_id
                self._next_id += 1
                self._tracks[new_pid] = Track(new_pid, tomato, self._alpha)
                matched_track_ids.add(new_pid)

        # ── Step 2: age unmatched tracks, drop stale ones ──────────────────
        to_drop = []
        for pid, track in self._tracks.items():
            if pid not in matched_track_ids:
                track.missed_frames += 1
                if track.missed_frames > self._max_missed:
                    to_drop.append(pid)
                    self.get_logger().info(
                        f"Track persistent_id={pid} dropped "
                        f"(missed {track.missed_frames} frames)."
                    )

        for pid in to_drop:
            self._tracks.pop(pid)
            self._picked_ids.discard(pid)

        # ── Step 3: publish active, non-picked tracks ──────────────────────
        active = [
            t.to_dict()
            for t in self._tracks.values()
            if t.persistent_id not in self._picked_ids
        ]

        out = String()
        out.data = json.dumps(active)
        self._tracks_pub.publish(out)

        ids = [t["persistent_id"] for t in active]
        self.get_logger().info(
            f"Frame {self._frame}: {len(active)} active tracks {ids} "
            f"(registry size={len(self._tracks)})."
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TomatoTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
