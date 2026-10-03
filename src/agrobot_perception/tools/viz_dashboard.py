"""
viz_dashboard.py — Real-time Perception Visualization Dashboard (TOOL)

Four-panel PyQt5 GUI consuming live ROS 2 topics from the Agrobot TOM v2
picking pipeline:

  Panel 1 (top-left)     Camera feed with track-state overlays + safe-to-pick bar
  Panel 2 (top-right)    Scrolling detection event stream (newest at top)
  Panel 3 (bottom-left)  Per-tomato catalog grid — session-persistent cards
  Panel 4 (bottom-right) System health metrics and node liveness

Architecture
------------
AgroVizNode spins in a daemon thread (rclpy.spin).  All 11 ROS callbacks write
into threading.Lock-protected state dicts/deques.  Four QTimers on the Qt main
thread read from that state — no Qt calls are ever made from inside a callback.

Target environment: [NUCBOX] — run inside the ROCm Docker container with a
live pipeline (detector → spatial → tracker → qwen_vl) already publishing.
Requires an X11 display:
  SSH with X11:  ssh -X <user>@<nucbox_ip>   # then run the command below
  Headless:      Xvfb :99 -screen 0 1280x800x24 &
                 DISPLAY=:99 PYTHONPATH=perception python3 ...

Sprint: 4 — HIL visualization for VLM-guided pick selection.

Usage
-----
    PYTHONPATH=perception python3 perception/tools/viz_dashboard.py

Subscribed topics (11)
-----------------------
  /agrobot/debug_image              sensor_msgs/Image        (SENSOR_QOS)
  /camera/camera/color/image_raw    sensor_msgs/Image        (SENSOR_QOS, fallback)
  /camera/camera/color/camera_info  sensor_msgs/CameraInfo   (SENSOR_QOS, for projection)
  /agrobot/detections               vision_msgs/Detection2DArray
  /agrobot/tomato_spatial           std_msgs/String (JSON)
  /agrobot/tomato_tracks            std_msgs/String (JSON)
  /agrobot/pick_target              geometry_msgs/PoseStamped
  /agrobot/vlm_reasoning            std_msgs/String
  /agrobot/vlm_selection            std_msgs/String (JSON)
  /agrobot/safe_to_pick             std_msgs/Bool
  /agrobot/mark_picked              std_msgs/String (JSON {"persistent_id": N})
"""

# requires: PyQt5, cv_bridge, rclpy, sensor_msgs, vision_msgs, geometry_msgs, std_msgs

from __future__ import annotations

import base64
import collections
import json
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from vision_msgs.msg import Detection2DArray

try:
    from cv_bridge import CvBridge
except ImportError as exc:
    raise ImportError(
        "cv_bridge not found — run inside the ROS 2 Docker container:\n"
        "  docker compose -f deployment/compose/docker-compose.yml run --rm dev bash"
    ) from exc

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ─── Named constants ───────────────────────────────────────────────────────────

MODEL_INPUT_SIZE = 518      # DINOv2: 37 × 14 px — baked into preprocess_for_dino()
MAX_LOG_ENTRIES  = 200
CATALOG_COLS     = 4
CARD_IMAGE_W, CARD_IMAGE_H = 72, 72
CARD_W,       CARD_H       = 112, 158
HEALTH_TIMEOUT_S = 10.0
METRICS_WINDOW   = 30       # rolling window for per-frame detection rate
# Demo-friendly catalog turnover: drop LOST cards 20 s after they go missing.
# Picked cards are never auto-evicted (session history).
LOST_CARD_TTL_S  = 20.0

# Camera topics from RealSense publish with BEST_EFFORT reliability.
# A RELIABLE subscription here causes a QoS mismatch — the node would
# subscribe successfully but receive zero messages.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)

# BGR order (OpenCV convention) — drawn on frames before resize-to-display
_BGR_CYAN   = (255, 200,   0)   # new / untracked
_BGR_GREEN  = (  0, 230,   0)   # active + smoothed (age ≥ 3)
_BGR_YELLOW = (  0, 220, 220)   # VLM-selected pick target
_BGR_RED    = (  0,   0, 210)   # already picked

# Qt hex strings for text, badges, and borders
_QT_GREEN = "#00e676"
_QT_AMBER = "#ffab40"
_QT_RED   = "#f44336"
_QT_CYAN  = "#00e5ff"
_QT_BLUE  = "#2979ff"
_QT_GRAY  = "#757575"

_APP_TITLE = "Agrobot TOM v2 — Perception Dashboard"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _unletterbox_pt(
    x: float, y: float, orig_w: int, orig_h: int
) -> Tuple[float, float]:
    """Reverse the letterbox applied by resize_with_aspect() in image_utils.py.

    resize_with_aspect scales to fit 518×518 with centred black padding;
    this undoes that transform to recover native image coordinates.
    """
    scale = min(MODEL_INPUT_SIZE / orig_w, MODEL_INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (MODEL_INPUT_SIZE - new_w) // 2
    pad_y = (MODEL_INPUT_SIZE - new_h) // 2
    return (x - pad_x) / scale, (y - pad_y) / scale


def _project_to_native(
    centroid: dict,
    cam: Optional[dict],
) -> Optional[Tuple[float, float]]:
    """Project a 3D camera-frame centroid to native image pixel (u, v).

    Standard pinhole model: u = fx·(X/Z) + cx.
    Returns None when intrinsics are not yet cached or the centroid is behind
    the camera (Z ≤ 0).
    """
    if cam is None or centroid.get("z", 0.0) <= 0.0:
        return None
    X, Y, Z = centroid["x"], centroid["y"], centroid["z"]
    return cam["fx"] * (X / Z) + cam["cx"], cam["fy"] * (Y / Z) + cam["cy"]


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to a QImage backed by its own data copy.

    .copy() is required — without it, QImage holds a raw pointer into the
    numpy buffer which may be deallocated before Qt renders the frame.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


def _decode_b64_jpeg(b64: Optional[str]) -> Optional[np.ndarray]:
    """Decode a base64 JPEG to a BGR ndarray; returns None on any failure."""
    if not b64:
        return None
    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _extract_per_id_reasoning(text: str) -> Dict[int, str]:
    """Split a VLM response into per-persistent_id snippets.

    The prompt asks the model to identify imposters by ID ("Tomato 0 is an
    apple, Tomato 1 is real..."). We split the response into sentences and
    associate each sentence with whichever Tomato numbers it mentions. A
    sentence that mentions multiple IDs (e.g. "Tomato 0 and Tomato 1 are
    imposters") attaches to all of them. Returns {pid: trimmed_snippet}.

    Best-effort only: when the VLM departs from the requested format (e.g.
    only one sentence with no IDs, or a different sentence boundary), the
    affected IDs simply get an empty/default entry the caller can replace.
    """
    import re

    # Sentence-ish split. Newlines and ". " count as boundaries; we trim
    # leading/trailing whitespace and skip empty fragments.
    chunks = [
        c.strip()
        for c in re.split(r"(?<=[.!?])\s+|\n+", text)
        if c.strip()
    ]

    # Match all known ID formats: "Tomato 1", "Candidate 0", "Candidate #2"
    id_pattern = re.compile(r"(?i)\b(?:tomato|candidate)\s*#?\s*(\d+)\b")

    result: Dict[int, list] = {}
    for chunk in chunks:
        for match in id_pattern.finditer(chunk):
            pid = int(match.group(1))
            result.setdefault(pid, []).append(chunk)

    # Collapse list of sentences per pid into a single short string.
    return {pid: " ".join(snippets) for pid, snippets in result.items()}


def _parse_vlm_verdict(text: str) -> Tuple[str, str, str]:
    """Extract a verdict label + colors from a VLM reasoning string.

    The detector-side prompt requests a `VERDICT: READY|NOT_READY|UNCERTAIN`
    line, but we also degrade gracefully on legacy "YES./NO." style responses
    and on free-form text. Returns (label, fg_color, bg_color) ready for
    direct QLabel styling.
    """
    upper = text.upper()
    if "READY" in upper and "NOT_READY" not in upper and "NOT READY" not in upper:
        return "✓  READY TO PICK", _QT_GREEN, "#0d2218"
    if "NOT_READY" in upper or "NOT READY" in upper:
        return "✗  NOT READY", _QT_RED, "#2a0d0d"
    if "UNCERTAIN" in upper:
        return "?  UNCERTAIN", _QT_AMBER, "#261a0d"
    # Legacy compatibility: "YES." / "NO." at the start of the response.
    head = upper.strip().split(".", 1)[0].strip()
    if head == "YES":
        return "✓  READY TO PICK", _QT_GREEN, "#0d2218"
    if head == "NO":
        return "✗  NOT READY", _QT_RED, "#2a0d0d"
    return "↻  ANALYZING", _QT_AMBER, "#261a0d"


def _placeholder_img() -> np.ndarray:
    """120×90 dark placeholder shown when a track has no clipped_image JPEG."""
    img = np.full((CARD_IMAGE_H, CARD_IMAGE_W, 3), 52, dtype=np.uint8)
    cv2.putText(
        img, "?", (46, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (130, 130, 130), 2, cv2.LINE_AA,
    )
    return img


def _dark_palette() -> QPalette:
    p = QPalette()
    dark = QColor(28, 28, 28)
    mid  = QColor(44, 44, 44)
    text = QColor(218, 218, 218)
    p.setColor(QPalette.Window,          dark)
    p.setColor(QPalette.WindowText,      text)
    p.setColor(QPalette.Base,            mid)
    p.setColor(QPalette.AlternateBase,   QColor(56, 56, 56))
    p.setColor(QPalette.ToolTipBase,     dark)
    p.setColor(QPalette.ToolTipText,     text)
    p.setColor(QPalette.Text,            text)
    p.setColor(QPalette.Button,          mid)
    p.setColor(QPalette.ButtonText,      text)
    p.setColor(QPalette.BrightText,      QColor(255, 80, 80))
    p.setColor(QPalette.Highlight,       QColor(42, 130, 218))
    p.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    return p


# ─── ROS 2 node ────────────────────────────────────────────────────────────────

class AgroVizNode(Node):
    """Aggregates all pipeline topics into thread-safe state.

    Runs exclusively in a daemon thread (rclpy.spin).  QTimer slots on the
    Qt main thread read from this state under _lock — never the reverse.
    """

    def __init__(self) -> None:
        super().__init__("agrobot_viz_dashboard")
        self._bridge = CvBridge()
        self._lock   = threading.Lock()

        # ── Camera frames ────────────────────────────────────────────────────
        self._debug_frame: Optional[np.ndarray] = None
        self._raw_frame:   Optional[np.ndarray] = None
        self._debug_ts:    float = 0.0
        self._raw_ts:      float = 0.0

        # Cached once — RealSense intrinsics are static per session
        self._cam_info: Optional[dict] = None

        # ── Detection / track state ───────────────────────────────────────────
        self._detections: List[dict] = []           # [{bbox, score, cx_518, cy_518}]
        self._tracks:     List[dict] = []           # latest tomato_tracks JSON array
        self._catalog:    Dict[int, dict] = {}      # persistent_id → enriched track dict
        self._picked_ids: set  = set()
        self._vlm_id:     Optional[int] = None      # persistent_id selected by VLM
        # Per-track classification votes. Each cycle, the VLM parses the
        # response and tells us which IDs it considers imposter vs real.
        # We accumulate votes here and decide the catalog badge based on
        # majority. This:
        #   - smooths over single-call VLM mistakes (one flip won't change
        #     the verdict if 4 prior calls agreed),
        #   - decouples per-card verdict from the global -1 veto (a card
        #     the VLM consistently calls real shows as ACTIVE even when the
        #     global veto fires),
        #   - and lets the demo viewer watch the verdict stabilise as
        #     evidence accumulates.
        # Schema: {pid: {"imposter": int, "real": int}}
        self._vlm_votes: Dict[int, dict] = {}
        # Decision threshold: a tag locks once it has at least this many
        # MORE votes than the other side. Lower = more responsive, higher
        # = more stable. 2 means "any 2-vote lead locks it" — empirically
        # the right balance for ~10 s VLM cycle on the demo wall.
        self._vlm_vote_margin: int = 2
        self._vlm_reason: str = ""

        # ── Misc UI state ─────────────────────────────────────────────────────
        self._safe_to_pick: bool = False
        # Rolling history of the last few VLM responses for the dashboard
        # bottom-right panel — newest first.
        self._vlm_history: collections.deque = collections.deque(maxlen=4)
        self._vlm_history_gen: int = 0
        # Per-track VLM reasoning snippets. Populated by parsing "Tomato N"
        # mentions out of the response text — used by the clickable catalog
        # cards to show what the VLM specifically said about each track.
        self._per_id_reasoning: Dict[int, str] = {}

        # Panel 2: HTML log entries, newest first; generation counter avoids
        # a full string comparison on every tick
        self._log_entries: collections.deque = collections.deque(maxlen=MAX_LOG_ENTRIES)
        self._log_gen:     int = 0

        # Panel 4 metrics
        self._det_window: collections.deque = collections.deque(maxlen=METRICS_WINDOW)
        self._track_ts:   collections.deque = collections.deque(maxlen=20)
        self._lost_count: int = 0
        self._seen_ids:   set = set()

        # Node health: epoch-time of last message per logical node
        self._health: Dict[str, float] = {
            "detector": 0.0,
            "spatial":  0.0,
            "tracker":  0.0,
            "qwen_vl":  0.0,
        }

        # ── Subscriptions ─────────────────────────────────────────────────────
        _r = rclpy.qos.QoSProfile(depth=10)    # reliable QoS for pipeline topics

        self.create_subscription(Image,            "/agrobot/debug_image",               self._cb_debug_img,   SENSOR_QOS)
        self.create_subscription(Image,            "/camera/camera/color/image_raw",     self._cb_raw_img,     SENSOR_QOS)
        self.create_subscription(CameraInfo,       "/camera/camera/color/camera_info",   self._cb_cam_info,    SENSOR_QOS)
        self.create_subscription(Detection2DArray, "/agrobot/detections",                self._cb_detections,  _r)
        self.create_subscription(String,           "/agrobot/tomato_spatial",            self._cb_spatial,     _r)
        self.create_subscription(String,           "/agrobot/tomato_tracks",             self._cb_tracks,      _r)
        self.create_subscription(PoseStamped,      "/agrobot/pick_target",               self._cb_pick_target, _r)
        self.create_subscription(String,           "/agrobot/vlm_reasoning",             self._cb_vlm_reason,  _r)
        self.create_subscription(String,           "/agrobot/vlm_selection",             self._cb_vlm_select,  _r)
        self.create_subscription(Bool,             "/agrobot/safe_to_pick",              self._cb_safe,        _r)
        self.create_subscription(String,           "/agrobot/mark_picked",               self._cb_mark_picked, _r)

        self._reset_pub = self.create_publisher(String, "/agrobot/reset_tracker", 10)

        self.get_logger().info(f"{_APP_TITLE} — node ready, 11 subscriptions active.")

    # ── Image callbacks ───────────────────────────────────────────────────────

    def _cb_debug_img(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._debug_frame = frame
            self._debug_ts    = time.monotonic()

    def _cb_raw_img(self, msg: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._raw_frame = frame
            self._raw_ts    = time.monotonic()
            # Sample once per raw frame so the rolling window tracks actual
            # camera frames, not our 33ms poll interval
            self._det_window.append(len(self._detections) > 0)

    def _cb_cam_info(self, msg: CameraInfo) -> None:
        with self._lock:
            if self._cam_info is None:
                self._cam_info = {
                    "fx": msg.k[0], "fy": msg.k[4],
                    "cx": msg.k[2], "cy": msg.k[5],
                    "w":  msg.width, "h": msg.height,
                }

    # ── Detection callback ────────────────────────────────────────────────────

    def _cb_detections(self, msg: Detection2DArray) -> None:
        dets: List[dict] = []
        for d in msg.detections:
            cx_518 = d.bbox.center.position.x
            cy_518 = d.bbox.center.position.y
            hw     = d.bbox.size_x / 2.0
            hh     = d.bbox.size_y / 2.0
            score  = d.results[0].hypothesis.score if d.results else 0.0
            dets.append({
                "bbox":   [cx_518 - hw, cy_518 - hh, cx_518 + hw, cy_518 + hh],
                "score":  score,
                "cx_518": cx_518,
                "cy_518": cy_518,
            })
        with self._lock:
            self._detections = dets
            self._health["detector"] = time.monotonic()

    # ── Spatial callback (health stamp only; tracks carry the full 3D data) ───

    def _cb_spatial(self, msg: String) -> None:
        with self._lock:
            self._health["spatial"] = time.monotonic()

    # ── Tracks callback ───────────────────────────────────────────────────────

    def _cb_tracks(self, msg: String) -> None:
        try:
            tracks: List[dict] = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        now = time.monotonic()
        ts  = datetime.now()

        with self._lock:
            self._health["tracker"] = now
            self._track_ts.append(now)
            self._tracks = tracks

            active_ids = {t["persistent_id"] for t in tracks}
            self._seen_ids.update(active_ids)

            # Merge into catalog — entries are never removed, only status-flagged
            for t in tracks:
                pid = t["persistent_id"]
                self._catalog[pid] = {
                    **t,
                    "_lost":   False,
                    "_picked": pid in self._picked_ids,
                }

            # Tracks absent this frame are marked lost; timestamp the
            # transition so panel 3 can auto-evict stale LOST cards.
            for pid, entry in self._catalog.items():
                if (pid not in active_ids
                        and not entry.get("_picked")
                        and not entry.get("_lost")):
                    self._catalog[pid] = {
                        **entry,
                        "_lost": True,
                        "_lost_at": now,
                    }

            self._lost_count = sum(
                1 for e in self._catalog.values()
                if e.get("_lost") and not e.get("_picked")
            )

            # Build Panel 2 HTML card for this batch.
            # We call them "candidates" (not "tomatoes") because the upstream
            # detector matches on round-fruit-like visual features and only
            # Qwen-VL semantically confirms which are actually tomatoes.
            k      = len(tracks)
            id_str = [t["persistent_id"] for t in tracks]
            t_str  = ts.strftime("%H:%M:%S.") + f"{ts.microsecond // 1000:03d}"
            noun   = "candidate" if k == 1 else "candidates"
            lines  = [
                f'<span style="color:{_QT_CYAN};font-weight:bold;">'
                f'[{t_str}]  🍅  |  {k} {noun}  |  IDs: {id_str}</span>'
            ]
            for t in tracks:
                pid  = t["persistent_id"]
                z    = t["centroid"]["z"]
                r    = t["sphere"]["radius"] * 100.0
                conf = t["confidence"]
                age  = t.get("age", 0)
                sm   = t.get("smoothed", False)
                if pid in self._picked_ids:
                    color, badge, deco = _QT_RED,   "✗ picked",      "text-decoration:line-through;"
                elif sm and age >= 3:
                    color, badge, deco = _QT_GREEN, "✓ smoothed",    "font-weight:bold;"
                else:
                    color, badge, deco = _QT_AMBER, "⏳ converging", ""
                lines.append(
                    f'<span style="color:{color};{deco}">'
                    f"  Track #{pid}  z={z:.2f}m  r={r:.1f}cm  "
                    f"conf={conf:.2f}  age={age}  {badge}</span>"
                )
            self._log_entries.appendleft("<br>".join(lines))
            self._log_gen += 1

    # ── VLM callbacks ─────────────────────────────────────────────────────────

    def _cb_pick_target(self, msg: PoseStamped) -> None:
        # pick_target fires only on an actual selection, not every frame;
        # it's a coarse liveness signal, not a per-frame heartbeat
        with self._lock:
            self._health["qwen_vl"] = time.monotonic()

    def _cb_vlm_reason(self, msg: String) -> None:
        ts_str = datetime.now().strftime("%H:%M:%S")
        html = (
            f'<span style="color:{_QT_GRAY};font-style:italic;">'
            f"[{ts_str}] VLM: {msg.data[:300]}</span>"
        )
        per_id = _extract_per_id_reasoning(msg.data)
        with self._lock:
            self._vlm_reason = msg.data
            self._health["qwen_vl"] = time.monotonic()
            self._log_entries.appendleft(html)
            self._log_gen += 1
            self._vlm_history.appendleft(
                {"ts": ts_str, "text": msg.data, "pid": self._vlm_id}
            )
            self._vlm_history_gen += 1
            # Store per-id reasoning snippets from multi-candidate responses.
            for pid_key, snippet in per_id.items():
                self._per_id_reasoning[pid_key] = snippet
            # Single-tomato path: response contains "VERDICT: NOT_TOMATO /
            # READY / NOT_READY" with no "Candidate N" — store the whole
            # text keyed to the currently-tracked pid so the popup can show
            # it even though no candidate number appears in the text.
            if not per_id and self._vlm_id is not None:
                self._per_id_reasoning[self._vlm_id] = msg.data.strip()

    def _cb_vlm_select(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            pid = int(payload["persistent_id"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return

        def _as_int_set(value) -> set:
            try:
                return {int(x) for x in (value or [])}
            except (TypeError, ValueError):
                return set()

        new_imposters = _as_int_set(payload.get("imposters"))
        new_reals     = _as_int_set(payload.get("real_tomatoes"))

        with self._lock:
            # Accumulate votes. A pid mentioned as imposter and real in the
            # same call (rare, but possible if the parser is ambiguous)
            # gets a vote on each side; majority decides.
            for ipid in new_imposters:
                self._vlm_votes.setdefault(ipid, {"imposter": 0, "real": 0})["imposter"] += 1
            for rpid in new_reals:
                self._vlm_votes.setdefault(rpid, {"imposter": 0, "real": 0})["real"] += 1
            # Selected pid also counts as a "real" vote — VLM picked it.
            if pid != -1:
                self._vlm_votes.setdefault(pid, {"imposter": 0, "real": 0})["real"] += 1
                # Only set the PICKING highlight when the current vote
                # tally agrees the chosen ID is real (avoids the flip-flop
                # the user saw before, where one bad pick would override
                # multiple imposter calls).
                if self._verdict_for(pid) != "imposter":
                    self._vlm_id = pid
            self._health["qwen_vl"] = time.monotonic()

    def _verdict_for(self, pid: int) -> str:
        """Compute the current per-track verdict from accumulated votes.

        Returns "imposter", "real", or "undecided".

        Real votes only arrive when the VLM actually SELECTS a candidate
        (confident commitment). Imposter votes arrive whenever the VLM
        mentions a candidate as a non-tomato — even during veto responses.

        Rules:
        - "imposter" locks immediately on the FIRST imposter vote (any
          evidence of imposter identity is enough to suppress picking).
        - "real" requires at least one real vote with ZERO imposter votes
          (the VLM must have confidently selected this ID at least once
          AND never called it an imposter).
        - "undecided" otherwise — show ACTIVE/CONVERGING state as normal.

        This is asymmetric by design: we'd rather suppress a valid tomato
        once than pick an apple once. Reset Catalog clears all votes.
        """
        votes = self._vlm_votes.get(pid)
        if not votes:
            return "undecided"
        if votes["imposter"] >= 1:
            return "imposter"
        if votes["real"] >= 1:
            return "real"
        return "undecided"

    def _cb_safe(self, msg: Bool) -> None:
        with self._lock:
            self._safe_to_pick = msg.data

    def _cb_mark_picked(self, msg: String) -> None:
        try:
            pid = int(json.loads(msg.data)["persistent_id"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return
        with self._lock:
            self._picked_ids.add(pid)
            if pid in self._catalog:
                self._catalog[pid]["_picked"] = True
                self._catalog[pid]["_lost"]   = False


# ─── Tomato card widget ────────────────────────────────────────────────────────

class TomatoCard(QFrame):
    """One card in the Panel 3 catalog grid.

    Widget instance is stable for the session; only content and border are
    updated on each tick to avoid layout thrashing.

    Emits `clicked(int)` carrying the persistent_id whenever the user
    left-clicks the card. MainWindow handles the click to open a per-track
    VLM reasoning popup.
    """

    clicked = pyqtSignal(int)

    _STATUS_ORDER: Dict[str, int] = {
        "PICKING": 0,
        "ACTIVE": 1,
        "CONVERGING": 2,
        "NOT_TOMATO": 3,
        "LOST": 4,
        "PICKED": 5,
    }

    def __init__(self, pid: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.pid    = pid
        self.status = "CONVERGING"
        self.setFixedSize(CARD_W, CARD_H)
        self.setFrameShape(QFrame.Box)
        # Visual affordance: pointer cursor so the user knows the card is
        # interactive even before they discover the popup.
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._img_lbl = QLabel()
        self._img_lbl.setFixedSize(CARD_IMAGE_W, CARD_IMAGE_H)
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setStyleSheet("background:#2a2a2a;border-radius:4px;")
        layout.addWidget(self._img_lbl, alignment=Qt.AlignHCenter)

        self._id_lbl = QLabel(f"#{pid}")
        self._id_lbl.setFont(QFont("Monospace", 9, QFont.Bold))
        self._id_lbl.setAlignment(Qt.AlignCenter)
        self._id_lbl.setStyleSheet("border:none;")
        layout.addWidget(self._id_lbl)

        self._info_lbl = QLabel()
        self._info_lbl.setFont(QFont("Monospace", 7))
        self._info_lbl.setAlignment(Qt.AlignCenter)
        self._info_lbl.setWordWrap(True)
        self._info_lbl.setStyleSheet("border:none;")
        layout.addWidget(self._info_lbl)

        self._badge = QLabel()
        self._badge.setFont(QFont("Monospace", 7, QFont.Bold))
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setFixedHeight(15)
        layout.addWidget(self._badge)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.pid)
        super().mousePressEvent(event)

    def refresh(
        self,
        entry: dict,
        picked: bool,
        vlm_id: Optional[int],
        is_imposter: bool = False,
    ) -> None:
        pid  = entry["persistent_id"]
        z    = entry["centroid"]["z"]
        r    = entry["sphere"]["radius"] * 100.0
        conf = entry["confidence"]
        age  = entry.get("age", 0)
        lost = entry.get("_lost", False)

        raw = _decode_b64_jpeg(entry.get("clipped_image"))
        src = raw if raw is not None else _placeholder_img()
        # Fit the source crop into the card thumbnail without squashing.
        # Scale-to-fill + centre-crop keeps the tomato round; if the source
        # is portrait-ish (typical for hanging fruit) the bottom/sides get
        # gently cropped rather than horizontally distorted.
        sh, sw = src.shape[:2]
        scale = max(CARD_IMAGE_W / sw, CARD_IMAGE_H / sh)
        new_w, new_h = max(1, int(sw * scale)), max(1, int(sh * scale))
        scaled = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cx = max(0, (new_w - CARD_IMAGE_W) // 2)
        cy = max(0, (new_h - CARD_IMAGE_H) // 2)
        img = scaled[cy:cy + CARD_IMAGE_H, cx:cx + CARD_IMAGE_W]
        self._img_lbl.setPixmap(QPixmap.fromImage(_bgr_to_qimage(img)))
        self._info_lbl.setText(
            f"z:{z:.2f}m  r:{r:.1f}cm\nconf:{conf:.2f}  age:{age}"
        )

        # PICKED takes priority over everything (terminal state). NOT_TOMATO
        # outranks PICKING/ACTIVE/CONVERGING so a vetoed track is visibly
        # disqualified even if it's still being detected each cycle.
        if picked:
            self.status = "PICKED"
            txt, fg, bg, border = "✓ PICKED",    _QT_RED,   "#2a0d0d", _QT_RED
        elif is_imposter:
            self.status = "NOT_TOMATO"
            txt, fg, bg, border = "✗ NOT TOMATO", _QT_RED,  "#2a0d0d", _QT_RED
        elif pid == vlm_id:
            self.status = "PICKING"
            txt, fg, bg, border = "→ PICKING",   _QT_BLUE,  "#0d1733", _QT_BLUE
        elif lost:
            self.status = "LOST"
            txt, fg, bg, border = "LOST",         _QT_GRAY,  "#222",   "#444"
        elif entry.get("smoothed") and age >= 3:
            self.status = "ACTIVE"
            txt, fg, bg, border = "ACTIVE",       _QT_GREEN, "#0d2218", _QT_GREEN
        else:
            self.status = "CONVERGING"
            txt, fg, bg, border = "CONVERGING",   _QT_AMBER, "#261a0d", _QT_AMBER

        self._badge.setText(txt)
        self._badge.setStyleSheet(
            f"color:{fg};background:{bg};border-radius:3px;border:none;"
        )
        self.setStyleSheet(
            f"QFrame{{border:1px solid {border};border-radius:4px;}}"
        )

    def sort_key(self) -> int:
        return self._STATUS_ORDER.get(self.status, 9)


# ─── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self, node: AgroVizNode) -> None:
        super().__init__()
        self._node     = node
        self._cards:   Dict[int, TomatoCard] = {}
        self._sort_order: List[int] = []     # last rendered sort order for Panel 3
        self._last_log_gen: int = -1         # last log generation rendered in Panel 2
        self._last_vlm_history_gen: int = -1  # last VLM history gen rendered in Panel 4
        # Click-popup state for the catalog cards. One popup is reused; the
        # currently-shown pid sits in _popup_pid so the next click on the
        # SAME card hides it (toggle), and a click on a DIFFERENT card
        # re-targets the popup without closing/reopening.
        self._popup: Optional[QFrame] = None
        self._popup_pid: Optional[int] = None

        self.setWindowTitle(_APP_TITLE)
        self.setMinimumSize(1280, 800)
        self._setup_theme()
        self._build_menu()
        self._build_ui()
        self._start_timers()

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _setup_theme(self) -> None:
        QApplication.setStyle("Fusion")
        QApplication.setPalette(_dark_palette())
        self.setStyleSheet("""
            QMainWindow,QWidget { background:#1c1c1c; color:#dcdcdc; }
            QMenuBar            { background:#2d2d2d; }
            QMenuBar::item:selected { background:#3c3c3c; }
            QMenu               { background:#2d2d2d; }
            QMenu::item:selected{ background:#3c3c3c; }
            QScrollBar:vertical { background:#252525; width:8px; }
            QScrollBar::handle:vertical {
                background:#555; border-radius:4px; min-height:20px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height:0; }
        """)

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self.menuBar()

        view = mb.addMenu("View")
        a_reset = QAction("Reset catalog", self)
        a_reset.triggered.connect(self._reset_catalog)
        view.addAction(a_reset)
        a_log = QAction("Clear log", self)
        a_log.triggered.connect(self._clear_log)
        view.addAction(a_log)

        help_m = mb.addMenu("Help")
        a_topics = QAction("Topic list", self)
        a_topics.triggered.connect(self._show_topics)
        help_m.addAction(a_topics)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Two side-by-side vertical stacks instead of a 2×2 grid. This lets
        # the LEFT column have its own vertical split (camera bigger, catalog
        # smaller) while the RIGHT column keeps its own independent ratio
        # (event stream / metrics) — QGridLayout's row stretch is shared
        # across columns, so it can't model asymmetric splits.
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setSpacing(6)
        outer.setContentsMargins(6, 6, 6, 6)

        left_col = QVBoxLayout()
        left_col.setSpacing(6)
        # 9:4 → camera ~69% of left-column height, catalog ~31%. Gives the
        # catalog enough room for one row of cards without dominating.
        left_col.addWidget(self._panel1(), stretch=9)
        left_col.addWidget(self._panel3(), stretch=4)

        right_col = QVBoxLayout()
        right_col.setSpacing(6)
        # Event stream and metrics keep an even split (unchanged from before).
        right_col.addWidget(self._panel2(), stretch=3)
        right_col.addWidget(self._panel4(), stretch=2)

        # Left column wider than right so the 4:3 camera image lands closer
        # to its native aspect ratio after letterboxing.
        outer.addLayout(left_col, stretch=3)
        outer.addLayout(right_col, stretch=2)

    def _titled_frame(self, title: str) -> Tuple[QFrame, QVBoxLayout]:
        f = QFrame()
        f.setFrameShape(QFrame.Box)
        f.setStyleSheet("QFrame{border:1px solid #3a3a3a;border-radius:5px;}")
        v = QVBoxLayout(f)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        lbl = QLabel(title)
        lbl.setFont(QFont("Monospace", 9, QFont.Bold))
        lbl.setStyleSheet(f"color:{_QT_CYAN};border:none;")
        v.addWidget(lbl)
        return f, v

    def _panel1(self) -> QFrame:
        f, v = self._titled_frame("▶  CAMERA FEED")
        self._cam_lbl = QLabel()
        self._cam_lbl.setAlignment(Qt.AlignCenter)
        self._cam_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_lbl.setStyleSheet("background:#111;border:none;")
        v.addWidget(self._cam_lbl)

        self._safe_bar = QLabel("SAFE TO PICK: ---")
        self._safe_bar.setAlignment(Qt.AlignCenter)
        self._safe_bar.setFont(QFont("Monospace", 10, QFont.Bold))
        self._safe_bar.setFixedHeight(26)
        self._safe_bar.setStyleSheet("border:none;background:#252525;border-radius:4px;")
        v.addWidget(self._safe_bar)
        return f

    def _panel2(self) -> QFrame:
        f, v = self._titled_frame("▶  DETECTION EVENT STREAM")
        self._event_log = QTextEdit()
        self._event_log.setReadOnly(True)
        self._event_log.setFont(QFont("Monospace", 9))
        self._event_log.setStyleSheet("background:#111;border:none;color:#ccc;")
        v.addWidget(self._event_log)
        return f

    def _panel3(self) -> QFrame:
        f, v = self._titled_frame("▶  TOMATO CATALOG")

        btn = QPushButton("⟳  Reset Catalog")
        btn.setFixedHeight(26)
        btn.setStyleSheet(
            f"QPushButton{{background:#2a1a0d;color:{_QT_AMBER};border:1px solid {_QT_AMBER};"
            f"border-radius:4px;font-family:Monospace;font-size:9pt;}}"
            f"QPushButton:hover{{background:#3a2a1d;}}"
            f"QPushButton:pressed{{background:#1a0d00;}}"
        )
        btn.clicked.connect(self._reset_catalog)
        v.addWidget(btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none;")
        self._grid_w = QWidget()
        self._grid_l = QGridLayout(self._grid_w)
        self._grid_l.setSpacing(6)
        self._grid_l.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(self._grid_w)
        v.addWidget(scroll)
        return f

    def _panel4(self) -> QFrame:
        f, v = self._titled_frame("▶  SYSTEM METRICS")

        def _metric_row(label: str) -> QLabel:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            l_lbl = QLabel(label)
            l_lbl.setFont(QFont("Monospace", 9))
            l_lbl.setStyleSheet("border:none;color:#888;")
            l_lbl.setFixedWidth(200)
            val = QLabel("—")
            val.setFont(QFont("Monospace", 9, QFont.Bold))
            val.setStyleSheet("border:none;")
            h.addWidget(l_lbl)
            h.addWidget(val)
            h.addStretch()
            v.addWidget(row)
            return val

        self._m_rate    = _metric_row("Detection rate:")
        self._m_latency = _metric_row("Mean frame latency:")
        self._m_active  = _metric_row("Active tracks:")
        self._m_lost    = _metric_row("Lost this session:")
        self._m_picked  = _metric_row("Picked this session:")

        v.addWidget(self._sep())

        sec = QLabel("Node health:")
        sec.setFont(QFont("Monospace", 9, QFont.Bold))
        sec.setStyleSheet("border:none;color:#888;")
        v.addWidget(sec)

        def _health_row(name: str, topic: str) -> QLabel:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            dot = QLabel("●")
            dot.setFont(QFont("Monospace", 11))
            dot.setFixedWidth(18)
            dot.setStyleSheet(f"color:{_QT_GRAY};border:none;")
            lbl = QLabel(f"{name:<12} {topic}")
            lbl.setFont(QFont("Monospace", 8))
            lbl.setStyleSheet("border:none;")
            h.addWidget(dot)
            h.addWidget(lbl)
            h.addStretch()
            v.addWidget(row)
            return dot

        self._h_det = _health_row("detector",  "/agrobot/detections")
        self._h_spa = _health_row("spatial",   "/agrobot/tomato_spatial")
        self._h_trk = _health_row("tracker",   "/agrobot/tomato_tracks")
        self._h_vlm = _health_row("qwen_vl",   "/agrobot/vlm_reasoning")

        v.addWidget(self._sep())

        # ── VLM Live Reasoning ────────────────────────────────────────────
        vlm_header = QLabel("⚡  VLM LIVE REASONING")
        vlm_header.setFont(QFont("Monospace", 9, QFont.Bold))
        vlm_header.setStyleSheet(f"color:{_QT_CYAN};border:none;")
        v.addWidget(vlm_header)

        # Verdict badge — large, color-coded READY / NOT_READY / UNCERTAIN
        self._vlm_verdict_lbl = QLabel("— awaiting first response —")
        self._vlm_verdict_lbl.setFont(QFont("Monospace", 12, QFont.Bold))
        self._vlm_verdict_lbl.setAlignment(Qt.AlignCenter)
        self._vlm_verdict_lbl.setFixedHeight(34)
        self._vlm_verdict_lbl.setStyleSheet(
            "background:#252525;border:1px solid #3a3a3a;border-radius:4px;color:#888;"
        )
        v.addWidget(self._vlm_verdict_lbl)

        # Reasoning text area — scrolls through last N responses
        self._vlm_reasoning_area = QTextEdit()
        self._vlm_reasoning_area.setReadOnly(True)
        self._vlm_reasoning_area.setFont(QFont("Monospace", 9))
        self._vlm_reasoning_area.setMinimumHeight(140)
        self._vlm_reasoning_area.setStyleSheet(
            "background:#111;border:1px solid #2a2a2a;color:#ccc;padding:4px;"
        )
        v.addWidget(self._vlm_reasoning_area, stretch=1)
        return f

    def _sep(self) -> QLabel:
        s = QLabel()
        s.setFixedHeight(1)
        s.setStyleSheet("background:#3a3a3a;border:none;")
        return s

    # ── Timers ────────────────────────────────────────────────────────────────

    def _start_timers(self) -> None:
        for interval, slot in (
            (33,  self._tick_panel1),
            (100, self._tick_panel2),
            (100, self._tick_panel3),
            (500, self._tick_panel4),
        ):
            t = QTimer(self)
            t.timeout.connect(slot)
            t.start(interval)

    # ── Panel 1 ───────────────────────────────────────────────────────────────

    def _tick_panel1(self) -> None:
        with self._node._lock:
            d_frame = self._node._debug_frame
            r_frame = self._node._raw_frame
            d_ts    = self._node._debug_ts
            dets    = list(self._node._detections)
            tracks  = list(self._node._tracks)
            picked  = set(self._node._picked_ids)
            vlm_id  = self._node._vlm_id
            safe    = self._node._safe_to_pick
            cam     = self._node._cam_info
            # Imposter IDs from vote history — these override yellow/green
            # even when vlm_id points at them (VLM selected inconsistently).
            cam_imposters = {
                p for p in self._node._vlm_votes
                if self._node._verdict_for(p) == "imposter"
            }

        now = time.monotonic()

        # Always prefer raw_frame — the debug_frame has boxes pre-drawn by the
        # detector in 518×518 coordinates on a 640×480 canvas, which produces a
        # second misaligned overlay when the dashboard draws its own boxes.
        if r_frame is not None:
            frame = r_frame.copy()
        elif d_frame is not None:
            frame = d_frame.copy()
        else:
            self._cam_lbl.setText("⏳ Waiting for camera…")
            return

        orig_h, orig_w = frame.shape[:2]

        # Confidence-keyed lookup for the no-intrinsics fallback
        by_conf = {round(t["confidence"], 4): t for t in tracks}

        def _match_det(det: dict) -> Optional[dict]:
            if cam is not None and tracks:
                # Project each track's 3D centroid into native image coords;
                # match the detection centre (unletterboxed) by pixel proximity
                dx, dy = _unletterbox_pt(det["cx_518"], det["cy_518"], orig_w, orig_h)
                best, d_min = None, float("inf")
                for t in tracks:
                    proj = _project_to_native(t["centroid"], cam)
                    if proj is None:
                        continue
                    dist = ((proj[0] - dx) ** 2 + (proj[1] - dy) ** 2) ** 0.5
                    if dist < d_min:
                        d_min, best = dist, t
                # 80 px threshold — generous to handle sync jitter between topics
                return best if d_min < 80.0 else None
            # No intrinsics yet: match by nearest confidence value
            if not by_conf:
                return None
            key = min(by_conf, key=lambda k: abs(k - round(det["score"], 4)))
            return by_conf[key]

        # Draw overlays at native resolution; resize once at the end.
        # Track which persistent_ids were covered by a fresh detection box.
        drawn_pids: set = set()

        for det in dets:
            x1_n, y1_n = _unletterbox_pt(det["bbox"][0], det["bbox"][1], orig_w, orig_h)
            x2_n, y2_n = _unletterbox_pt(det["bbox"][2], det["bbox"][3], orig_w, orig_h)
            x1, y1, x2, y2 = int(x1_n), int(y1_n), int(x2_n), int(y2_n)

            t = _match_det(det)
            if t:
                pid = t["persistent_id"]
                drawn_pids.add(pid)
                z   = t["centroid"]["z"]
                r   = t["sphere"]["radius"] * 100.0
                if pid in picked or pid in cam_imposters:
                    color = _BGR_RED
                elif pid == vlm_id:
                    color = _BGR_YELLOW
                elif t.get("smoothed") and t.get("age", 0) >= 3:
                    color = _BGR_GREEN
                else:
                    color = _BGR_CYAN
                label = f"#{pid} z={z:.2f}m r={r:.1f}cm"
            else:
                color = _BGR_CYAN
                label = f"conf={det['score']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, cv2.FILLED)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )

        # For active tracks not covered by a fresh detection, project the EMA-smoothed
        # 3D centroid back to 2D and draw an estimated box (thin border, ~ suffix).
        if cam is not None:
            for t in tracks:
                pid = t["persistent_id"]
                if pid in drawn_pids:
                    continue
                proj = _project_to_native(t["centroid"], cam)
                if proj is None:
                    continue
                u, v = int(proj[0]), int(proj[1])
                z = t["centroid"]["z"]
                r_px = max(10, int(cam["fx"] * t["sphere"]["radius"] / z))
                if pid in picked or pid in cam_imposters:
                    color = _BGR_RED
                elif pid == vlm_id:
                    color = _BGR_YELLOW
                elif t.get("smoothed") and t.get("age", 0) >= 3:
                    color = _BGR_GREEN
                else:
                    color = _BGR_CYAN
                bx1, by1, bx2, by2 = u - r_px, v - r_px, u + r_px, v + r_px
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
                # Thin border + ellipsis label communicates "tracker memory,
                # detector is between cycles" — better UX than showing a stale
                # numeric estimate the user might misread as a fresh measurement.
                elabel = f"#{pid} \u00b7 processing..."
                (tw, th), _ = cv2.getTextSize(elabel, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (bx1, by1 - th - 4), (bx1 + tw + 4, by1), color, cv2.FILLED)
                cv2.putText(
                    frame, elabel, (bx1 + 2, by1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
                )

        lw, lh = self._cam_lbl.width(), self._cam_lbl.height()
        if lw > 10 and lh > 10:
            # Scale-to-fill with centre crop: preserves native aspect (no
            # tomato squashing) AND fills the label edge-to-edge (no dark
            # bars). The tradeoff is a thin strip at the edges may be
            # cropped when the panel aspect differs from the camera aspect.
            # For a tomato-picking demo the subjects sit near the centre,
            # so the crop is invisible.
            fh, fw = frame.shape[:2]
            scale = max(lw / fw, lh / fh)
            new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
            display = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            crop_x = max(0, (new_w - lw) // 2)
            crop_y = max(0, (new_h - lh) // 2)
            display = display[crop_y:crop_y + lh, crop_x:crop_x + lw]
            self._cam_lbl.setPixmap(QPixmap.fromImage(_bgr_to_qimage(display)))

        if safe:
            self._safe_bar.setText("SAFE TO PICK: YES")
            self._safe_bar.setStyleSheet(
                f"border:none;background:#0d2218;color:{_QT_GREEN};border-radius:4px;"
            )
        else:
            self._safe_bar.setText("SAFE TO PICK: NO")
            self._safe_bar.setStyleSheet(
                f"border:none;background:#2a0d0d;color:{_QT_RED};border-radius:4px;"
            )

    # ── Panel 2 ───────────────────────────────────────────────────────────────

    def _tick_panel2(self) -> None:
        with self._node._lock:
            gen     = self._node._log_gen
            entries = list(self._node._log_entries)

        # Avoid rebuilding the entire HTML on every 100ms tick when nothing changed
        if gen == self._last_log_gen or not entries:
            return
        self._last_log_gen = gen

        sep  = "<hr style='border:none;border-top:1px solid #333;margin:3px 0;'>"
        body = sep.join(entries)
        self._event_log.setHtml(
            f'<div style="font-family:monospace;font-size:9pt;line-height:1.5;">'
            f"{body}</div>"
        )
        # Newest entry is at top; keep scroll there
        self._event_log.verticalScrollBar().setValue(0)

    # ── Panel 3 ───────────────────────────────────────────────────────────────

    def _tick_panel3(self) -> None:
        with self._node._lock:
            catalog = dict(self._node._catalog)
            picked  = set(self._node._picked_ids)
            vlm_id  = self._node._vlm_id
            # Vote-based per-ID verdict — see AgroVizNode._vlm_votes /
            # _verdict_for. imposters = IDs currently leaning NOT_TOMATO.
            imposters = {
                pid for pid in self._node._vlm_votes
                if self._node._verdict_for(pid) == "imposter"
            }

        # Auto-evict LOST cards that have aged past LOST_CARD_TTL_S so the
        # catalog stays current during demos. Picked cards are kept forever
        # as session history.
        now = time.monotonic()
        to_evict = [
            pid for pid, entry in catalog.items()
            if (entry.get("_lost")
                and not entry.get("_picked")
                and now - entry.get("_lost_at", now) > LOST_CARD_TTL_S)
        ]
        if to_evict:
            with self._node._lock:
                for pid in to_evict:
                    self._node._catalog.pop(pid, None)
                    catalog.pop(pid, None)

        # Instantiate cards for newly seen IDs
        for pid in catalog:
            if pid not in self._cards:
                card = TomatoCard(pid, self._grid_w)
                card.clicked.connect(self._on_card_clicked)
                self._cards[pid] = card

        # Tear down widgets for evicted ids
        for pid in to_evict:
            card = self._cards.pop(pid, None)
            if card is not None:
                self._grid_l.removeWidget(card)
                card.deleteLater()
            # If the popup was inspecting an evicted card, close it.
            if self._popup_pid == pid:
                self._close_card_popup()

        # Refresh content of all cards
        for pid, card in self._cards.items():
            if pid in catalog:
                card.refresh(
                    catalog[pid],
                    pid in picked,
                    vlm_id,
                    is_imposter=pid in imposters,
                )

        # Re-layout when sort order or membership changed
        sorted_pids = sorted(self._cards, key=lambda p: self._cards[p].sort_key())
        if sorted_pids != self._sort_order:
            self._sort_order = sorted_pids
            for i, pid in enumerate(sorted_pids):
                row, col = divmod(i, CATALOG_COLS)
                self._grid_l.addWidget(self._cards[pid], row, col)

    # ── Panel 4 ───────────────────────────────────────────────────────────────

    def _tick_panel4(self) -> None:
        with self._node._lock:
            det_window = list(self._node._det_window)
            track_ts   = list(self._node._track_ts)
            catalog    = dict(self._node._catalog)
            picked     = set(self._node._picked_ids)
            lost_count = self._node._lost_count
            health     = dict(self._node._health)
            vlm_id     = self._node._vlm_id
            vlm_reason = self._node._vlm_reason
            vlm_history = list(self._node._vlm_history)
            vlm_history_gen = self._node._vlm_history_gen

        # Detection rate
        if det_window:
            n_det = sum(det_window)
            rate  = n_det / len(det_window) * 100.0
            self._m_rate.setText(f"{rate:.0f}%  ({n_det}/{len(det_window)} frames)")
        else:
            self._m_rate.setText("— (no frames yet)")

        # Mean inter-frame latency from tomato_tracks publish timestamps
        if len(track_ts) >= 2:
            deltas = [track_ts[i + 1] - track_ts[i] for i in range(len(track_ts) - 1)]
            self._m_latency.setText(f"~{sum(deltas) / len(deltas):.1f}s/frame")
        else:
            self._m_latency.setText("—")

        active_n = sum(
            1 for e in catalog.values()
            if not e.get("_lost") and not e.get("_picked")
        )
        self._m_active.setText(str(active_n))
        self._m_lost.setText(str(lost_count))
        self._m_picked.setText(str(len(picked)))

        # Node health dots
        now = time.monotonic()
        for dot, key in (
            (self._h_det, "detector"),
            (self._h_spa, "spatial"),
            (self._h_trk, "tracker"),
            (self._h_vlm, "qwen_vl"),
        ):
            last = health[key]
            if last == 0.0:
                color = _QT_GRAY
            elif now - last <= HEALTH_TIMEOUT_S:
                color = _QT_GREEN
            else:
                color = _QT_RED
            dot.setStyleSheet(f"color:{color};border:none;")

        # ── VLM verdict badge (parsed from latest response) ────────────────
        if vlm_reason:
            label, fg, bg = _parse_vlm_verdict(vlm_reason)
            self._vlm_verdict_lbl.setText(label)
            self._vlm_verdict_lbl.setStyleSheet(
                f"background:{bg};color:{fg};border:1px solid {fg};border-radius:4px;"
            )
        else:
            self._vlm_verdict_lbl.setText("— awaiting first response —")
            self._vlm_verdict_lbl.setStyleSheet(
                "background:#252525;border:1px solid #3a3a3a;border-radius:4px;color:#888;"
            )

        # ── VLM reasoning history — rebuild only when contents changed ─────
        if vlm_history_gen != self._last_vlm_history_gen:
            self._last_vlm_history_gen = vlm_history_gen
            import re as _re
            _id_pat = _re.compile(r"(?i)\b(?:candidate|tomato)\s*#?\s*\d+\b")
            lines: list[str] = []
            for entry in vlm_history:
                raw = entry["text"].strip()

                # Only show entries that are per-ID structured responses.
                # This filters out long free-form narratives ("Tomato 0 is
                # likely a tomato due to its red color and shape, but it
                # could also be mistaken...") that are inaccurate and
                # confusing. Structured responses have at least one
                # "Candidate #N" or "VERDICT:" tag. Skip everything else.
                is_structured = (
                    len(_id_pat.findall(raw)) >= 1
                    or "VERDICT:" in raw.upper()
                )
                if not is_structured:
                    continue

                pid_str = f"#{entry['pid']}" if entry["pid"] is not None else "—"
                text = raw.replace("\n", "  ").strip()

                # Highlight structural tags.
                for tag, color in (
                    ("RIPENESS:", _QT_AMBER),
                    ("PICK PATH:", _QT_CYAN),
                    ("VERDICT:", _QT_GREEN),
                ):
                    text = text.replace(
                        tag,
                        f'<span style="color:{color};font-weight:bold;">{tag}</span>',
                    )
                lines.append(
                    f'<div style="margin:0 0 6px 0;">'
                    f'<span style="color:{_QT_AMBER};font-weight:bold;">'
                    f'[{entry["ts"]}] target {pid_str}</span><br>'
                    f'<span style="color:#cccccc;">{text}</span></div>'
                )
            if lines:
                self._vlm_reasoning_area.setHtml(
                    '<div style="font-family:monospace;font-size:9pt;line-height:1.4;">'
                    + "".join(lines)
                    + "</div>"
                )
            else:
                self._vlm_reasoning_area.setPlainText(
                    "Waiting for first VLM analysis... (Qwen-VL only runs on smoothed tracks, age ≥ 3)"
                )
            self._vlm_reasoning_area.verticalScrollBar().setValue(0)

    # ── Menu actions ──────────────────────────────────────────────────────────

    def _reset_catalog(self) -> None:
        with self._node._lock:
            self._node._catalog.clear()
            self._node._picked_ids.clear()
            self._node._vlm_id    = None
            self._node._vlm_votes.clear()
            self._node._vlm_reason = ""
            self._node._vlm_history.clear()
            self._node._vlm_history_gen += 1
            self._node._per_id_reasoning.clear()
            self._node._seen_ids.clear()
            self._node._lost_count = 0
        for card in self._cards.values():
            self._grid_l.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._sort_order = []
        self._close_card_popup()
        # Tell the tracker to wipe its registry and restart IDs from 0.
        msg = String()
        msg.data = "reset"
        self._node._reset_pub.publish(msg)

    def _clear_log(self) -> None:
        with self._node._lock:
            self._node._log_entries.clear()
            self._node._log_gen += 1
        self._event_log.clear()

    # ── Catalog card click → VLM reasoning popup ───────────────────────────

    def _on_card_clicked(self, pid: int) -> None:
        """Toggle a small popup showing the VLM's analysis of this track.

        Same pid clicked twice → close.
        Different pid clicked → re-target the existing popup.
        """
        if self._popup_pid == pid and self._popup is not None and self._popup.isVisible():
            self._popup.hide()
            self._popup_pid = None
            return

        with self._node._lock:
            catalog = dict(self._node._catalog)
            picked  = set(self._node._picked_ids)
            imposters = {
                p for p in self._node._vlm_votes
                if self._node._verdict_for(p) == "imposter"
            }
            vlm_id       = self._node._vlm_id
            latest_text  = self._node._vlm_reason        # full latest VLM response
            stored_text  = self._node._per_id_reasoning.get(pid, "")
            votes = self._node._vlm_votes.get(
                pid, {"imposter": 0, "real": 0}
            )

        # Build the reasoning text for this specific pid, computed fresh at
        # click time so it always reflects the most recent VLM output rather
        # than whatever was stored on an earlier cycle.
        #
        # Priority:
        #   1. Live parse of the latest VLM response (most responsive).
        #   2. Stored per-id reasoning dict (previous cycles).
        #   3. The whole latest response if it contains a single-candidate
        #      VERDICT verdict (single-tomato path has no "Candidate N").
        #   4. Fallback message.
        live = _extract_per_id_reasoning(latest_text)
        reasoning_text = (
            live.get(pid)
            or stored_text
            or ("VERDICT: " in latest_text.upper() and latest_text.strip())
            or "No VLM analysis for this track yet."
        )

        entry = catalog.get(pid)
        if entry is None:
            return

        z = entry["centroid"]["z"]
        r = entry["sphere"]["radius"] * 100.0
        conf = entry["confidence"]
        age = entry.get("age", 0)

        # Status verdict matches the catalog card priority order.
        if pid in picked:
            verdict, vfg, vbg = "✓ PICKED",      _QT_RED,   "#2a0d0d"
        elif pid in imposters:
            verdict, vfg, vbg = "✗ NOT TOMATO",  _QT_RED,   "#2a0d0d"
        elif pid == vlm_id:
            verdict, vfg, vbg = "→ PICKING",      _QT_BLUE,  "#0d1733"
        elif entry.get("_lost"):
            verdict, vfg, vbg = "LOST",           _QT_GRAY,  "#222"
        elif entry.get("smoothed") and age >= 3:
            verdict, vfg, vbg = "ACTIVE",         _QT_GREEN, "#0d2218"
        else:
            verdict, vfg, vbg = "CONVERGING",     _QT_AMBER, "#261a0d"

        if self._popup is None:
            self._popup = self._build_card_popup()

        self._popup_id_lbl.setText(f"#{pid}")
        self._popup_verdict_lbl.setText(verdict)
        self._popup_verdict_lbl.setStyleSheet(
            f"color:{vfg};background:{vbg};border:1px solid {vfg};"
            f"border-radius:4px;padding:2px 6px;"
        )
        self._popup_stats_lbl.setText(
            f"distance: {z:.2f} m   |   radius: {r:.1f} cm\n"
            f"confidence: {conf:.2f}   |   age: {age} frames\n"
            f"VLM votes:  imposter ×{votes['imposter']}   "
            f"real ×{votes['real']}"
        )
        self._popup_reason_lbl.setText(reasoning_text)

        # Position the popup in MainWindow-local coords so it stays inside
        # the dashboard frame (important under VNC where free-floating Qt
        # windows can land off-screen). Prefer to the RIGHT of the card,
        # vertically aligned to its top. If the card sits near the right
        # edge of the window, fall back to BELOW the card. Final clamp
        # keeps the popup fully on-screen either way.
        card = self._cards.get(pid)
        if card is not None:
            self._popup.adjustSize()
            pw, ph = self._popup.width(), self._popup.height()
            ww, wh = self.width(), self.height()
            margin = 8

            top_right = card.mapTo(self, card.rect().topRight())
            bottom_left = card.mapTo(self, card.rect().bottomLeft())

            # Try to the right of the card first.
            x = top_right.x() + 6
            y = top_right.y()
            if x + pw + margin > ww:
                # Not enough room on the right; place it below the card.
                x = bottom_left.x()
                y = bottom_left.y() + 6
                # If the card is also near the bottom, lift the popup so it
                # fits above the card instead.
                if y + ph + margin > wh:
                    y = max(margin, card.mapTo(self, card.rect().topLeft()).y() - ph - 6)

            x = max(margin, min(x, ww - pw - margin))
            y = max(margin, min(y, wh - ph - margin))
            self._popup.move(x, y)

        self._popup.show()
        self._popup.raise_()
        self._popup_pid = pid

    def _build_card_popup(self) -> QFrame:
        """Construct the reusable popup widget for catalog card details.

        Child of MainWindow (not Qt.Tool window) so it renders reliably under
        VNC/Xvfb. Stays inside the dashboard frame and can be positioned in
        MainWindow-local coords. We raise() it on show to put it above the
        catalog grid.
        """
        popup = QFrame(self)
        popup.setFixedWidth(340)
        popup.setStyleSheet(
            "QFrame{background:#1c1c1c;border:1px solid #4a4a4a;border-radius:6px;}"
        )
        # Hidden until first show; on top of sibling panels when visible.
        popup.hide()

        outer = QVBoxLayout(popup)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._popup_id_lbl = QLabel("#?")
        self._popup_id_lbl.setFont(QFont("Monospace", 11, QFont.Bold))
        self._popup_id_lbl.setStyleSheet(f"color:{_QT_CYAN};border:none;")
        header.addWidget(self._popup_id_lbl)

        self._popup_verdict_lbl = QLabel("—")
        self._popup_verdict_lbl.setFont(QFont("Monospace", 9, QFont.Bold))
        header.addWidget(self._popup_verdict_lbl)
        header.addStretch()

        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton{background:#3a3a3a;color:#ddd;border:none;border-radius:10px;}"
            "QPushButton:hover{background:#5a1a1a;color:#fff;}"
        )
        close_btn.clicked.connect(self._close_card_popup)
        header.addWidget(close_btn)
        outer.addLayout(header)

        self._popup_stats_lbl = QLabel("")
        self._popup_stats_lbl.setFont(QFont("Monospace", 8))
        self._popup_stats_lbl.setStyleSheet(f"color:#aaa;border:none;")
        outer.addWidget(self._popup_stats_lbl)

        outer.addWidget(self._sep())

        reason_header = QLabel("VLM analysis of this track:")
        reason_header.setFont(QFont("Monospace", 8, QFont.Bold))
        reason_header.setStyleSheet(f"color:{_QT_GRAY};border:none;")
        outer.addWidget(reason_header)

        self._popup_reason_lbl = QLabel("")
        self._popup_reason_lbl.setFont(QFont("Monospace", 9))
        self._popup_reason_lbl.setWordWrap(True)
        self._popup_reason_lbl.setStyleSheet(
            "color:#cccccc;border:none;background:#111;padding:6px;border-radius:3px;"
        )
        self._popup_reason_lbl.setMinimumHeight(60)
        outer.addWidget(self._popup_reason_lbl)

        return popup

    def _close_card_popup(self) -> None:
        if self._popup is not None:
            self._popup.hide()
        self._popup_pid = None

    def _show_topics(self) -> None:
        QMessageBox.information(self, "Subscribed topics", "\n".join([
            "/agrobot/debug_image              sensor_msgs/Image         (SENSOR_QOS)",
            "/camera/camera/color/image_raw    sensor_msgs/Image         (SENSOR_QOS, fallback)",
            "/camera/camera/color/camera_info  sensor_msgs/CameraInfo    (SENSOR_QOS)",
            "/agrobot/detections               vision_msgs/Detection2DArray",
            "/agrobot/tomato_spatial           std_msgs/String (JSON)",
            "/agrobot/tomato_tracks            std_msgs/String (JSON)",
            "/agrobot/pick_target              geometry_msgs/PoseStamped",
            "/agrobot/vlm_reasoning            std_msgs/String",
            "/agrobot/vlm_selection            std_msgs/String (JSON)",
            "/agrobot/safe_to_pick             std_msgs/Bool",
            "/agrobot/mark_picked              std_msgs/String (JSON {persistent_id: N})",
        ]))

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802
        rclpy.shutdown()
        event.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()
    node = AgroVizNode()

    # rclpy.spin blocks; run it in a daemon thread so the Qt event loop owns
    # the main thread (required by most platform GUI toolkits, including PyQt5)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    win = MainWindow(node)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
