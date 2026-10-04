"""
qwen_vl_node.py — VLM-Guided Tomato Pick Selection (NODE 3)

Architecture
------------
This is NODE 3 in the Agrobot TOM v2 picking pipeline.
It receives persistent tomato tracks from /agrobot/tomato_tracks (NODE 2b output).
It shows each tomato's JPEG crop to Qwen2.5-VL-3B-Instruct.
It selects a tomato for the arm using ripeness, size, and accessibility reasoning.

Why Qwen2.5-VL-3B over heuristics?
  The naive heuristic (pick closest / highest score) cannot distinguish a ripe
  red tomato from an unripe green one at the same depth. Qwen-VL gives a
  language-conditioned pick policy — the operator says "pick the reddest" and
  the system adapts without retraining. All inference runs locally on the NucBox
  (96GB unified RAM, ~2GB for 3B bfloat16) — no cloud dependency.

Why not GPT-4V?
  API latency (1-5s + network) + no offline capability. Greenhouse rows often
  lack reliable internet. Local Qwen-VL runs in ~10-30s on CPU, acceptable
  since the detector itself takes ~17 s per frame.

Inference model:
  Qwen/Qwen2.5-VL-3B-Instruct loaded in bfloat16 on CPU.
  3B × 2 bytes ≈ 6GB RAM — well within the 96GB NucBox budget.
  First run downloads from HuggingFace (~6GB). Subsequent runs load from
  models/qwen_vl/ if pre-saved there.

Data Flow
---------
  /agrobot/tomato_tracks (String JSON, from NODE 2b)
      │  list of {persistent_id, centroid, sphere, confidence, clipped_image, smoothed}
      ▼
  QwenVLNode._tracks_callback()
      │  skip if ≤0 smoothed tracks or inference already running
      │  decode clipped_image JPEGs → PIL Images
      │  build multi-image prompt → model.generate()
      │  parse persistent_id from response
      ▼
  /agrobot/pick_target     geometry_msgs/PoseStamped  Selected centroid, camera frame
  /agrobot/vlm_reasoning   std_msgs/String            Full VLM text response (logging)
  /agrobot/vlm_selection   std_msgs/String            JSON: selected tomato details

Topics
------
  Subscribed:
    /agrobot/tomato_tracks   std_msgs/String   Tracked + smoothed tomato JSON array

  Published:
    /agrobot/pick_target     geometry_msgs/PoseStamped
                             frame_id=camera_color_optical_frame, identity orientation.
                             Dani's arm planner feeds this directly to MoveIt2.
    /agrobot/vlm_reasoning   std_msgs/String   Raw VLM response for HIL logging.
    /agrobot/vlm_selection   std_msgs/String   JSON with full selected tomato record.

Parameters
----------
  model_path        string  Local path to saved model dir, or HuggingFace model ID.
                            (default: "Qwen/Qwen2.5-VL-3B-Instruct")
  pick_policy       string  Prompt style: "ripe_first" | "closest_first" | "largest_first"
                            (default: "ripe_first")
  min_smoothed_age  int     Only consider tracks with age >= this value (default: 3)
  max_new_tokens    int     Max tokens for VLM response (default: 64)
  tracks_topic      string  Input topic (default: /agrobot/tomato_tracks)

Target environment: [NUCBOX] — requires ~6GB RAM for model + PyTorch CPU inference.
Sprint: 4 — VLM-guided pick policy (GRAND_PLAN §4.1).
"""

from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped

try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# Qwen-VL imports — node starts without them but logs a warning and falls back
# to heuristic (pick closest smoothed tomato)
try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    _VLM_OK = True
except ImportError:
    _VLM_OK = False


# Prompt Templates

_POLICY_PROMPTS = {
    "ripe_first": (
        "You are guiding an agricultural robot arm to pick tomatoes.\n"
        "Prefer: red/ripe over green/unripe, then larger, then closer to camera.\n"
    ),
    "closest_first": (
        "You are guiding an agricultural robot arm to pick tomatoes.\n"
        "Prefer: the tomato closest to the camera (smallest z distance).\n"
    ),
    "largest_first": (
        "You are guiding an agricultural robot arm to pick tomatoes.\n"
        "Prefer: the largest tomato by apparent size, then ripeness.\n"
    ),
}

_SINGLE_PROMPT = (
    "{policy_text}"
    "You are assessing a tomato for an autonomous harvesting robot.\n"
    "Distance: {z:.2f}m | Estimated radius: {r:.1f}cm\n\n"
    "This tomato has already been confirmed by the detection pipeline.\n"
    "Your ONLY job: evaluate ripeness and pick readiness.\n\n"
    "Respond in EXACTLY this format (3 lines, no extras):\n"
    "RIPENESS: [color uniformity, surface quality — 1 sentence]\n"
    "PICK PATH: [clear access or occlusion — 1 sentence]\n"
    "VERDICT: READY | NOT_READY | UNCERTAIN"
)

def _extract_per_id_verdicts(
    response: str, candidate_pids: list[int]
) -> dict[int, str]:
    """Parse a VLM response into a per-candidate verdict.

    Returns {pid: "real" | "imposter" | "unknown"}.

    Supports two formats:
    1. New structured: "Candidate #N: apple | reason" or "Candidate #N: tomato | reason"
    2. Legacy narrative: sentences containing "Candidate N is likely a tomato/apple/..."

    Priority: new structured format wins over legacy if both appear.
    """
    import re

    verdicts: dict[int, str] = {pid: "unknown" for pid in candidate_pids}

    # New format: "Candidate #N: [type] | [reason]"
    # Parse lines matching "Candidate #N:" at the start
    structured_pattern = re.compile(
        r"(?i)candidate\s*#?\s*(\d+)\s*:\s*([^\n|]+)",
        re.MULTILINE
    )
    # Keywords that indicate tomato in the type field
    tomato_types = re.compile(
        r"(?i)\b(ripe\s+tomato|tomato|cherry\s+tomato)\b"
    )
    # Keywords that indicate imposter in the type field
    # Only reject clearly non-tomato shapes. Red/round objects are tomatoes
    imposter_types = re.compile(
        r"(?i)\b(pear|pepper|green\s+apple|elongated|background|unknown\s+fruit)\b"
    )

    found_structured = False
    for m in structured_pattern.finditer(response):
        pid = int(m.group(1))
        type_text = m.group(2).strip().lower()
        if pid not in verdicts:
            continue
        found_structured = True
        if tomato_types.search(type_text):
            verdicts[pid] = "real"
        elif imposter_types.search(type_text):
            verdicts[pid] = "imposter"
        # Else: leave as unknown

    if found_structured:
        return verdicts

    # Legacy narrative format
    # Only reject clearly wrong shapes — NOT red round objects (those are tomatoes)
    imposter_keywords = (
        "a pear", "resembles a pear", "looks like a pear", "appears to be a pear",
        "a pepper", "green pepper", "resembles a pepper", "looks like a pepper",
        "green apple",
        "not a tomato", "not tomato",
    )
    real_keywords = (
        "real tomato", "a real tomato",
        "is a tomato", "is the tomato", "is likely a tomato",
        "a ripe tomato", "appears to be a tomato",
        "looks like a tomato", "looks like the tomato",
        "definitely a tomato", "clearly a tomato",
        "is the best choice", "best candidate",
        "ripe tomato", "cherry tomato",
    )

    sentences = re.split(r"(?<=[.!?])\s+|\n+", response)
    id_pattern = re.compile(r"(?i)\b(?:tomato|candidate)\s*#?\s*(\d+)\b")

    for sentence in sentences:
        s_lower = sentence.lower()
        ids_here = [int(m.group(1)) for m in id_pattern.finditer(sentence)
                    if int(m.group(1)) in verdicts]
        if not ids_here:
            continue

        is_imposter = any(kw in s_lower for kw in imposter_keywords)
        is_real     = any(kw in s_lower for kw in real_keywords)

        if is_imposter:
            for pid in ids_here:
                verdicts[pid] = "imposter"
        elif is_real:
            for pid in ids_here:
                if verdicts[pid] != "imposter":
                    verdicts[pid] = "real"

    return verdicts


_MULTI_PROMPT_HEADER = (
    "{policy_text}"
    "You are assessing {n} tomatoes for an autonomous harvesting robot.\n"
    "All candidates are confirmed tomatoes detected by the pipeline.\n"
    "Your ONLY job: rank by ripeness and accessibility, then pick the best one.\n\n"
    "RIPENESS PRIORITY (best → worst):\n"
    "  1. Deep uniform red — pick immediately\n"
    "  2. Red-orange — good, pick if nothing better\n"
    "  3. Partially green — not ready\n\n"
    "You MUST analyze ALL {n} candidates numbered 0 through {n_minus_1}.\n"
    "Do NOT skip any candidate.\n"
)

_MULTI_PROMPT_FOOTER = (
    "\nRespond in EXACTLY this format — one ripeness line per candidate, "
    "then PICK on the last line:\n\n"
    "Candidate #0: [color/ripeness — 1 phrase]\n"
    "Candidate #1: [color/ripeness — 1 phrase]\n"
    "(continue for ALL {n} candidates)\n"
    "PICK: [number of the MOST RIPE and accessible tomato]\n\n"
    "RULES:\n"
    "1. Every candidate gets exactly one line before PICK\n"
    "2. PICK must be the LAST line — a single integer\n"
    "3. If all look equally ripe, PICK the closest (smallest z distance)\n\n"
    "Example:\n"
    "Candidate #0: deep red, uniform, close to camera\n"
    "Candidate #1: red-orange, slightly less ripe\n"
    "PICK: 0\n"
)


class QwenVLNode(Node):
    """ROS 2 node that uses Qwen2.5-VL to select which tomato the arm should pick."""

    def __init__(self) -> None:
        super().__init__("qwen_vl")

        # Parameters
        self.declare_parameter("model_path", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.declare_parameter("pick_policy", "ripe_first")
        self.declare_parameter("min_smoothed_age", 3)
        self.declare_parameter("max_new_tokens", 256)
        self.declare_parameter("tracks_topic", "/agrobot/tomato_tracks")

        self._model_path: str = self.get_parameter("model_path").value
        self._policy: str = self.get_parameter("pick_policy").value
        self._min_age: int = self.get_parameter("min_smoothed_age").value
        self._max_tokens: int = self.get_parameter("max_new_tokens").value
        tracks_topic: str = self.get_parameter("tracks_topic").value

        # Model state
        self._model = None
        self._processor = None
        self._vlm_available = False
        self._inference_running = False   # Prevent callback re-entry during slow inference

        # Publishers
        self._pick_pub = self.create_publisher(
            PoseStamped, "/agrobot/pick_target", 10
        )
        self._reasoning_pub = self.create_publisher(
            String, "/agrobot/vlm_reasoning", 10
        )
        self._selection_pub = self.create_publisher(
            String, "/agrobot/vlm_selection", 10
        )

        # Per-candidate locked verdicts (evaluate-once architecture)
        # Evaluate each candidate EXACTLY ONCE when it first converges
        # Lock its verdict after the first VLM response that includes it
        # Keep the verdict until Reset Catalog
        # Do not call the VLM again for that candidate
        #
        # "tomato"     → candidate confirmed ripe. always publish pick_target
        # "not_tomato" → candidate confirmed non-tomato. never publish pick_target
        # Absent       → not yet evaluated. include in next VLM call
        #
        # Call the model once per candidate to prevent changes between verdicts
        # Keep the verdict so the system converges in 1 cycle
        self._locked_verdicts: dict[int, str] = {}
        # Legacy vote dict retained for dashboard compatibility (it reads
        # /agrobot/vlm_selection which carries imposters/real_tomatoes lists)
        self._vlm_votes: dict[int, dict] = {}
        self.create_subscription(
            String, "/agrobot/reset_tracker", self._on_reset, 10
        )

        # Subscription
        self.create_subscription(String, tracks_topic, self._tracks_callback, 10)

        # Load model in background thread so node starts immediately
        if _VLM_OK and _PIL_OK:
            t = threading.Thread(target=self._load_model, daemon=True)
            t.start()
        else:
            missing = []
            if not _VLM_OK:
                missing.append("transformers / qwen-vl-utils")
            if not _PIL_OK:
                missing.append("Pillow")
            self.get_logger().warn(
                f"Missing: {', '.join(missing)}. "
                "Running in heuristic mode (closest smoothed tomato). "
                "Install with: pip install transformers qwen-vl-utils Pillow"
            )

        self.get_logger().info(
            f"QwenVLNode initialized. "
            f"policy='{self._policy}'  min_age={self._min_age}  "
            f"model='{self._model_path}'"
        )

    # Model loading

    def _load_model(self) -> None:
        """Load Qwen2.5-VL in a background thread — ~30s first run from hub."""
        self.get_logger().info(
            f"Loading Qwen2.5-VL from '{self._model_path}' "
            "(this takes ~30s on first run; downloading ~6GB if not cached)..."
        )
        try:
            # Check for local save first (avoids re-download after first run)
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            local_path = repo_root / "models" / "qwen_vl"
            source = str(local_path) if local_path.exists() else self._model_path

            self._processor = AutoProcessor.from_pretrained(
                source,
                # Limit image resolution — crops are small (~100-200px)
                # Default max_pixels is 12845056 (1280*28*28). We don't need that
                min_pixels=224 * 224,
                max_pixels=448 * 448,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source,
                # bfloat16 on CPU: 3B × 2 bytes ≈ 6GB RAM. Fits in 96GB NucBox
                # No device_map — avoids the accelerate dependency. Without it,
                # transformers loads to CPU by default when no GPU is visible
                # (AGROBOT_FORCE_CPU=1 ensures HIP/CUDA are hidden)
                torch_dtype=torch.bfloat16,
            )
            self._model.eval()
            self._vlm_available = True
            self.get_logger().info(
                "Qwen2.5-VL loaded. VLM-guided pick selection active."
            )
        except Exception as exc:
            self.get_logger().error(
                f"Qwen2.5-VL load failed: {exc}. "
                "Falling back to heuristic (closest smoothed tomato)."
            )

    # Main callback

    def _tracks_callback(self, msg: String) -> None:
        """Called on each /agrobot/tomato_tracks message."""
        if self._inference_running:
            self.get_logger().debug(
                "Inference still running — skipping stale tracks message."
            )
            return

        try:
            tracks: list[dict] = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"JSON parse error: {exc}")
            return

        # Act only on converged tracks with enough observations for the EMA
        candidates = [
            t for t in tracks
            if t.get("age", 0) >= self._min_age
        ]

        if not candidates:
            self.get_logger().debug(
                f"No smoothed tracks yet (min_age={self._min_age}). "
                f"Received {len(tracks)} raw track(s)."
            )
            return

        # Evaluate-once: only call VLM for candidates that don't have a verdict yet
        unevaluated = [
            t for t in candidates
            if t["persistent_id"] not in self._locked_verdicts
        ]

        if not unevaluated:
            # All candidates already evaluated — re-publish the best without
            # running VLM again. All locked candidates are tomatoes
            self.get_logger().info(
                f"All {len(candidates)} candidates already evaluated — "
                "re-publishing best from locked verdicts."
            )
            ready = [
                t for t in candidates
                if self._locked_verdicts.get(t["persistent_id"]) == "tomato"
            ]
            if ready:
                best = min(ready, key=lambda t: t["centroid"]["z"])
                self._publish_selection(best)
            return

        self.get_logger().info(
            f"Received {len(candidates)} smoothed track(s) — "
            f"{len(unevaluated)} unevaluated, running VLM."
        )

        if self._vlm_available and self._model is not None:
            self._inference_running = True
            try:
                selected = self._run_vlm(candidates)
            finally:
                self._inference_running = False
        else:
            # Heuristic fallback: pick the closest (minimum z) smoothed tomato
            selected = min(candidates, key=lambda t: t["centroid"]["z"])
            self.get_logger().info(
                f"Heuristic: selected persistent_id={selected['persistent_id']} "
                f"(z={selected['centroid']['z']:.3f}m, closest)."
            )

        if selected is not None:
            self._publish_selection(selected)

    # VLM inference

    # Node-side imposter vote helpers

    def _on_reset(self, msg) -> None:
        self._locked_verdicts.clear()
        self._vlm_votes.clear()
        self.get_logger().info("VLM locked verdicts and votes cleared — fresh slate.")

    def _record_votes(
        self, imposters: list[int], reals: list[int]
    ) -> None:
        """Accumulate per-candidate classification votes."""
        for pid in imposters:
            self._vlm_votes.setdefault(pid, {"imposter": 0, "real": 0})["imposter"] += 1
        for pid in reals:
            self._vlm_votes.setdefault(pid, {"imposter": 0, "real": 0})["real"] += 1

    def _is_node_imposter(self, pid: int) -> bool:
        """True when the locked verdict for this candidate is not_tomato."""
        return self._locked_verdicts.get(pid) == "not_tomato"

    #

    def _decode_jpeg(self, b64: str) -> "PILImage.Image | None":
        """Decode a base64 JPEG string to a PIL Image."""
        try:
            return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as exc:
            self.get_logger().error(f"JPEG decode failed: {exc}")
            return None

    def _run_vlm(self, candidates: list[dict]) -> dict | None:
        """Run Qwen2.5-VL inference and return the selected tomato dict."""
        policy_text = _POLICY_PROMPTS.get(
            self._policy, _POLICY_PROMPTS["ripe_first"]
        )

        if len(candidates) == 1:
            t = candidates[0]
            pid = t["persistent_id"]
            img = self._decode_jpeg(t.get("clipped_image", ""))
            if img is None:
                self._locked_verdicts[pid] = "tomato"
                return t

            prompt_text = _SINGLE_PROMPT.format(
                policy_text=policy_text,
                z=t["centroid"]["z"],
                r=t["sphere"]["radius"] * 100,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            response = self._infer(messages)
            self.get_logger().info(f"VLM single-tomato response: '{response}'")
            reasoning_msg = String()
            reasoning_msg.data = response
            self._reasoning_pub.publish(reasoning_msg)

            # Lock as tomato — the upstream pipeline already confirmed it
            # evaluate-once: won't re-run VLM for this pid until reset
            self._locked_verdicts[pid] = "tomato"
            self._record_votes(imposters=[], reals=[pid])

            upper = response.upper()
            if "NOT_READY" in upper:
                self.get_logger().info(
                    f"VLM: tomato pid={pid} not ready to pick yet — skipping."
                )
                return None
            return t

        # Multiple candidates — show all crops in one prompt
        n = len(candidates)
        content: list[dict] = [
            {
                "type": "text",
                "text": _MULTI_PROMPT_HEADER.format(
                    policy_text=policy_text,
                    n=n,
                    n_minus_1=n - 1,
                ),
            }
        ]
        for t in candidates:
            img = self._decode_jpeg(t.get("clipped_image", ""))
            z = t["centroid"]["z"]
            r = t["sphere"]["radius"] * 100
            pid = t["persistent_id"]
            content.append({
                "type": "text",
                "text": f"\nCandidate #{pid} (distance={z:.2f}m, r={r:.1f}cm):",
            })
            if img is not None:
                content.append({"type": "image", "image": img})

        content.append({
            "type": "text",
            "text": _MULTI_PROMPT_FOOTER.format(n=n),
        })
        messages = [{"role": "user", "content": content}]

        response = self._infer(messages)
        self.get_logger().info(f"VLM multi-tomato response: '{response}'")
        reasoning_msg = String()
        reasoning_msg.data = response
        self._reasoning_pub.publish(reasoning_msg)

        # Lock every candidate because each has tomato confirmation
        # VLM only tells us which one to pick first
        candidate_pids = [t["persistent_id"] for t in candidates]
        newly_locked: list[int] = []
        for pid in candidate_pids:
            if pid not in self._locked_verdicts:
                self._locked_verdicts[pid] = "tomato"
                newly_locked.append(pid)
        self._record_votes(imposters=[], reals=newly_locked)

        # Use VLM's PICK: N selection. Use the closest candidate if parsing fails
        best = self._parse_selection(response, candidates)
        if best is None:
            best = min(candidates, key=lambda t: t["centroid"]["z"])
            self.get_logger().info(
                f"VLM gave no PICK — falling back to closest "
                f"persistent_id={best['persistent_id']}."
            )

        best["_per_id_imposters"] = []
        best["_per_id_reals"] = candidate_pids
        self.get_logger().info(
            f"VLM selected persistent_id={best['persistent_id']} "
            f"(z={best['centroid']['z']:.3f}m) — publishing pick_target."
        )
        return best

    def _infer(self, messages: list[dict]) -> str:
        """Run one forward pass through Qwen2.5-VL and return the response text."""
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                do_sample=False,     # Greedy decoding — deterministic, faster
            )

        # Strip the input prompt tokens. Keep only generated response
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    def _parse_selection(
        self, response: str, candidates: list[dict]
    ) -> dict | None:
        """Extract the best candidate from the VLM response.

        Search for 'PICK: N'. Use its last occurrence.
        If that fails, search for any integer in the response.
        Return None only if parsing fails completely.
        The caller then selects the closest candidate.
        """
        import re

        pid_set = {t["persistent_id"] for t in candidates}

        # Primary: "PICK: N" — last match wins
        pick_matches = re.findall(r"(?i)\bPICK\s*:\s*(\d+)", response)
        if pick_matches:
            pid = int(pick_matches[-1])
            for t in candidates:
                if t["persistent_id"] == pid:
                    return t
            self.get_logger().warn(
                f"VLM PICK:{pid} not in candidates {sorted(pid_set)}. Using closest."
            )
            return min(candidates, key=lambda t: t["centroid"]["z"])

        # Fallback: last integer in the response that is a valid pid
        nums = re.findall(r"\b(\d+)\b", response)
        for raw in reversed(nums):
            pid = int(raw)
            if pid in pid_set:
                return next(t for t in candidates if t["persistent_id"] == pid)

        self.get_logger().warn(
            f"Could not parse PICK from VLM response: '{response[:80]}…'. "
            "Caller will fall back to closest."
        )
        return None

    # Publishing

    def _publish_selection(self, tomato: dict) -> None:
        """Publish the selected tomato as a pick target for the arm planner."""
        now = self.get_clock().now().to_msg()
        pid = tomato["persistent_id"]
        c = tomato["centroid"]

        # PoseStamped: centroid in camera_color_optical_frame
        # Orientation is identity — arm planner determines approach angle from TF
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "camera_color_optical_frame"
        pose_msg.pose.position.x = float(c["x"])
        pose_msg.pose.position.y = float(c["y"])
        pose_msg.pose.position.z = float(c["z"])
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        self._pick_pub.publish(pose_msg)

        # Full selection record. Dashboard consumes this via /agrobot/vlm_selection
        # Imposters and real_tomatoes carry the per-ID natural-language
        # parse so the dashboard can render NOT_TOMATO / ACTIVE / PICKING
        # per card instead of applying a global verdict
        selection = {
            "persistent_id": pid,
            "centroid": tomato["centroid"],
            "sphere": tomato["sphere"],
            "confidence": tomato["confidence"],
            "age": tomato.get("age", 0),
            "imposters": tomato.get("_per_id_imposters", []),
            "real_tomatoes": tomato.get("_per_id_reals", []),
        }
        sel_msg = String()
        sel_msg.data = json.dumps(selection)
        self._selection_pub.publish(sel_msg)

        self.get_logger().info(
            f"Published pick_target: persistent_id={pid} "
            f"x={c['x']:+.3f} y={c['y']:+.3f} z={c['z']:.3f} m"
        )

    def _publish_veto(
        self,
        imposters: list[int],
        real_tomatoes: Optional[list[int]] = None,
    ) -> None:
        """Broadcast a VLM veto so consumers can flag the non-tomato tracks.

        Use the _publish_selection schema with persistent_id=-1 for "no pick this cycle".
        imposters contains IDs that the VLM explicitly marked as non-tomatoes.
        real_tomatoes contains IDs that it marked as real despite making no pick.
        For example, a multi-tomato pass may say "Tomato 1 is a tomato" but vote -1.
        The dashboard uses both lists to render each card's verdict.
        """
        veto = {
            "persistent_id": -1,
            "imposters": list(imposters),
            "real_tomatoes": list(real_tomatoes or []),
        }
        msg = String()
        msg.data = json.dumps(veto)
        self._selection_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QwenVLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
