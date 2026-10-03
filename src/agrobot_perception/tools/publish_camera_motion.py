"""
publish_camera_motion.py — Test helper for /agrobot/camera_motion

Publishes a single Vector3 to /agrobot/camera_motion so you can verify the
tracker's motion-compensation path without the arm controller in the loop.

Usage (inside the ROS container, ROS_DOMAIN_ID=42):
    python3 perception/tools/publish_camera_motion.py --dx 0.0 --dy 0.0 --dz 0.66

The displacement is in CAMERA OPTICAL FRAME:
    +X = right in image
    +Y = down in image
    +Z = forward (out of camera lens)

For a rail-mounted camera moving forward along the rail (advancing toward the
plant wall), use +dz. For a rail panning sideways (left → right across the
wall), use +dx.

Realistic per-step magnitudes for the 20%-overlap step-and-shoot scan
(640×480, fx≈387, working depth z=0.5–1.0 m):
    z=0.5 m  → advance ~0.66 m per step
    z=1.0 m  → advance ~1.32 m per step

The tracker logs `Applied camera motion compensation:` when it consumes
a motion vector during the next /agrobot/tomato_spatial callback.
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node


class _Publisher(Node):
    def __init__(self) -> None:
        super().__init__("camera_motion_test_publisher")
        self._pub = self.create_publisher(Vector3, "/agrobot/camera_motion", 10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0.0,
                        help="X displacement in camera optical frame (m).")
    parser.add_argument("--dy", type=float, default=0.0,
                        help="Y displacement in camera optical frame (m).")
    parser.add_argument("--dz", type=float, default=0.0,
                        help="Z displacement in camera optical frame (m).")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Publish N times spaced 0.5s apart (default: 1).")
    args = parser.parse_args()

    rclpy.init()
    node = _Publisher()

    # Wait briefly for the discovery handshake; without this the first
    # message can be silently dropped on a fresh node.
    time.sleep(0.5)

    msg = Vector3()
    msg.x, msg.y, msg.z = args.dx, args.dy, args.dz

    for i in range(args.repeat):
        node._pub.publish(msg)
        node.get_logger().info(
            f"[{i + 1}/{args.repeat}] published Δ=({msg.x:+.3f}, "
            f"{msg.y:+.3f}, {msg.z:+.3f}) m to /agrobot/camera_motion"
        )
        if i < args.repeat - 1:
            time.sleep(0.5)

    # Give the message a moment to leave the buffer before shutdown.
    time.sleep(0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
