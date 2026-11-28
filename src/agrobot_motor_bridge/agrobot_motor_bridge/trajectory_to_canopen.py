#!/usr/bin/env python3
"""Convert a MoveIt 2 JointTrajectory YAML into timed EPOS2 CANopen SDO commands."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml

# Default EPOS2 CANopen parameters matching epos2_setup.md
DEFAULT_JOINT = "arm_joint_1"
DEFAULT_NODE_ID = 24
DEFAULT_COUNTS_PER_REV = 25600
DEFAULT_GEAR_RATIO = 1.0
DEFAULT_INVERTED = False
DEFAULT_PROFILE_ACCEL_RPM_S = 200
DEFAULT_PROFILE_DECEL_RPM_S = 200
DEFAULT_SPEED_RPM = 1000
DEFAULT_CAN_INTERFACE = "vcan0"
DEFAULT_ZERO_START = True

# CANopen SDO request base ID and expedited write command bytes
SDO_REQUEST_BASE_ID = 0x600
SDO_WRITE_CMD_BY_SIZE = {1: 0x2F, 2: 0x2B, 4: 0x23}

# CiA 402 / EPOS2 object dictionary addresses
IDX_CONTROLWORD = 0x6040
IDX_MODES_OF_OPERATION = 0x6060
IDX_TARGET_POSITION = 0x607A
IDX_PROFILE_VELOCITY = 0x6081
IDX_PROFILE_ACCELERATION = 0x6083
IDX_PROFILE_DECELERATION = 0x6084

# CiA 402 Profile Position Mode and Controlword values
MODE_PROFILE_POSITION = 1
CW_DISABLE_VOLTAGE = 0x0000
CW_SHUTDOWN = 0x0006
CW_SWITCH_ON = 0x0007
CW_ENABLE_OPERATION = 0x000F
CW_NEW_SETPOINT_IMMEDIATE = 0x003F


def encode_sdo_write(
    node_id: int,
    index: int,
    subindex: int,
    value: int,
    size_bytes: int,
    signed: bool = False,
) -> tuple[int, bytes]:
    """Build an 8-byte expedited CANopen SDO write frame."""
    cmd = SDO_WRITE_CMD_BY_SIZE[size_bytes]
    raw_value = int(value).to_bytes(size_bytes, "little", signed=signed).ljust(4, b"\x00")
    payload = bytes([cmd]) + index.to_bytes(2, "little") + bytes([subindex]) + raw_value
    return SDO_REQUEST_BASE_ID + node_id, payload


def format_spaced_frame(can_id: int, payload: bytes) -> str:
    """Format a CAN frame as spaced uppercase hex bytes prefixed by CAN ID."""
    return f"{can_id:03X} " + " ".join(f"{byte:02X}" for byte in payload)


def format_cansend_arg(can_id: int, payload: bytes) -> str:
    """Format a CAN frame as a SocketCAN cansend argument."""
    return f"{can_id:03X}#" + payload.hex().upper()


def rad_to_counts(pos_rad: float, counts_per_rev: int, gear_ratio: float, inverted: bool) -> int:
    """Convert joint position in radians to signed EPOS2 quadcounts."""
    sign = -1.0 if inverted else 1.0
    return round(sign * pos_rad * gear_ratio * counts_per_rev / (2.0 * math.pi))


def rad_s_to_rpm(vel_rad_s: float, gear_ratio: float, default_speed_rpm: int) -> int:
    """Convert joint speed in rad/s to a positive EPOS2 motor speed in rpm."""
    if abs(vel_rad_s) == 0.0:
        return default_speed_rpm
    motor_rpm = abs(vel_rad_s) * gear_ratio * 60.0 / (2.0 * math.pi)
    return max(1, round(motor_rpm))


def load_joint_trajectory(yaml_path: Path) -> dict[str, Any]:
    """Load a JointTrajectory mapping from either flat or MoveIt-wrapped YAML."""
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if "joint_names" in data and "points" in data:
        return data
    if "trajectory" in data and "joint_trajectory" in data["trajectory"]:
        return data["trajectory"]["joint_trajectory"]
    raise ValueError(f"Unsupported trajectory YAML structure in {yaml_path}")


def convert_trajectory(
    input_path: Path,
    output_path: Path,
    joint_name: str = DEFAULT_JOINT,
    node_id: int = DEFAULT_NODE_ID,
    counts_per_rev: int = DEFAULT_COUNTS_PER_REV,
    gear_ratio: float = DEFAULT_GEAR_RATIO,
    inverted: bool = DEFAULT_INVERTED,
    accel_rpm_s: int = DEFAULT_PROFILE_ACCEL_RPM_S,
    decel_rpm_s: int = DEFAULT_PROFILE_DECEL_RPM_S,
    default_speed_rpm: int = DEFAULT_SPEED_RPM,
    can_interface: str = DEFAULT_CAN_INTERFACE,
    zero_start: bool = DEFAULT_ZERO_START,
) -> int:
    """Convert a trajectory YAML file into a timed EPOS2 CANopen command file."""
    trajectory = load_joint_trajectory(input_path)
    joint_names: list[str] = trajectory["joint_names"]
    if joint_name not in joint_names:
        raise ValueError(f"Joint '{joint_name}' not found in {joint_names}")
    joint_index = joint_names.index(joint_name)
    points: list[dict[str, Any]] = trajectory["points"]
    start_pos_rad = float(points[0]["positions"][joint_index]) if (zero_start and points) else 0.0

    sdo_tx_id = SDO_REQUEST_BASE_ID + node_id

    setup_entries = [
        ("Mode Profile Position (0x6060=1)", encode_sdo_write(node_id, IDX_MODES_OF_OPERATION, 0, MODE_PROFILE_POSITION, 1, signed=True)),
        (f"Profile Acceleration (0x6083={accel_rpm_s})", encode_sdo_write(node_id, IDX_PROFILE_ACCELERATION, 0, accel_rpm_s, 4)),
        (f"Profile Deceleration (0x6084={decel_rpm_s})", encode_sdo_write(node_id, IDX_PROFILE_DECELERATION, 0, decel_rpm_s, 4)),
        ("Shutdown (0x6040=0x0006)", encode_sdo_write(node_id, IDX_CONTROLWORD, 0, CW_SHUTDOWN, 2)),
        ("Switch On (0x6040=0x0007)", encode_sdo_write(node_id, IDX_CONTROLWORD, 0, CW_SWITCH_ON, 2)),
        ("Enable Operation (0x6040=0x000F)", encode_sdo_write(node_id, IDX_CONTROLWORD, 0, CW_ENABLE_OPERATION, 2)),
    ]

    trigger_arg = format_cansend_arg(*encode_sdo_write(node_id, IDX_CONTROLWORD, 0, CW_NEW_SETPOINT_IMMEDIATE, 2))
    clear_arg = format_cansend_arg(*encode_sdo_write(node_id, IDX_CONTROLWORD, 0, CW_ENABLE_OPERATION, 2))
    teardown_frame = encode_sdo_write(node_id, IDX_CONTROLWORD, 0, CW_DISABLE_VOLTAGE, 2)

    lines: list[str] = [
        f"# Joint ({joint_name}) EPOS2 CANopen commands generated from {input_path.name}",
        (
            f"# Node ID: {node_id} (SDO TX 0x{sdo_tx_id:03X}) | counts_per_rev: {counts_per_rev} "
            f"| gear_ratio: {gear_ratio} | inverted: {inverted} | accel_rpm_s: {accel_rpm_s} "
            f"| decel_rpm_s: {decel_rpm_s} | default_speed_rpm: {default_speed_rpm}"
        ),
        "#",
        "# Setup frames:",
    ]

    for label, (can_id, payload) in setup_entries:
        spaced = format_spaced_frame(can_id, payload)
        arg = format_cansend_arg(can_id, payload)
        lines.append(f"#   {label + ':':<36} {spaced}  (cansend {can_interface} {arg})")

    lines.extend([
        "#",
        "# Timed Profile Position Mode SDO frames (0x6081 velocity, 0x607A target, 0x6040 trigger 0x003F, 0x6040 clear 0x000F):",
        "# t_s       pos_rad   vel_rad_s   counts    rpm  cansend_args",
    ])

    for point in points:
        t_field = point["time_from_start"]
        t_s = float(t_field["sec"]) + float(t_field["nanosec"]) * 1e-9
        pos_rad = float(point["positions"][joint_index])
        velocities = point.get("velocities") or []
        vel_rad_s = float(velocities[joint_index]) if joint_index < len(velocities) else 0.0

        counts = rad_to_counts(pos_rad - start_pos_rad, counts_per_rev, gear_ratio, inverted)
        rpm = rad_s_to_rpm(vel_rad_s, gear_ratio, default_speed_rpm)

        vel_arg = format_cansend_arg(*encode_sdo_write(node_id, IDX_PROFILE_VELOCITY, 0, rpm, 4))
        pos_arg = format_cansend_arg(*encode_sdo_write(node_id, IDX_TARGET_POSITION, 0, counts, 4, signed=True))
        cansend_args = f"{vel_arg} {pos_arg} {trigger_arg} {clear_arg}"

        lines.append(f"{t_s:8.4f}  {pos_rad:9.5f}  {vel_rad_s:9.5f}  {counts:7d}  {rpm:5d}  {cansend_args}")

    lines.extend([
        "#",
        "# Teardown frame:",
        (
            f"#   {'Disable Voltage (0x6040=0x0000):':<36} "
            f"{format_spaced_frame(*teardown_frame)}  "
            f"(cansend {can_interface} {format_cansend_arg(*teardown_frame)})"
        ),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(points)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_yaml", type=Path, help="Path to MoveIt 2 JointTrajectory YAML")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("traj_output/arm_joint_1_canopen_commands.txt"),
        help="Output CANopen commands text file",
    )
    parser.add_argument("--joint", default=DEFAULT_JOINT, help="Joint name to convert")
    parser.add_argument("--node-id", type=int, default=DEFAULT_NODE_ID, help="CANopen node ID (1-127)")
    parser.add_argument("--counts-per-rev", type=int, default=DEFAULT_COUNTS_PER_REV, help="Encoder quadcounts per motor rev")
    parser.add_argument("--gear-ratio", type=float, default=DEFAULT_GEAR_RATIO, help="Motor-to-joint gear ratio")
    parser.add_argument("--inverted", action="store_true", default=DEFAULT_INVERTED, help="Invert joint direction")
    parser.add_argument("--profile-accel", type=int, default=DEFAULT_PROFILE_ACCEL_RPM_S, help="Profile acceleration in rpm/s")
    parser.add_argument("--profile-decel", type=int, default=DEFAULT_PROFILE_DECEL_RPM_S, help="Profile deceleration in rpm/s")
    parser.add_argument("--default-speed-rpm", type=int, default=DEFAULT_SPEED_RPM, help="Fallback speed in rpm when velocity is 0")
    parser.add_argument("--interface", default=DEFAULT_CAN_INTERFACE, help="CAN interface name shown in setup/teardown comments")
    parser.add_argument(
        "--absolute-counts",
        action="store_true",
        help="Use absolute URDF joint counts instead of offsetting the first waypoint to 0 qc",
    )
    return parser.parse_args()


def main() -> None:
    """Run the trajectory-to-CANopen converter."""
    args = parse_args()
    count = convert_trajectory(
        input_path=args.input_yaml,
        output_path=args.output,
        joint_name=args.joint,
        node_id=args.node_id,
        counts_per_rev=args.counts_per_rev,
        gear_ratio=args.gear_ratio,
        inverted=args.inverted,
        accel_rpm_s=args.profile_accel,
        decel_rpm_s=args.profile_decel,
        default_speed_rpm=args.default_speed_rpm,
        can_interface=args.interface,
        zero_start=not args.absolute_counts,
    )
    print(f"Wrote {count} timed CANopen waypoints to {args.output}")


if __name__ == "__main__":
    main()
