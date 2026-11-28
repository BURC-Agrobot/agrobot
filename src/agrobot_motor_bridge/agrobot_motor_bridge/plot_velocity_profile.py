#!/usr/bin/env python3
"""Plot position and velocity profiles from a generated CANopen commands file."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_canopen_commands(commands_path: Path) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Parse header metadata and timed waypoint columns from a CANopen command file."""
    meta: dict[str, str] = {"joint": "joint", "node_id": "?", "counts_per_rev": "25600", "gear_ratio": "1.0"}
    data: dict[str, list[float]] = {"t_s": [], "pos_rad": [], "vel_rad_s": [], "counts": [], "rpm": []}

    with commands_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                joint_match = re.search(r"Joint \(([^)]+)\)", line)
                if joint_match:
                    meta["joint"] = joint_match.group(1)
                for key, pattern in (
                    ("node_id", r"Node ID:\s*(\d+)"),
                    ("counts_per_rev", r"counts_per_rev:\s*(\d+)"),
                    ("gear_ratio", r"gear_ratio:\s*([0-9.]+)"),
                ):
                    match = re.search(pattern, line)
                    if match:
                        meta[key] = match.group(1)
                continue

            parts = line.split()
            data["t_s"].append(float(parts[0]))
            data["pos_rad"].append(float(parts[1]))
            data["vel_rad_s"].append(float(parts[2]))
            data["counts"].append(float(parts[3]))
            data["rpm"].append(float(parts[4]))

    return meta, data


def plot_profile(commands_path: Path, output_path: Path) -> None:
    """Render a two-panel position and velocity profile graph."""
    meta, data = parse_canopen_commands(commands_path)
    counts_per_rev = float(meta["counts_per_rev"])
    gear_ratio = float(meta["gear_ratio"])

    t_s = data["t_s"]
    counts = data["counts"]
    planned_rpm = [v * gear_ratio * 60.0 / (2.0 * math.pi) for v in data["vel_rad_s"]]
    cmd_rpm = data["rpm"]

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, dpi=150)

    ax_pos.plot(t_s, counts, color="#111111", linewidth=1.8)
    ax_pos.set_ylabel("Position (qc)")
    ax_pos.set_title(
        f"{meta['joint']} Trajectory (Node {meta['node_id']}, {int(counts_per_rev)} qc/rev)"
    )
    ax_pos.grid(True, linestyle="--", alpha=0.4)

    ax_deg = ax_pos.twinx()
    deg_per_qc = 360.0 / (counts_per_rev * gear_ratio)
    ax_deg.set_ylim(ax_pos.get_ylim()[0] * deg_per_qc, ax_pos.get_ylim()[1] * deg_per_qc)
    ax_deg.set_ylabel("Joint Angle (deg)")

    ax_vel.plot(t_s, planned_rpm, color="#111111", linewidth=1.8, label="MoveIt planned (rpm)")
    ax_vel.step(
        t_s,
        cmd_rpm,
        where="post",
        color="#D9534F",
        linewidth=1.3,
        linestyle="--",
        label="EPOS2 0x6081 commanded (int rpm)",
    )
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Motor Speed (rpm)")
    ax_vel.grid(True, linestyle="--", alpha=0.4)
    ax_vel.legend(loc="lower center", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands_txt", type=Path, help="Path to generated CANopen commands text file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (defaults to <input_stem>_velocity_profile.png beside input)",
    )
    return parser.parse_args()


def main() -> None:
    """Generate a velocity profile graph from a CANopen commands file."""
    args = parse_args()
    out_path = args.output or args.commands_txt.with_name(f"{args.commands_txt.stem}_velocity_profile.png")
    plot_profile(args.commands_txt, out_path)
    print(f"Saved velocity profile plot to {out_path}")


if __name__ == "__main__":
    main()
