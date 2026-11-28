#!/usr/bin/env python3
"""Play back a generated EPOS2 CANopen commands file over Linux SocketCAN."""

from __future__ import annotations

import argparse
import re
import socket
import struct
import time
from pathlib import Path

from agrobot_motor_bridge.trajectory_to_canopen import (
    CW_DISABLE_VOLTAGE,
    CW_ENABLE_OPERATION,
    CW_NEW_SETPOINT_IMMEDIATE,
    CW_SHUTDOWN,
    CW_SWITCH_ON,
    IDX_CONTROLWORD,
    IDX_MODES_OF_OPERATION,
    IDX_PROFILE_ACCELERATION,
    IDX_PROFILE_DECELERATION,
    IDX_PROFILE_VELOCITY,
    IDX_TARGET_POSITION,
    MODE_PROFILE_POSITION,
    SDO_REQUEST_BASE_ID,
    encode_sdo_write,
)

# CANopen SDO response base ID, abort specifier, and status/position indices
SDO_RESPONSE_BASE_ID = 0x580
SDO_READ_CMD = 0x40
SDO_ABORT_CMD = 0x80
IDX_STATUSWORD = 0x6041
IDX_POSITION_ACTUAL = 0x6064
CW_FAULT_RESET = 0x0080


class SdoClient:
    """Minimal blocking CANopen SDO client over a Linux SocketCAN raw socket."""

    def __init__(self, interface: str, node_id: int, timeout_s: float = 1.0) -> None:
        self.node_id = node_id
        self.tx_id = SDO_REQUEST_BASE_ID + node_id
        self.rx_id = SDO_RESPONSE_BASE_ID + node_id
        self.timeout_s = timeout_s
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.bind((interface,))
        self.sock.settimeout(timeout_s)

    def close(self) -> None:
        """Close the underlying SocketCAN socket."""
        self.sock.close()

    def send_sdo(self, payload: bytes) -> bytes:
        """Send an 8-byte SDO frame and wait for the matching node reply."""
        frame = struct.pack("=IB3x8s", self.tx_id, 8, payload)
        self.sock.send(frame)
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            rx = self.sock.recv(16)
            can_id, _, data = struct.unpack("=IB3x8s", rx)
            if (can_id & 0x7FF) == self.rx_id:
                if data[0] == SDO_ABORT_CMD:
                    abort_code = int.from_bytes(data[4:8], "little")
                    raise RuntimeError(f"SDO abort 0x{abort_code:08X} for {payload.hex().upper()}")
                return data
        raise TimeoutError(f"No SDO reply on 0x{self.rx_id:03X} for {payload.hex().upper()}")

    def write(self, index: int, value: int, size_bytes: int, signed: bool = False) -> bytes:
        """Write an expedited SDO object at subindex 0."""
        _, payload = encode_sdo_write(self.node_id, index, 0, value, size_bytes, signed=signed)
        return self.send_sdo(payload)

    def read_u16(self, index: int) -> int:
        """Read a 16-bit unsigned SDO object at subindex 0."""
        data = self.send_sdo(bytes([SDO_READ_CMD]) + index.to_bytes(2, "little") + b"\x00\x00\x00\x00\x00")
        return int.from_bytes(data[4:6], "little")

    def read_i32(self, index: int) -> int:
        """Read a 32-bit signed SDO object at subindex 0."""
        data = self.send_sdo(bytes([SDO_READ_CMD]) + index.to_bytes(2, "little") + b"\x00\x00\x00\x00\x00")
        return int.from_bytes(data[4:8], "little", signed=True)


def load_command_file(commands_path: Path) -> tuple[int, int, int, list[tuple[float, int, int]]]:
    """Parse Node ID, ramp rates, and (t_s, counts, rpm) waypoints from a command file."""
    node_id = 24
    accel_rpm_s = 200
    decel_rpm_s = 200
    waypoints: list[tuple[float, int, int]] = []

    with commands_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m_node = re.search(r"Node ID:\s*(\d+)", line)
                if m_node:
                    node_id = int(m_node.group(1))
                m_acc = re.search(r"accel_rpm_s:\s*(\d+)", line)
                if m_acc:
                    accel_rpm_s = int(m_acc.group(1))
                m_dec = re.search(r"decel_rpm_s:\s*(\d+)", line)
                if m_dec:
                    decel_rpm_s = int(m_dec.group(1))
                continue
            parts = line.split()
            waypoints.append((float(parts[0]), int(parts[3]), int(parts[4])))

    if not waypoints:
        raise ValueError(f"No waypoints found in {commands_path}")
    return node_id, accel_rpm_s, decel_rpm_s, waypoints


def play_commands(
    commands_path: Path,
    interface: str,
    node_id_override: int | None = None,
    absolute: bool = False,
) -> None:
    """Enable the EPOS2, stream timed Profile Position waypoints, and disable voltage on exit."""
    file_node_id, accel_rpm_s, decel_rpm_s, waypoints = load_command_file(commands_path)
    node_id = node_id_override if node_id_override is not None else file_node_id

    client = SdoClient(interface, node_id)
    try:
        sw = client.read_u16(IDX_STATUSWORD)
        if sw & 0x0008:
            print(f"Drive in Fault (0x{sw:04X}); sending Fault Reset...")
            client.write(IDX_CONTROLWORD, CW_FAULT_RESET, 2)
            time.sleep(0.05)
            sw = client.read_u16(IDX_STATUSWORD)
            if sw & 0x0008:
                raise RuntimeError(f"Fault did not clear (Statusword=0x{sw:04X})")

        # Configure Profile Position Mode and enable operation
        client.write(IDX_MODES_OF_OPERATION, MODE_PROFILE_POSITION, 1, signed=True)
        client.write(IDX_PROFILE_ACCELERATION, accel_rpm_s, 4)
        client.write(IDX_PROFILE_DECELERATION, decel_rpm_s, 4)
        for cw in (CW_SHUTDOWN, CW_SWITCH_ON, CW_ENABLE_OPERATION):
            client.write(IDX_CONTROLWORD, cw, 2)
            time.sleep(0.02)

        sw = client.read_u16(IDX_STATUSWORD)
        if (sw & 0x006F) != 0x0027:
            raise RuntimeError(f"Drive did not reach Operation Enable (Statusword=0x{sw:04X})")

        start_qc = client.read_i32(IDX_POSITION_ACTUAL)
        offset_qc = 0 if absolute else (start_qc - waypoints[0][1])
        print(
            f"Node {node_id} on {interface}: Operation Enable (0x{sw:04X}), "
            f"start={start_qc} qc, playing {len(waypoints)} waypoints ({waypoints[-1][0]:.2f} s)..."
        )

        t0 = time.monotonic()
        last_print_s = -1.0
        for t_s, counts_qc, rpm in waypoints:
            delay_s = (t0 + t_s) - time.monotonic()
            if delay_s > 0:
                time.sleep(delay_s)
            target_qc = counts_qc + offset_qc
            client.write(IDX_PROFILE_VELOCITY, rpm, 4)
            client.write(IDX_TARGET_POSITION, target_qc, 4, signed=True)
            client.write(IDX_CONTROLWORD, CW_NEW_SETPOINT_IMMEDIATE, 2)
            client.write(IDX_CONTROLWORD, CW_ENABLE_OPERATION, 2)
            elapsed_s = time.monotonic() - t0
            if elapsed_s - last_print_s >= 4.0:
                actual_qc = client.read_i32(IDX_POSITION_ACTUAL)
                print(f"  t={elapsed_s:6.2f}s: target={target_qc:7d} qc, actual={actual_qc:7d} qc")
                last_print_s = elapsed_s

        time.sleep(0.3)
        final_qc = client.read_i32(IDX_POSITION_ACTUAL)
        final_sw = client.read_u16(IDX_STATUSWORD)
        print(f"Done: Position Actual={final_qc} qc (delta {final_qc - start_qc} qc), Statusword=0x{final_sw:04X}")
    finally:
        try:
            client.write(IDX_CONTROLWORD, CW_DISABLE_VOLTAGE, 2)
            print(f"Teardown: Disable Voltage sent (Statusword=0x{client.read_u16(IDX_STATUSWORD):04X})")
        except Exception as exc:
            print(f"Teardown warning: {exc}")
        client.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands_txt", type=Path, help="Path to generated CANopen commands text file")
    parser.add_argument("--interface", default="can0", help="SocketCAN interface name (default: can0)")
    parser.add_argument("--node-id", type=int, default=None, help="Override CANopen node ID from file header")
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Send raw file counts directly instead of offsetting from the motor's current position",
    )
    return parser.parse_args()


def main() -> None:
    """Run the SocketCAN CANopen trajectory player."""
    args = parse_args()
    play_commands(
        commands_path=args.commands_txt,
        interface=args.interface,
        node_id_override=args.node_id,
        absolute=args.absolute,
    )


if __name__ == "__main__":
    main()
