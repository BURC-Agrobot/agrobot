# Agrobot motor bridge

C++ code that talks to EPOS2 motor drives over CAN, and `drive_tool`, a command
line tool for trying it by hand. It currently runs against
[epos2-emulator](https://github.com/BURC-Agrobot/epos2-emulator), a fake drive on
`vcan0`, a virtual CAN bus, so no hardware is needed.

The bridge does not talk to ROS or run on a real drive yet. To run the robot in
RViz with the mock controller, see the [main README](../../README.md#run), which
does not use this package.

Run every command from the repository root.

## Install

Run once per machine, after the [main install](../../README.md#install).

```bash
sudo apt-get install -y can-utils
git clone https://github.com/BURC-Agrobot/epos2-emulator.git ../epos2-emulator
python3 -m venv ../epos2-emulator/.venv
../epos2-emulator/.venv/bin/pip install -r ../epos2-emulator/requirements.txt
```

## Start

You need three terminals.

**Any terminal:** create the bus. Do this after every reboot or WSL restart.
`File exists` means it is already there.

```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
```

**Terminal 1:** watch every frame on the bus, and leave this running.

```bash
candump vcan0
```

**Terminal 2:** run the fake drive, and leave this running. Run only one per bus,
so first check that `pgrep -af emulator.py` prints nothing.

```bash
cd ../epos2-emulator && .venv/bin/python emulator.py
```

**Terminal 3:** build, test and load the environment. Run `drive_tool` commands
here.

```bash
source build.sh --packages-select agrobot_motor_bridge
```

In a later session, `source setup_env.sh` loads the environment without building.

To check that everything is connected:

```bash
ros2 run agrobot_motor_bridge drive_tool read-type
```

```
[drive_tool] Node 1 Device Type = 0x00020192
```

## Commands

```
ros2 run agrobot_motor_bridge drive_tool <command> [interface] [node] [target]
```

| Command | What it does | Needs the emulator | Powers the motor |
| --- | --- | --- | --- |
| `send` | Sends one Statusword read request to node 1 | For a reply | No |
| `receive` | Prints the first frame on the bus within 5 s | No | No |
| `read-type` | Reads the drive's Device Type | Yes | No |
| `check-write` | Tests that the drive stores a write and refuses a bad one | Yes | No |
| `walk` | Steps the drive through its power states and back | Yes | Yes |
| `move` | Moves the motor to a target and powers off | Yes | Yes |

| Argument | Default | Meaning |
| --- | --- | --- |
| `interface` | `vcan0` | CAN interface to use |
| `node` | `1` | Drive's node ID, 1 to 127. Ignored by `send` and `receive` |
| `target` | `20000` | `move` only: absolute position in quadcounts, and may be negative |

Arguments are by position, so setting a later one means giving the earlier ones:

```bash
ros2 run agrobot_motor_bridge drive_tool move vcan0 1 -5000
```

A command exits with 0 on success. On failure it prints a line starting
`[drive_tool] ERROR:` and exits with 1, and `ros2 run` adds a
`Process exited with failure 1` line.

`walk` and `move` only run on `vcan` interfaces, because they power the motor with
no way to stop it. The emulator only answers as node 1 for now, so other nodes time
out.

### send

```bash
ros2 run agrobot_motor_bridge drive_tool send
```

```
[drive_tool] Sent 601#4041600000000000 on vcan0
```

`601#4041600000000000` is the CAN ID, then the data bytes. The tool does not wait
for a reply, so look in terminal 1:

```
vcan0  601   [8]  40 41 60 00 00 00 00 00
vcan0  581   [8]  4B 41 60 00 40 01 00 00
```

See [Read candump](#read-candump) for the bytes. The reply's value is `0x0140`,
which means [Switch On Disabled](#read-a-statusword).

### receive

Wait up to 5 seconds for a frame, whatever its ID:

```bash
ros2 run agrobot_motor_bridge drive_tool receive
```

Then send one from another terminal:

```bash
cansend vcan0 123#DEADBEEF
```

```
[drive_tool] Waiting 5 s on vcan0
[drive_tool] 123  [4]  DE AD BE EF
```

The output shows the CAN ID, the number of data bytes in brackets, and the bytes.
With no frame, it fails with `Timed out with no frame`.

### read-type

```bash
ros2 run agrobot_motor_bridge drive_tool read-type
```

```
[drive_tool] Node 1 Device Type = 0x00020192
```

Device Type (`0x1000`) is the value every CANopen device must have, which makes this
the simplest check that a drive is answering. In `0x00020192`, the low half `0x0192`
is 402 in decimal, meaning the drive follows the CiA 402 motor drive standard.

### check-write

```bash
ros2 run agrobot_motor_bridge drive_tool check-write
```

```
[drive_tool] Controlword = 0x0006
[drive_tool] Statusword = 0x0121
[drive_tool] Refused as expected: SDO write of 0x6041 sub 0x00 aborted with 0x06010002 (object is read-only)
```

| Line | Meaning |
| --- | --- |
| `Controlword = 0x0006` | The Shutdown command was written and read back, so writes are stored |
| `Statusword = 0x0121` | The drive acted on it and is now Ready to Switch On |
| `Refused as expected` | Writing the read-only Statusword was rejected, as it should be |

The refusal line is the test passing, not an error. The command then puts the drive
back in Switch On Disabled.

### walk

```bash
ros2 run agrobot_motor_bridge drive_tool walk
```

```
[drive_tool] 0x0140 Switch On Disabled
[drive_tool] 0x0121 Ready to Switch On
[drive_tool] 0x0123 Switched On
[drive_tool] 0x0537 Operation Enable
[drive_tool] 0x0140 Switch On Disabled
```

Each line is the Statusword and the state it means, read after each step. A drive
must go through these states in order, and it has power on the motor only in
Operation Enable. The last line is the drive powered off again.

### move

```bash
ros2 run agrobot_motor_bridge drive_tool move
```

```
[drive_tool] 0x0140 Switch On Disabled
[drive_tool] 0x0537 Operation Enable
[drive_tool] Moving to 20000
[drive_tool] Position = 20000
[drive_tool] 0x0140 Switch On Disabled
```

| Line | Meaning |
| --- | --- |
| `Switch On Disabled` | The drive started clean with no power |
| `Operation Enable` | Profile Position Mode is set, and the motor has power |
| `Moving to 20000` | The target was sent, and the tool is waiting for the move to finish |
| `Position = 20000` | The move finished, and this is the position read back from the drive |
| `Switch On Disabled` | Power is off again |

The target is absolute, so moving to 20000 twice only moves once. The tool fails if
the move takes over 10 seconds or stops away from the target, and tries to power off
the motor on any failure after the clean start check.

## Read the output

### Read a Statusword

The Statusword (`0x6041`) is a 16-bit value the drive uses to report its state.

| Value | State | Motor power |
| --- | --- | --- |
| `0x0140` | Switch On Disabled, where the drive starts | No |
| `0x0121` | Ready to Switch On | No |
| `0x0123` | Switched On | No |
| `0x0537` | Operation Enable, with Target Reached set | Yes |

Other bits can change the value without changing the state. For example, `0x0537` is
Operation Enable (`0x0137`) plus bit 10, Target Reached.

To see every set bit, type `status` in terminal 2:

```
State: Switch On Disabled
Statusword: 0x0140 0b0000_0001_0100_0000
Bits set:
   8 Offset current measured
   6 Switch on disable
```

### Read candump

Each line in terminal 1 is one frame:

```
vcan0  581   [8]  4B 41 60 00 40 01 00 00
│      │     │    └ data bytes
│      │     └ number of data bytes
│      └ CAN ID
└ interface
```

The CAN ID says who a frame is for. Requests go to `0x600 + node` and replies come
from `0x580 + node`, so node 1 uses `601` and `581`.

These frames are SDOs, the CANopen way to read and write a drive's values:

| Bytes | Meaning |
| --- | --- |
| 0 | Command, from the table below |
| 1 and 2 | Value's index, low byte first: `41 60` is `0x6041` |
| 3 | Value's subindex |
| 4 to 7 | The value, low byte first: `40 01 00 00` is `0x0140` |

| Byte 0 | Direction | Meaning |
| --- | --- | --- |
| `40` | To the drive | Read this value |
| `4F`, `4B`, `43` | From the drive | Here is the value, in 1, 2 or 4 bytes |
| `2F`, `2B`, `23` | To the drive | Write this value, in 1, 2 or 4 bytes |
| `60` | From the drive | Write accepted |
| `80` | From the drive | Refused, with the abort code in bytes 4 to 7 |

### Read the emulator's log

Terminal 2 explains each frame it receives:

```
RX id=0x601 data=2B 40 60 00 06 00 00 00
SDO write index=0x6040 sub=0x00 value=0x6 size_bytes=2
Controlword 0x0006: Shutdown, Switch On Disabled -> Ready to Switch On
SDO write stored, sending ok on 0x581
```

Type these in terminal 2 at any time:

| Input | Shows |
| --- | --- |
| `status` | The drive state and every set Statusword bit |
| `reg 0x6064` | One value, here the position. Add `-b` for binary |
| `regs` | All values |
| `--help` | What each log line means |

## Stop

Press Ctrl-C in terminals 1 and 2. Ctrl-Z only suspends a program, and a suspended
emulator answers again if resumed.

## Troubleshooting

| Message | Cause | Fix |
| --- | --- | --- |
| `if_nametoindex(vcan0) failed: No such device` | The bus does not exist | [Create it](#start) |
| `SDO read of 0x... timed out` | Nothing answered within 500 ms | Start the emulator, and check the node ID |
| `This command only runs on virtual buses` | `walk` or `move` was given a real interface | Use a `vcan` interface |
| `Drive must start in Switch On Disabled, but is in ...` | An earlier run left the drive in another state | Restart the emulator |
| `Node ID must be 1 to 127` | The node argument is out of range | Use a valid node ID |
| `Usage: drive_tool ...` | The command is missing or misspelled | Use a name from [Commands](#commands) |
| Several replies to one request in `candump` | More than one emulator is running | Run `pkill -9 -f emulator.py`, then restart one |
