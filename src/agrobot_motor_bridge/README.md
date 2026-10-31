# Agrobot motor bridge

Test programs that drive a fake EPOS2,
[epos2-emulator](https://github.com/BURC-Agrobot/epos2-emulator), on `vcan0`, a
virtual CAN bus. No hardware is needed. The bridge does not talk to ROS or run on
a real drive yet.

Start with the repository root as your working directory in every terminal.

## Install

Run once per machine, after the [main install](../../README.md#install).

```bash
sudo apt-get install -y can-utils
git clone https://github.com/BURC-Agrobot/epos2-emulator.git ../epos2-emulator
python3 -m venv ../epos2-emulator/.venv
../epos2-emulator/.venv/bin/pip install -r ../epos2-emulator/requirements.txt
```

## Run with the emulator

### 1. Create vcan0

Run after every reboot or WSL restart. `File exists` means it is already there.

```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
```

### 2. Build

```bash
source build.sh --packages-select agrobot_motor_bridge
```

### 3. Watch the bus

In **terminal 1**, leave this running to see every frame:

```bash
candump vcan0
```

### 4. Start the emulator

Run only one emulator per bus. Check that this prints nothing:

```bash
pgrep -af emulator.py
```

In **terminal 2**, leave this running:

```bash
cd ../epos2-emulator && .venv/bin/python emulator.py
```

### 5. Load the environment

In **terminal 3**, run this. Run the remaining steps here too:

```bash
source setup_env.sh
```

### 6. Send a frame

Send a Statusword read request:

```bash
ros2 run agrobot_motor_bridge send_frame
```

Terminal 1 shows the request and the emulator's reply, Statusword `0x0140`:

```
vcan0  601   [8]  40 41 60 00 00 00 00 00
vcan0  581   [8]  4B 41 60 00 40 01 00 00
```

### 7. Read a value

```bash
ros2 run agrobot_motor_bridge read_device_type
```

```
[read_device_type] Node 1 Device Type = 0x00020192
```

### 8. Write a value

Write Controlword, read it back, and try to write read-only Statusword. The
refusal is expected:

```bash
ros2 run agrobot_motor_bridge check_sdo_write
```

```
[check_sdo_write] Controlword = 0x0006
[check_sdo_write] Statusword = 0x0121
[check_sdo_write] Refused as expected: SDO write of 0x6041 sub 0x00 aborted with 0x06010002 (object is read-only)
```

### 9. Walk the drive states

Go up to Operation Enable and back:

```bash
ros2 run agrobot_motor_bridge walk_drive_states
```

```
[walk_drive_states] 0x0140 Switch On Disabled
[walk_drive_states] 0x0121 Ready to Switch On
[walk_drive_states] 0x0123 Switched On
[walk_drive_states] 0x0537 Operation Enable
[walk_drive_states] 0x0140 Switch On Disabled
```

### 10. Move the motor

Enable the drive, move to 20000 quadcounts, and disable it:

```bash
ros2 run agrobot_motor_bridge move_drive
```

```
[move_drive] 0x0140 Switch On Disabled
[move_drive] 0x0537 Operation Enable
[move_drive] Moving to 20000
[move_drive] Position = 20000
[move_drive] 0x0140 Switch On Disabled
```

To choose the target, also name the interface and node:

```bash
ros2 run agrobot_motor_bridge move_drive vcan0 1 -5000
```

### 11. Stop

Press Ctrl-C in terminals 1 and 2. Ctrl-Z only suspends a program, and a suspended
emulator answers again if resumed.

## Run with the mock controller

The mock controller runs the robot in software for RViz. It does not use the
bridge, CAN or the emulator.

In one terminal, start the robot and leave it running:

```bash
source setup_env.sh
ros2 launch agrobot_motion robot.launch.py hardware:=mock
```

In a second terminal, open RViz:

```bash
source setup_env.sh
ros2 launch agrobot_moveit_config moveit_rviz.launch.py
```

To send joint commands, see the [main README](../../README.md#send-joint-commands).

## Run with the real controller

## Receive a frame

This works without the emulator. Wait up to 5 seconds for one frame:

```bash
ros2 run agrobot_motor_bridge receive_frame
```

Send one from another terminal:

```bash
cansend vcan0 123#DEADBEEF
```

```
[receive_frame] 123  [4]  DE AD BE EF
```

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No such device` | Create the interface. See [step 1](#1-create-vcan0). |
| `timed out` | Start the emulator, and check the node ID. |
| Several replies to one request | Other emulators are running. Stop them all with `pkill -9 -f emulator.py`, then restart one. |
| `Drive must start in Switch On Disabled` | Restart the emulator. |

## Notes

- All programs take an interface name, and all but `send_frame` and
  `receive_frame` also take a node ID, such as `read_device_type vcan0 24`.
- `send_frame` always addresses node 1.
- `receive_frame` returns the first frame on the bus, whatever its ID.
- `walk_drive_states` and `move_drive` only run on `vcan` interfaces, because they
  power the motor with no way to stop it.
