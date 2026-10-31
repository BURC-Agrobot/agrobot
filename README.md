# Agrobot

Runs on WSL with Ubuntu 26.04 and ROS 2 Lyrical. For CAN work, see the
[motor bridge README](src/agrobot_motor_bridge/README.md).

## Install

Run once per machine.

```bash
curl -fL -o /tmp/ros2-apt-source.deb https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.3.0/ros2-apt-source_1.3.0.resolute_all.deb
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential clang-tidy mypy python3-pytest python3-colcon-common-extensions python3-numpy python3-opencv python3-venv ros-lyrical-ros-base ros-lyrical-moveit-ros-move-group ros-lyrical-moveit-kinematics ros-lyrical-moveit-planners ros-lyrical-moveit-simple-controller-manager ros-lyrical-moveit-configs-utils ros-lyrical-controller-manager ros-lyrical-joint-state-broadcaster ros-lyrical-joint-trajectory-controller ros-lyrical-joint-state-publisher ros-lyrical-robot-state-publisher ros-lyrical-xacro ros-lyrical-cv-bridge ros-lyrical-image-transport ros-lyrical-vision-msgs ros-lyrical-sensor-msgs-py ros-lyrical-ament-copyright ros-lyrical-ament-flake8 ros-lyrical-ament-pep257 ros-lyrical-demo-nodes-cpp ros-lyrical-demo-nodes-py ros-lyrical-rviz2 ros-lyrical-moveit-ros-visualization ros-lyrical-joint-state-publisher-gui ros-lyrical-ros2controlcli
python3 -m venv --system-site-packages ~/.venvs/agrobot
~/.venvs/agrobot/bin/python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu
~/.venvs/agrobot/bin/python -m pip check
git clone https://github.com/BURC-Agrobot/agrobot.git
cd agrobot
```

Run all later commands from this folder.

## Build and test

Build, run the workspace tests, and load the environment:

```bash
source build.sh
```

Build only some packages (tests still cover the whole workspace):

```bash
source build.sh --packages-select agrobot_motion
```

Load the environment in a new terminal:

```bash
source setup_env.sh
```

Rerun the tests without rebuilding:

```bash
colcon test
colcon test-result --all
```

## Run

```bash
ros2 launch agrobot_motion robot.launch.py hardware:=mock
```

`hardware` is `mock` (default), `emulator` or `real`. Only `mock` works so far.
The other two need the motor bridge plugin, which is not built yet.

Stop with Ctrl-C. Ctrl-Z only suspends: run `fg`, then press Ctrl-C.

In a second terminal, check that all three controllers are `active` and joint
states are published:

```bash
ros2 control list_controllers
ros2 topic echo --once /joint_states
```

Open RViz:

```bash
ros2 launch agrobot_moveit_config moveit_rviz.launch.py
```

## Send joint commands

Move the arm, with `J1` to `J6` in radians:

```bash
ros2 action send_goal /agrobot_motion/move_joints agrobot_motion/action/MoveJoints "{group: 0, arm_positions_rad: [J1, J2, J3, J4, J5, J6]}"
```

Move the rail, here to 0.2 m:

```bash
ros2 action send_goal /agrobot_motion/move_joints agrobot_motion/action/MoveJoints "{group: 1, rail_position_m: 0.2}"
```

Show positions (arm joints 1 to 6, then rail):

```bash
ros2 topic echo --once /joint_states --field position
```

Add `--feedback` to see each phase. Success ends with `code: 0` and
`Goal finished with status: SUCCEEDED`. On `Goal was rejected.`, the launch
terminal gives the reason.

Rules:

- Move the arm or the rail, not both.
- Give all 6 arm joints. Targets are absolute, and an omitted joint means 0.
- Wait for the current command to finish. A new one is rejected, not queued.
- Stay within the limits:

| Joint | Lower | Upper | Unit |
| --- | --- | --- | --- |
| `arm_joint_1` | -3.14 | 3.14 | rad |
| `arm_joint_2` | -1.57 | 1.57 | rad |
| `arm_joint_3` | -4.5 | 1.5 | rad |
| `arm_joint_4` | -3.14 | 3.14 | rad |
| `arm_joint_5` | -5 | 5 | rad |
| `arm_joint_6` | 0 | 2.09 | rad |
| `rail_joint` | 0 | 1.565 | m |

## Known problems

- `move_group` segfaults when the mock stack stops.
- `RTPS_TRANSPORT_SHM Error` and `A message was lost!!!` are harmless if the
  command still returns data.
- RViz cannot execute plans in mock mode. Use `move_joints`.
- Tools and nodes that need `AGROBOT_ROOT` get it from `setup_env.sh`.
- Perception launch files are not covered. They need model files that are not in
  the repository.
