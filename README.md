# Agrobot

These instructions build and run the workspace natively on WSL with
Ubuntu 26.04 and ROS 2 Lyrical. They cover mock mode, not hardware.

## Install

Run this section once per machine. The `sudo` commands ask for your password.

Download the package that adds the ROS repository to apt:

```bash
curl -fL -o /tmp/ros2-apt-source.deb https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.3.0/ros2-apt-source_1.3.0.resolute_all.deb
```

Install that package:

```bash
sudo dpkg -i /tmp/ros2-apt-source.deb
```

Refresh the package lists:

```bash
sudo apt-get update
```

Install ROS 2 and the build tools:

```bash
sudo apt-get install -y --no-install-recommends build-essential clang-tidy mypy python3-pytest python3-colcon-common-extensions python3-numpy python3-opencv python3-venv ros-lyrical-ros-base ros-lyrical-moveit-ros-move-group ros-lyrical-moveit-kinematics ros-lyrical-moveit-planners ros-lyrical-moveit-simple-controller-manager ros-lyrical-moveit-configs-utils ros-lyrical-controller-manager ros-lyrical-joint-state-broadcaster ros-lyrical-joint-trajectory-controller ros-lyrical-joint-state-publisher ros-lyrical-robot-state-publisher ros-lyrical-xacro ros-lyrical-cv-bridge ros-lyrical-image-transport ros-lyrical-vision-msgs ros-lyrical-sensor-msgs-py ros-lyrical-ament-copyright ros-lyrical-ament-flake8 ros-lyrical-ament-pep257 ros-lyrical-demo-nodes-cpp ros-lyrical-demo-nodes-py ros-lyrical-rviz2 ros-lyrical-moveit-ros-visualization ros-lyrical-joint-state-publisher-gui ros-lyrical-ros2controlcli
```

Create the Python environment, which can also use the ROS Python packages:

```bash
python3 -m venv --system-site-packages ~/.venvs/agrobot
```

Install CPU PyTorch into that environment:

```bash
~/.venvs/agrobot/bin/python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu
```

Check that the Python packages are compatible:

```bash
~/.venvs/agrobot/bin/python -m pip check
```

## Get the code

Clone the repository:

```bash
git clone https://github.com/BURC-Agrobot/agrobot.git
```

Enter the repository. The commands below all run from this folder:

```bash
cd agrobot
```

## Build

Build all packages, run the tests, and load the environment into this terminal:

```bash
source build.sh
```

To build only some packages, add `colcon build` options. The tests still cover
the whole workspace:

```bash
source build.sh --packages-select agrobot_motion
```

In every new terminal, load the environment without rebuilding:

```bash
source setup_env.sh
```

## Test

`build.sh` already runs the tests. To run them again without rebuilding:

```bash
colcon test
```

Show the test results:

```bash
colcon test-result --all
```

## Run in mock mode

Start the mock robot in one terminal. Press Ctrl-C to stop it:

```bash
ros2 launch agrobot_motion mock.launch.py
```

Ctrl-Z only suspends a command. To stop it, run `fg` and press Ctrl-C.

Open a second terminal in the repository and load the environment:

```bash
source setup_env.sh
```

Check that all three controllers are `active`:

```bash
ros2 control list_controllers
```

Check that the robot publishes joint states:

```bash
ros2 topic echo --once /joint_states
```

Open RViz through WSLg:

```bash
ros2 launch agrobot_moveit_config moveit_rviz.launch.py
```

## Send joint commands

Keep the mock robot running, and send commands from a second terminal with the
environment loaded.

Move the arm by replacing `J1` to `J6` with target positions in radians:

```bash
ros2 action send_goal /agrobot_motion/move_joints agrobot_motion/action/MoveJoints "{group: 0, arm_positions_rad: [J1, J2, J3, J4, J5, J6]}"
```

Move the rail by replacing `POSITION` with a target position in metres:

```bash
ros2 action send_goal /agrobot_motion/move_joints agrobot_motion/action/MoveJoints "{group: 1, rail_position_m: POSITION}"
```

For example, this moves the rail to 0.2 m:

```bash
ros2 action send_goal /agrobot_motion/move_joints agrobot_motion/action/MoveJoints "{group: 1, rail_position_m: 0.2}"
```

Add `--feedback` to either command to see each phase of the motion. A successful
command ends with `code: 0` and `Goal finished with status: SUCCEEDED`.

Show the current positions, listed as arm joints 1 to 6 and then the rail:

```bash
ros2 topic echo --once /joint_states --field position
```

Each command must follow these rules:

- Move either the arm or the rail. Leave the other group's field unset or zero.
- Give all 6 arm joints, because targets are absolute and an omitted joint
  defaults to 0.
- Wait for the current command to finish, because a new one is rejected rather
  than queued.
- Stay within the joint limits:

| Joint | Lower limit | Upper limit | Unit |
| --- | --- | --- | --- |
| `arm_joint_1` | -3.14 | 3.14 | rad |
| `arm_joint_2` | -1.57 | 1.57 | rad |
| `arm_joint_3` | -4.5 | 1.5 | rad |
| `arm_joint_4` | -3.14 | 3.14 | rad |
| `arm_joint_5` | -5 | 5 | rad |
| `arm_joint_6` | 0 | 2.09 | rad |
| `rail_joint` | 0 | 1.565 | m |

If a command prints `Goal was rejected.`, the first terminal's log gives the
reason.

## Known problems

- `move_group` segfaults when the mock stack stops.
- ROS commands can print `RTPS_TRANSPORT_SHM Error` or `A message was lost!!!`.
  Both are harmless if the command still returns data.
- RViz cannot execute plans in mock mode, because `mock.launch.py` disables
  `move_group` execution. Use `move_joints` to move the robot.
- `AGROBOT_ROOT` must be set before running tools or nodes that need it.
  `setup_env.sh` sets it.
- This README does not cover perception launch files, which need model files
  that are not in the repository.
