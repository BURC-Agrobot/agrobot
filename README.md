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

Load the environment. Do this in every new terminal:

```bash
source setup_env.sh
```

Build all packages:

```bash
colcon build
```

Load the environment again so the shell can find the packages you just built:

```bash
source setup_env.sh
```

## Test

Run the tests:

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

## Known problems

- `move_group` segfaults when the mock stack stops.
- MoveIt planning from RViz does not work yet, because the plugin does not
  receive `robot_description_semantic`.
- `AGROBOT_ROOT` must be set before running tools or nodes that need it.
  `setup_env.sh` sets it.
- This README does not cover perception launch files, which need model files
  that are not in the repository.
