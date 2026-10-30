#!/bin/bash
# Prepare ROS and the workspace overlay for this Linux development machine

# Stop startup if a required environment setup fails
set -e

# Source ROS 2 Lyrical environment
source /opt/ros/lyrical/setup.bash

# Add all local packages if the workspace build exists
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
    echo "[agrobot] Sourced local ROS 2 workspace overlay."
fi

# Report the prepared environment without claiming the CV models are loaded
echo "[agrobot] AGROBOT_ROOT=${AGROBOT_ROOT}"
echo "[agrobot] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[agrobot] ROS_DISTRO=${ROS_DISTRO}"
echo "[agrobot] ROS environment ready."

# Replace the startup shell so the user's command receives container signals
exec "$@"
