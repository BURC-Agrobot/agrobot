#!/bin/bash
# Prepare the shared ROS environment for both CV containers
# Serve the EVO-X2 AMD ROCm and x86-64 CPU/NVIDIA development images
# Keep device selection in the perception code and each image's runtime

# Stop startup if a required environment setup fails
set -e

# Default model discovery to the shared workspace and preserve explicit overrides
export AGROBOT_ROOT="${AGROBOT_ROOT:-/workspace}"

# Source ROS 2 Jazzy environment
source /opt/ros/jazzy/setup.bash

# Add the local perception packages if the workspace has been built
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
