# Prepare a shell to build and run Agrobot on WSL with ROS 2 Lyrical
# Use with: source setup_env.sh

# Locate the repository so the script works from any directory
AGROBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AGROBOT_ROOT

# Stop setup with a clear message if the ROS installation is missing
if [ ! -f /opt/ros/lyrical/setup.bash ]; then
    echo "[agrobot] ERROR: ROS 2 Lyrical is not installed in /opt/ros/lyrical" >&2
    return 1
fi

# Stop setup with a clear message if the Python environment is missing
if [ ! -d "${HOME}/.venvs/agrobot" ]; then
    echo "[agrobot] ERROR: Python environment ~/.venvs/agrobot is missing" >&2
    return 1
fi

# Load the ROS 2 Lyrical environment
source /opt/ros/lyrical/setup.bash

# Expose PyTorch to interactive Python and system-Python ROS executables
export PATH="${HOME}/.venvs/agrobot/bin:${PATH}"
export PYTHONPATH="${HOME}/.venvs/agrobot/lib/python3.14/site-packages:${PYTHONPATH}"

# Add the local packages once the workspace has been built
if [ -f "${AGROBOT_ROOT}/install/setup.bash" ]; then
    source "${AGROBOT_ROOT}/install/setup.bash"
fi

# Limit ROS discovery to processes on this machine
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# Keep Python from writing cache files into the source tree
export PYTHONDONTWRITEBYTECODE=1

# Open Qt windows through X11 because RViz cannot render on Wayland
export QT_QPA_PLATFORM=xcb

# Render RViz without requiring GPU access
export LIBGL_ALWAYS_SOFTWARE=1

# Report the prepared environment
echo "[agrobot] ROS_DISTRO=${ROS_DISTRO} AGROBOT_ROOT=${AGROBOT_ROOT}"
