# Build Agrobot and prepare this shell to run it
# Use with: source build.sh
# Options are passed to colcon: source build.sh --packages-select agrobot_motion

# Stop with a clear message because a child process cannot change this shell
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "[agrobot] ERROR: Run this with: source build.sh" >&2
    exit 1
fi

# Load the environment that the build needs
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh" || return 1

# Build from the repository root so output stays in one place
(cd "${AGROBOT_ROOT}" && colcon build "$@") || {
    echo "[agrobot] ERROR: Build failed" >&2
    return 1
}

# Load the environment again so the shell can find the built packages
source "${AGROBOT_ROOT}/setup_env.sh"
