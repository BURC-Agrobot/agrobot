# Build and test Agrobot and prepare this shell to run it
# Use with: source build.sh
# Options are passed to colcon build: source build.sh --packages-select agrobot_motion
# Tests always cover the whole workspace

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
source "${AGROBOT_ROOT}/setup_env.sh" || return 1

# Run the tests, which can report success even when a test fails
(cd "${AGROBOT_ROOT}" && colcon test) || {
    echo "[agrobot] ERROR: Tests could not finish" >&2
    return 1
}

# Show the results and report any failed test
(cd "${AGROBOT_ROOT}" && colcon test-result --all) || {
    echo "[agrobot] ERROR: Tests failed" >&2
    return 1
}
