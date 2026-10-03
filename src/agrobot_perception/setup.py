from setuptools import find_packages, setup

# setup.py is required by ament_python (ROS 2's Python build type).
# It tells colcon how to install this package into the ROS 2 workspace overlay.
# Bazel does NOT use this file — Bazel uses BUILD.bazel instead.

package_name = "agrobot_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # Register the package with the ROS 2 ament index.
        # Without this, `ros2 pkg list` won't find it.
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # Install launch files so `ros2 launch agrobot_perception` works.
        (f"share/{package_name}/launch", ["launch/perception.launch.py", "launch/perception_gpu.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Agrobot Team",
    maintainer_email="howardw1120@gmail.com",
    description="Agrobot TOM v2 perception pipeline",
    license="None",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Format: "executable_name = package.module:main_function"
            "tomato_detector = agrobot_perception.tomato_detector_node:main",
            "tomato_spatial = agrobot_perception.tomato_spatial_node:main",
            "tomato_tracker = agrobot_perception.tomato_tracker_node:main",
            "qwen_vl = agrobot_perception.qwen_vl_node:main",
        ],
    },
)
