from setuptools import find_packages, setup

package_name = "agrobot_motor_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Agrobot Team",
    maintainer_email="howardw1120@gmail.com",
    description="EPOS2 CANopen trajectory bridge for Agrobot",
    license="None",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "trajectory_to_canopen = agrobot_motor_bridge.trajectory_to_canopen:main",
            "plot_velocity_profile = agrobot_motor_bridge.plot_velocity_profile:main",
            "play_canopen_commands = agrobot_motor_bridge.play_canopen_commands:main",
        ],
    },
)
