# Read configuration paths and inspect the expanded hardware definition
from pathlib import Path
from xml.etree import ElementTree

# Use a narrow typing exception for the untyped launch package
from launch import LaunchDescription  # type: ignore[import-untyped]
from launch_ros.actions import Node
# Use a narrow typing exception for the untyped MoveIt builder
from moveit_configs_utils import MoveItConfigsBuilder  # type: ignore[import-untyped]


# Start a headless mock stack with one application motion owner
def generate_launch_description() -> LaunchDescription:
    # Load the existing model and planner settings as the shared configuration
    config = MoveItConfigsBuilder(
        "agrobot", package_name="agrobot_moveit_config"
    ).to_moveit_configs()
    # Read expanded robot text before checking the hardware plugin
    description = config.robot_description["robot_description"]
    # Reject unexpanded descriptions before parsing the hardware definition
    if not isinstance(description, str):
        raise TypeError("Expected expanded robot description text")
    # Inspect control plugins before any hardware process can start
    hardware = ElementTree.fromstring(description).findall("ros2_control/hardware/plugin")
    # Refuse launch unless exactly one supported mock plugin is configured
    if len(hardware) != 1 or hardware[0].text != "mock_components/GenericSystem":
        raise ValueError("This launch requires mock_components/GenericSystem")
    # Reuse the existing controller setup instead of copying joint mappings
    controllers_path = Path(config.package_path) / "config" / "ros2_controllers.yaml"
    # Launch the state publisher, controllers, planner, and motion owner
    return LaunchDescription(
        [
            # Publish link transforms and the shared robot description
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[config.robot_description],
                output="screen",
            ),
            # Run controllers with the validated mock description
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[str(controllers_path)],
                output="screen",
            ),
            # Activate feedback and the separate arm and rail controllers
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster", "arm_controller", "rail_controller"],
                output="screen",
            ),
            # Provide collision-aware planning from the existing MoveIt setup
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                parameters=[
                    config.to_dict(),
                    {
                        # Disable other execution paths to keep one motion owner
                        "allow_trajectory_execution": False,
                        "disable_capabilities": (
                            "move_group/MoveGroupExecuteTrajectoryAction "
                            "move_group/MoveGroupMoveAction "
                            "pilz_industrial_motion_planner/MoveGroupSequenceAction"
                        ),
                    },
                ],
                output="screen",
            ),
            # Reuse the shared model and limits for motion requests
            Node(
                package="agrobot_motion",
                executable="motion_node",
                parameters=[
                    config.robot_description,
                    config.robot_description_semantic,
                    config.joint_limits,
                ],
                output="screen",
            ),
        ]
    )
