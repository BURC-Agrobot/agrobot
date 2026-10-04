# Control flow: Level 1, system

- source_pdf: `pdf_docs/control-flow/system-overview.pdf`
- workspace: current
- configuration: `agrobot_motion` / `mock.launch.py` with mock joint hardware.
- verification: Source inspection establishes these relationships. These relationships have no runtime verification.

## Semantics

- Edge `A -> B`: The edge shows a command, reply, or data path from A to B.
- The edge kind identifies the ROS mechanism: ACTION, SERVICE, TOPIC, or LOCAL.
- `LOCAL`: The call occurs inside `ros2_control_node`.
- `group: ros2_control`: The node runs inside the ros2_control subsystem (`/controller_manager`).

## Nodes

- `external_motion_client` | External motion client | This launch does not start the client.
- `motion_control` | `/agrobot_motion` | It validates, plans, executes, and confirms stopping. It handles one arm OR rail request at a time.
- `motion_planner` | `/move_group` | Trajectory execution is disabled.
- `robot_transforms` | `/robot_state_publisher`
- `controller_manager` | group: ros2_control | `/controller_manager`
- `trajectory_controllers` | group: ros2_control | `arm_controller`, `rail_controller`
- `mock_joint_hardware` | group: ros2_control | plugin: FakeSystem, GenericSystem
- `joint_state_publishing` | group: ros2_control | `joint_state_broadcaster`
- `integration_gaps` | context | no edges | This workspace has no picking coordinator, gripper control, or interface to real drives.

## Edges

- `external_motion_client -> motion_control` | ACTION `/agrobot_motion/move_joints` | Send a joint target or cancellation request.
- `motion_control -> external_motion_client` | ACTION | Return feedback or the outcome.
- `motion_control -> controller_manager` | SERVICE | Check whether the mock hardware and controllers are ready.
- `controller_manager -> motion_control` | SERVICE | Return a reply.
- `motion_control -> trajectory_controllers` | ACTION | Send a trajectory or cancellation request.
- `trajectory_controllers -> motion_control` | ACTION | Return acceptance or the result.
- `motion_control -> motion_planner` | SERVICE | Request a plan for joint motion.
- `motion_planner -> motion_control` | SERVICE | Return a trajectory or planning error.
- `trajectory_controllers -> mock_joint_hardware` | LOCAL | Send position commands.
- `mock_joint_hardware -> joint_state_publishing` | LOCAL | Read the state.
- `joint_state_publishing -> motion_control` | TOPIC `/joint_states` | Send joint feedback.
- `joint_state_publishing -> motion_planner` | TOPIC `/joint_states` | Send joint feedback.
- `joint_state_publishing -> robot_transforms` | TOPIC `/joint_states` | Send joint feedback.
