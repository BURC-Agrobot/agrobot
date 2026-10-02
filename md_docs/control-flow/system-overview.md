# Control flow: Level 1, system

- source_pdf: `pdf_docs/control-flow/system-overview.pdf`
- workspace: current
- configuration: `agrobot_motion` / `mock.launch.py`; mock joint hardware
- verification: source-derived; not runtime-verified

## Semantics

- Edge `A -> B`: command, reply, or data path from A to B.
- Edge kind is the ROS mechanism: ACTION, SERVICE, TOPIC, or LOCAL.
- `LOCAL`: call inside `ros2_control_node`.
- `group: ros2_control`: node runs inside the ros2_control subsystem
  (`/controller_manager`).

## Nodes

- `external_motion_client` | External motion client | not started by this launch
- `motion_control` | `/agrobot_motion` | validate, plan, execute, confirm stop; one arm OR rail request at a time
- `motion_planner` | `/move_group` | trajectory execution disabled
- `robot_transforms` | `/robot_state_publisher`
- `controller_manager` | group: ros2_control | `/controller_manager`
- `trajectory_controllers` | group: ros2_control | `arm_controller`, `rail_controller`
- `mock_joint_hardware` | group: ros2_control | plugin: FakeSystem, GenericSystem
- `joint_state_publishing` | group: ros2_control | `joint_state_broadcaster`
- `integration_gaps` | context; no edges | no picking coordinator, gripper control, or real-drive interface in this workspace

## Edges

- `external_motion_client -> motion_control` | ACTION `/agrobot_motion/move_joints` | joint target / cancel
- `motion_control -> external_motion_client` | ACTION | feedback / outcome
- `motion_control -> controller_manager` | SERVICE | readiness: mock hardware + controllers
- `controller_manager -> motion_control` | SERVICE | reply
- `motion_control -> trajectory_controllers` | ACTION | trajectory / cancel
- `trajectory_controllers -> motion_control` | ACTION | acceptance / result
- `motion_control -> motion_planner` | SERVICE | plan joint motion
- `motion_planner -> motion_control` | SERVICE | trajectory or planning error
- `trajectory_controllers -> mock_joint_hardware` | LOCAL | position commands
- `mock_joint_hardware -> joint_state_publishing` | LOCAL | state read
- `joint_state_publishing -> motion_control` | TOPIC `/joint_states` | joint feedback
- `joint_state_publishing -> motion_planner` | TOPIC `/joint_states` | joint feedback
- `joint_state_publishing -> robot_transforms` | TOPIC `/joint_states` | joint feedback
