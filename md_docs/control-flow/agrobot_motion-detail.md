# Control flow: package/node detail, agrobot_motion

- source_pdf: `pdf_docs/control-flow/agrobot_motion-detail.pdf`
- workspace: current
- configuration: `mock.launch.py`. The node handles one arm OR rail request at a time.
- verification: Source inspection establishes these relationships. These relationships have no runtime verification.
- node: `/agrobot_motion` | executable: `motion_node`
- executor: SingleThreadedExecutor runs callbacks and shutdown one at a time.

## Semantics

- Edge `A -> B`: The edge shows a call or message path.
  Its kind is ACTION, SERVICE, TOPIC, LOCAL, or DATA.
- `LOCAL`: The call occurs inside `/agrobot_motion`.
- `DATA`: The edge represents stored data or context, not a call.
- Asynchronous edges do not specify callback order.
- Uppercase node names (CHECKING, PLANNING, EXECUTING, STOPPING) identify MotionState phases.
- `inside: yes`: The node belongs to `/agrobot_motion`.
- `inside: no`: The node represents an external peer or context.

## Nodes

- `requesting_client_external` | inside: no | Requesting client (external)
  - This launch does not start the client.
  - An ARM goal contains six angles [rad].
  - A RAIL goal contains one position [m].
- `requesting_client_peer` | inside: no | Requesting client (same peer as `requesting_client_external`)
  - Feedback fields: state, elapsed_s.
  - Result fields: code, message, stop_confirmed.
  - The terminal status is succeeded, canceled, or aborted.
- `joint_feedback` | inside: no | context | Joint feedback source
  - `joint_state_broadcaster` supplies feedback. ros2_control runs at 100 Hz.
  - Hardware: FakeSystem, GenericSystem.
  - Stop evidence applies only to mock hardware.
- `controller_manager` | inside: no | context | `/controller_manager`
  - Services: `list_hardware_components`, `list_controllers` (`controller_manager_msgs/srv`).
- `planning` | inside: no | context | `/move_group` with execution disabled
  - Service: `/plan_kinematic_path` (`moveit_msgs/srv/GetMotionPlan`).
- `controllers` | inside: no | context | `/arm_controller` OR `/rail_controller`
  - Action: `.../follow_joint_trajectory` (`control_msgs/action/FollowJointTrajectory`).
  - The controllers command mock hardware.
- `configuration` | inside: no | context | Configuration
  - The configuration contains URDF, SRDF, and YAML acceleration limits.
  - `mock.launch.py` supplies these parameters.
  - If the model is invalid, the node exits at startup.
- `goal_cancel_callbacks` | inside: yes | Goal and cancel callbacks
  - If idle with a valid goal, enter CHECKING. Call begin().
  - Otherwise, reject the goal. Log the reason.
  - To cancel the current goal, call stop().
- `receive_joints` | inside: yes | receive_joints()
  - Check all 7 joints for finite values.
  - Ignore old or repeated timestamps.
  - For a bad sample, mark feedback stale. Limit the warning frequency.
  - Track stillness for stop checks.
- `stored_feedback` | inside: yes | data | Stored feedback
  - Feedback is fresh if it arrived < 0.5 s ago.
  - Every phase check reads this feedback.
- `tick` | inside: yes | 20 ms timer calls tick()
  - Publish feedback. Enforce deadlines.
  - CHECKING: Call preflight(). Then call plan().
  - PLANNING: Wait for the planner reply.
  - EXECUTING: Require fresh feedback. Require the other group to remain still.
  - STOPPING: Confirm stopping or reach the deadline.
- `shutdown` | inside: yes | Shutdown on SIGINT / SIGTERM
  - Block new goals.
  - If a goal is active, call stop().
  - Exit after the active goal's result.
- `CHECKING` | inside: yes | phase | preflight() and replies
  - Wait for services, both actions, and a fresh pose.
  - Require active mock hardware and both controllers.
  - Return UNAVAILABLE if the 3 s deadline expires.
- `PLANNING` | inside: yes | phase | plan() and reply
  - Use OMPL for the selected group.
  - Apply 10% scaling to speed and acceleration.
  - Check the trajectory. Check that the start remains unchanged.
  - Allow 5 s for the search.
  - Return PLANNING_FAILED if the 7 s deadline expires.
- `EXECUTING` | inside: yes | phase | dispatch() and callbacks
  - Send the trajectory to the arm OR rail controller.
  - Require acceptance within 3 s.
  - Require completion within the trajectory duration + 5 s.
  - Success requires SUCCEEDED and SUCCESSFUL.
- `STOPPING` | inside: yes | phase | stop() and tick()
  - Discard pending replies. Cancel the controller goal.
  - Wait for controller completion and still joints.
  - Stillness requires >= 10 samples over >= 200 ms.
  - The deadline is 3 s.
- `finish` | inside: yes | finish() returns the action result
  - Success requires the target position and an unchanged position for the other group.
  - If stopping remains unconfirmed, return STOP_FAILED. Latch FAULTED.

## Edges

- `requesting_client_external -> goal_cancel_callbacks` | ACTION `/agrobot_motion/move_joints` (`agrobot_motion/action/MoveJoints`) | Send a goal or cancellation request.
- `goal_cancel_callbacks -> requesting_client_external` | ACTION | Accept or reject the goal or cancellation request.
- `goal_cancel_callbacks -> CHECKING` | LOCAL | The next tick() calls preflight().
- `CHECKING -> controller_manager` | SERVICE | Send two asynchronous requests.
- `controller_manager -> CHECKING` | SERVICE | Return replies for the readiness check.
- `CHECKING -> PLANNING` | LOCAL | tick() advances when checks pass and feedback is fresh.
- `PLANNING -> planning` | SERVICE | Send the start pose and joint target.
- `planning -> PLANNING` | SERVICE | Return an error code and trajectory.
- `PLANNING -> EXECUTING` | LOCAL | Advance when the plan is valid and the start remains unchanged.
- `EXECUTING -> controllers` | ACTION | Send the trajectory goal.
- `controllers -> EXECUTING` | ACTION | Return acceptance and the terminal result.
- `EXECUTING -> STOPPING` | LOCAL | Handle every controller outcome.
- `goal_cancel_callbacks -> STOPPING` | LOCAL | Cancellation, a fault, or a deadline calls stop().
- `CHECKING -> STOPPING` | LOCAL | Cancellation, a fault, or a deadline calls stop().
- `PLANNING -> STOPPING` | LOCAL | Cancellation, a fault, or a deadline calls stop().
- `tick -> STOPPING` | LOCAL | Cancellation, a fault, or a deadline calls stop().
- `shutdown -> STOPPING` | LOCAL | An active goal calls stop().
- `STOPPING -> controllers` | ACTION | Cancel the active controller goal.
- `STOPPING -> finish` | LOCAL | Advance after stop confirmation or the deadline.
- `finish -> requesting_client_peer` | ACTION | Return the terminal status and result.
- `tick -> requesting_client_peer` | ACTION | Send feedback.
- `joint_feedback -> receive_joints` | TOPIC `/joint_states` (`sensor_msgs/msg/JointState`)
- `receive_joints -> stored_feedback` | DATA | Store feedback.
- `stored_feedback -> tick` | DATA | Read feedback on each tick.
- `configuration -> /agrobot_motion` | DATA | start() loads and checks the configuration.
