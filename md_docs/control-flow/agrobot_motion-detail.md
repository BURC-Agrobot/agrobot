# Control flow: package/node detail, agrobot_motion

- source_pdf: `pdf_docs/control-flow/agrobot_motion-detail.pdf`
- workspace: current
- configuration: `mock.launch.py`; one arm OR rail request at a time
- verification: source-derived; not runtime-verified
- node: `/agrobot_motion` | executable: `motion_node`
- executor: SingleThreadedExecutor; callbacks and shutdown run one at a time

## Semantics

- Edge `A -> B`: call or message path; kind is ACTION, SERVICE, TOPIC, LOCAL,
  or DATA.
- `LOCAL`: in-process call inside `/agrobot_motion`.
- `DATA`: stored data or context, not a call.
- Async edges do not imply callback order.
- Uppercase node names (CHECKING, PLANNING, EXECUTING, STOPPING) are
  MotionState phases.
- `inside: yes`: part of the `/agrobot_motion` node; `inside: no`: external
  peer or context.

## Nodes

- `requesting_client_external` | inside: no | Requesting client (external)
  - Not launched here.
  - ARM goal: six angles [rad].
  - RAIL goal: one position [m].
- `requesting_client_peer` | inside: no | Requesting client (same peer as `requesting_client_external`)
  - Feedback: state, elapsed_s.
  - Result: code, message, stop_confirmed.
  - Terminal status: succeeded, canceled, or aborted.
- `joint_feedback` | inside: no | context | Joint feedback source
  - `joint_state_broadcaster`; ros2_control at 100 Hz.
  - Hardware: FakeSystem, GenericSystem.
  - Stop evidence: mock only.
- `controller_manager` | inside: no | context | `/controller_manager`
  - Services: `list_hardware_components`, `list_controllers` (`controller_manager_msgs/srv`).
- `planning` | inside: no | context | `/move_group`; execution disabled
  - Service: `/plan_kinematic_path` (`moveit_msgs/srv/GetMotionPlan`).
- `controllers` | inside: no | context | `/arm_controller` OR `/rail_controller`
  - Action: `.../follow_joint_trajectory` (`control_msgs/action/FollowJointTrajectory`).
  - Commands mock hardware.
- `configuration` | inside: no | context | Configuration
  - URDF + SRDF + YAML acceleration limits.
  - Passed as `mock.launch.py` parameters.
  - Invalid model: node exits at startup.
- `goal_cancel_callbacks` | inside: yes | Goal / cancel callbacks
  - Idle + valid goal: go to CHECKING, call begin().
  - Otherwise: reject and log the reason.
  - Cancel of current goal: call stop().
- `receive_joints` | inside: yes | receive_joints()
  - Checks all 7 joints for finite values.
  - Ignores old or repeated stamps.
  - Bad sample: mark stale + throttled warning.
  - Tracks stillness for stop checks.
- `stored_feedback` | inside: yes | data | Stored feedback
  - Fresh if received < 0.5 s ago.
  - Read by every phase check.
- `tick` | inside: yes | 20 ms timer -> tick()
  - Publishes feedback; enforces deadlines.
  - CHECKING: preflight(), then plan().
  - PLANNING: wait for planner reply.
  - EXECUTING: require fresh feedback and other group still.
  - STOPPING: confirm stop or time out.
- `shutdown` | inside: yes | Shutdown on SIGINT / SIGTERM
  - Blocks new goals.
  - Active goal: call stop().
  - Exits after the active goal's result.
- `CHECKING` | inside: yes | phase | preflight() + replies
  - Waits for services, both actions, and a fresh pose.
  - Requires active mock hardware + both controllers.
  - Deadline 3 s: UNAVAILABLE.
- `PLANNING` | inside: yes | phase | plan() + reply
  - OMPL; selected group; 10% speed/accel scaling.
  - Checks trajectory + unchanged start.
  - Search 5 s; deadline 7 s: PLANNING_FAILED.
- `EXECUTING` | inside: yes | phase | dispatch() + callbacks
  - Sends to arm OR rail controller.
  - Accept within 3 s; finish within trajectory duration + 5 s.
  - Success needs SUCCEEDED + SUCCESSFUL.
- `STOPPING` | inside: yes | phase | stop() + tick()
  - Drops pending replies; cancels controller goal.
  - Waits for controller done + joints still.
  - Stillness: >= 10 samples over >= 200 ms; deadline 3 s.
- `finish` | inside: yes | finish() -> action result
  - Success: target reached, other group held.
  - Unconfirmed stop: STOP_FAILED; latch FAULTED.

## Edges

- `requesting_client_external -> goal_cancel_callbacks` | ACTION `/agrobot_motion/move_joints` (`agrobot_motion/action/MoveJoints`) | goal / cancel
- `goal_cancel_callbacks -> requesting_client_external` | ACTION | accept or reject goal / cancel
- `goal_cancel_callbacks -> CHECKING` | LOCAL | next tick() runs preflight()
- `CHECKING -> controller_manager` | SERVICE | two async requests
- `controller_manager -> CHECKING` | SERVICE | replies; check readiness
- `CHECKING -> PLANNING` | LOCAL | tick(): checks pass + fresh feedback, ready
- `PLANNING -> planning` | SERVICE | start pose + joint target
- `planning -> PLANNING` | SERVICE | error code + trajectory
- `PLANNING -> EXECUTING` | LOCAL | valid plan, start unchanged
- `EXECUTING -> controllers` | ACTION | trajectory goal
- `controllers -> EXECUTING` | ACTION | acceptance + terminal result
- `EXECUTING -> STOPPING` | LOCAL | every controller outcome
- `goal_cancel_callbacks -> STOPPING` | LOCAL | cancel, fault, or deadline calls stop()
- `CHECKING -> STOPPING` | LOCAL | cancel, fault, or deadline calls stop()
- `PLANNING -> STOPPING` | LOCAL | cancel, fault, or deadline calls stop()
- `tick -> STOPPING` | LOCAL | cancel, fault, or deadline calls stop()
- `shutdown -> STOPPING` | LOCAL | active goal calls stop()
- `STOPPING -> controllers` | ACTION | cancel active controller goal
- `STOPPING -> finish` | LOCAL | stop confirmed OR deadline
- `finish -> requesting_client_peer` | ACTION | terminal status + result
- `tick -> requesting_client_peer` | ACTION | feedback
- `joint_feedback -> receive_joints` | TOPIC `/joint_states` (`sensor_msgs/msg/JointState`)
- `receive_joints -> stored_feedback` | DATA | stores
- `stored_feedback -> tick` | DATA | read each tick
- `configuration -> /agrobot_motion` | DATA | loaded + checked in start()
