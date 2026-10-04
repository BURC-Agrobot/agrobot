# States: Level 1, whole-robot operation

- source_pdf: `pdf_docs/agrobot-states.pdf`
- workspace: current
- view: Conceptual view of the current mock configuration.
- method: Static source inspection.
- whole_robot_state_machine: The source implements no whole-robot state machine.
- system_state_owner: The current source defines no system state owner.
- motion_level_states: `/agrobot_motion` contains the motion states (Level 3).
- transitions: The source defines no whole-robot transitions.
- initial_or_final_markers: The source defines no whole-robot initial or final markers.

## Semantics

- `kind: context`: The node describes a background fact, not a state.
- `kind: inferred_mode`: Source inspection suggests this configuration mode.
  The source does not store it as a state.
- `kind: missing`: The source does not implement this required capability.
- Edges connect context nodes. They do not represent transitions.

## Nodes

- `robot_model` | kind: context | Robot model
  - The arm has six revolute joints.
  - The rail has one prismatic joint.
  - "ready" and "start" name poses, not operating states.
- `mock_motion_configuration` | kind: inferred_mode | Mock motion configuration
  - Source inspection suggests this mode. The source does not store it as a state.
  - The hardware uses only `mock_components/GenericSystem`.
  - Controller activation has no runtime verification.
- `motion_owner` | kind: context | Motion owner
  - `/agrobot_motion` handles one arm OR rail request.
  - It uses the arm and rail controllers and joint feedback.
  - Motion state FAULTED rejects new goals until restart.
  - This fault applies locally, not to the whole robot.
- `system_coordination` | kind: missing | System coordination is absent
  - The source has no supervisor, pick coordinator, gripper control, or interface to real drives.
  - The source has no whole-robot readiness, stop, fault, or recovery transitions.
  - `/agrobot_motion` provides stopping only for its own request.
  - Supervisor diagrams from the previous codebase describe a separate system.

## Edges

- `robot_model -- mock_motion_configuration` | context
- `mock_motion_configuration -- motion_owner` | context
- `mock_motion_configuration -- system_coordination` | context | scope boundary
