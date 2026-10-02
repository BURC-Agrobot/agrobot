# States: Level 1, whole-robot operation

- source_pdf: `pdf_docs/agrobot-states.pdf`
- workspace: current
- view: conceptual; current mock configuration
- method: static source inspection
- whole_robot_state_machine: none implemented
- system_state_owner: none in current source
- motion_level_states: live in `/agrobot_motion` (Level 3)
- transitions: none in source
- initial_or_final_markers: none in source

## Semantics

- `kind: context`: background fact, not a state.
- `kind: inferred_mode`: configuration mode inferred from source, not a stored
  state.
- `kind: missing`: required capability with no implementation.
- Edges are context links only; none are transitions.

## Nodes

- `robot_model` | kind: context | Robot model
  - Arm: six revolute joints.
  - Rail: one prismatic joint.
  - "ready" and "start" are named poses, not operating states.
- `mock_motion_configuration` | kind: inferred_mode | Mock motion configuration
  - Inferred mode, not a stored state.
  - Hardware: `mock_components/GenericSystem` only.
  - Controller activation is not runtime-verified.
- `motion_owner` | kind: context | Motion owner
  - `/agrobot_motion` handles one arm OR rail request.
  - Uses arm + rail controllers and joint feedback.
  - Motion FAULTED rejects new goals until restart (local, not robot-wide).
- `system_coordination` | kind: missing | System coordination is absent
  - No supervisor, pick coordinator, gripper control, or real-drive interface.
  - No robot-wide readiness, stop, fault, or recovery transitions.
  - Stopping exists only inside `/agrobot_motion` for its own request.
  - Previous-codebase supervisor diagrams describe a separate system.

## Edges

- `robot_model -- mock_motion_configuration` | context
- `mock_motion_configuration -- motion_owner` | context
- `mock_motion_configuration -- system_coordination` | context; scope boundary
