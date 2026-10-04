# Agrobot agent instructions

## Mission

Rebuild Agrobot's software from the beginning with the engineering team.
The FAST (Function Analysis System Technique) diagram in
`md_docs/agrobot-functions.md` guides all development.

## Before writing code

- Read the FAST diagram. Name the functions that your change serves.
  Use these functions to guide design, implementation, and tests.
  If the diagram is unclear, ask the engineer to clarify it.
- Read the project diagrams in `md_docs/`. Each Markdown file lists diagram
  nodes and edges for agents. `pdf_docs/` contains the same diagrams for people.
  Open a PDF only when an engineer asks.
  If a Markdown file is missing or disagrees with its PDF, stop.
  Ask the engineer which version is correct.
- Reason from first principles. Decide what the system must do and why.
  Derive the solution from FAST functions, current requirements, and verified
  physical constraints. Do not reuse old designs or translate old code without
  this reasoning.
- Do NOT search, open, or read `previous-codebase-docs/`, its copies, or its
  summaries unless the user explicitly requests historical context.
  Historical context never replaces the FAST diagram.

## Work step by step

- Develop and generate software one step at a time.
  Start the next step only when an engineer gives permission.
- Treat each engineer's goal as a separate task.
  If a request contains several goals, divide them into separate parts.
  Confirm their order.
- Before changing any file, reply with a numbered list of steps.
  State what each step changes and how you will check it.
  Wait for the engineer to approve the list before starting step 1.
- Keep each step small. Make one change that you can check separately.
  For example, change one file, one node, or one behavior.
- If the engineer requests more detail about a step, divide it into small
  numbered substeps, such as 3.1 and 3.2.
  This includes requests to "zoom in on", "expand", or "break down" a step.
  State what each substep changes and how you will check it.
  Reply with only the substep list. Wait for approval before starting any substep.
  Apply the same process to substeps, such as 3.2.1 and 3.2.2.
- Complete exactly one step per approval.
  Report what changed and what you checked.
  Show the step list with completed steps marked and the next step identified.
  Then stop.
- If new work appears during a step, do not complete it during that step.
  Examples include a bug, a missing setting, or a failed check outside the step.
  Add the work to the list as a new step. Ask the engineer for approval.
- Finish each step. Check the result. Report the result.
  Then stop and wait for the engineer's approval.
  Do not start later steps or combine work before approval.

## Scope and decisions

- Complete only the requested work. Do not add unrequested features, refactors,
  or fixes. Keep changes minimal and focused on one change.
  Report unrelated problems in one line.
- Decide routine, reversible matters yourself.
  Present major choices and unclear hardware requirements to the team.
  Include a recommendation and alternatives.
- If a missing detail would change the result, ask one question.
  Otherwise, use the simplest reasonable interpretation.

## System design

- **One owner:** Give every command and state exactly one owner.
  Before connecting components, define owners for these areas:
  - Perception.
  - Task coordination.
  - Planning.
  - Motion execution.
  - Hardware.
  - Robot configuration.
  - Startup.
  Package boundaries may change.
- **Coordinator and execution:** The coordinator requests operations.
  The execution layer owns motion limits, watchdogs, stopping, and stop confirmation.
  Watchdogs check for stalled or missing activity.
- **Interfaces:** Keep components replaceable.
  Each interface specifies units, coordinate frames, timing, feedback, completion,
  cancellation, timeouts, failures, and recovery.
  Report every failure to the calling component.
- **Hardware:** Access hardware through interfaces that support mocks and simulation.
  Mocks are test substitutes.
- **Configuration:** Use one authoritative robot configuration for geometry,
  planning, joint mappings, units, and drive settings.
- **Known problems:** Resolve competing command owners, missing gripper control,
  and conflicting joint settings.

## Language policy

Use a C++ control core. Keep Python where it helps development.
The codebase does not have to use C++ exclusively.

| Area | Default language |
| --- | --- |
| Hardware interfaces, drive communication, and trajectory execution | C++ |
| Motion validation, stopping, fault handling, and planning integration | C++ |
| Picking coordinator and perception/ML | Python initially |
| Launch, calibration, analysis, and test orchestration | Python |

- C++ alone does not guarantee timing. Measure delays.
  Check scheduling, blocking calls, memory use, drivers, and hardware against
  required deadlines.
- Change a language choice only for measured performance needs or clear
  maintenance benefits. Include the cost of two toolchains and tests across
  language boundaries.
- Define interfaces before any broad language conversion.
  Complete one mock picking workflow before that conversion.
  Present any proposal to use C++ exclusively to the team with evidence and tradeoffs.

## Code standards

### General

- **Simplicity:** Write the smallest correct solution in plain, readable code.
  Add abstractions, configuration options, or generic helpers only when a second
  real use exists. Do not add speculative fallbacks, retries, or feature flags.
  Add a dependency only if it removes substantial code or risk.
  Explain why the dependency is necessary.
- **Style:** Match nearby names, layout, and patterns.
  Keep functions short with one job. Use early returns instead of deep nesting.
  Use descriptive names. Delete dead code instead of converting it to comments.
  Do not reformat lines outside the change.
- **Errors:** Handle errors explicitly. Never silently ignore them.

### Comments

- Comment every logical step, including simple steps.
  Put a short summary comment above each function and major block.
- Put each comment on its own line directly above its code.
  Use the same indentation as the code.
- Keep each comment to one line under about 80 characters.
  Use plain present tense. State what the step does and why.
  For example: "Skip the header row so parsing starts at the data".
- Use domain terms instead of variable names, such as "stack pointer" instead of "sp".
- Put a space after the comment marker. Start the comment with a capital letter.
  Do not add a final period.
- Update comments whenever the code changes.

### Implementation standards

| Area | Standard |
| --- | --- |
| C++ baseline | Follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines). Pay particular attention to types, array bounds, and object lifetimes. |
| Types and conversions | Use distinct types for units, joint IDs, and states. Initialize variables. Check numeric conversions for lost precision or values outside the permitted range. |
| Unit naming | Name every variable that carries units as `variablename_unit`. Examples: `position_m`, `angle_rad`, `duration_s`, and `velocity_rad_per_s`. |
| Angular values | Use `pi` expressions for recognizable angles in Xacro, such as `${pi/2}` and `${2*pi/3}`. Keep arbitrary limits and speeds numeric, with clear units. Do not round them to convenient fractions of `pi`. |
| Memory ownership | Prefer values and `std::unique_ptr`. Use `std::shared_ptr` only for shared ownership. Avoid manual `new`/`delete` and owning raw pointers. |
| Resource lifetime | Use RAII: objects automatically release memory, locks, and device handles when their lifetime ends. Keep referenced objects alive while callbacks, pointers, or views can access them. |
| Bounds and inputs | Validate user input, I/O, API data, ROS messages, and configuration at runtime. Trust internal calls. Check array indexes, message lengths, ranges, units, and non-finite numbers (NaN or infinity). Standard containers alone do not ensure safe indexing. |
| Concurrency | Define who may change shared data. Use locks or suitable atomic operations to prevent conflicting access from concurrent threads. |
| Python typing | Annotate code. Annotations do not replace runtime validation. Limit `Any` and suppressions of type checks to justified, reviewed cases. |

## Build and verify

### Workflow

- Build a complete picking workflow in small steps that you can test.
  Use evidence and team priorities to guide the work.
  Track unknowns. Continue work that those unknowns do not prevent.
- Identify the execution mode: mock, simulation, or hardware.
  Test full workflows and failures without hardware.
  Verify physical limits and assumptions on hardware.

### State machines and formal verification

- Use small, explicit state machines. Check these conditions:
  - Only one motion owner is active.
  - Cancellation ends in a confirmed stop or an explicit stop failure.
  - Success requires every required component to finish.
  - Missing required configuration or homing prevents motion.
    Homing establishes joint reference positions.
- Write precise rules for critical behavior.
  Check those rules with mathematical models or automated reasoning tools
  (formal verification). Align the rules with code checks and tests that
  deliberately introduce failures, in any language.

### Tests and analysis

- Before finishing, run existing tests and linters.
  Check affected interfaces and failure handling.
- Engineers guide tests. Write, change, or remove tests only when an engineer
  requests or approves this work.
  Propose tests as listed steps instead of adding them yourself.
- On every run, computer vision tests save all images they produce to
  `src/agrobot_perception/test/test_images/`.
  These images include inputs, heat maps, masks, and overlays.
  Use fixed filenames so each run replaces the previous images.
  Engineers can then check the results visually.
- Enforce compiler warnings, selected `clang-tidy` checks, and strict mypy in
  automated builds. Fix failures or use specific, reviewed exceptions.
- Run C++ tests with AddressSanitizer and UndefinedBehaviorSanitizer.
  These tools check for memory errors and undefined behavior.
  Run a separate ThreadSanitizer build to check for data races.
  Data races are conflicting, unsynchronized accesses to memory.
- Recognize the limits. Sanitizers check only code that runs.
  State-machine proofs do not prove memory safety.

### Measurement

Log enough context to explain each task outcome.
Measure picking success, speed, damage, and recovery.

## Commits

- Commit only when an engineer requests or approves a commit.
  Make one commit per completed, checked step.
- Write each commit message as one sentence only.
  Do not include a body, bullet list, or final period.
- Start the sentence with an imperative verb. Keep it under about 72 characters.
  For example: "Add mock gripper driver with open and close commands".
- State what changed and why it matters.
  Use domain terms instead of file or variable names.
- Stage only the files that the step changed.
  Never commit secrets, build outputs, or logs.
- Push each commit to GitHub immediately after creating it.
  If the push fails, report the error. Do not retry or force the push.
- Do not amend, rebase, or force-push shared history without engineer approval.

## Reporting and documentation

- Report findings as work progresses.
  Start with the outcome or recommendation.
  Include only the evidence and tradeoffs necessary to understand it.
  Base recommendations on requirements, gaps, or test evidence.
  Explain benefits, costs, and effects on connected components.
- Summarize changes in 1–3 bullets without repeating the code.
  Report working behavior and remaining integration gaps.
  State whether each result comes from a build, mock, simulation, or hardware.
  Do not suggest next steps unless the engineer asks.
- Never invent commands or claim checks that you did not run.
- Create or update documentation, including READMEs, only when an engineer
  explicitly requests or approves this work.
  Then document verified dependencies and setup, build, launch, and check commands.
- Keep additions and edits to this file brief and readable for a general
  undergraduate engineering audience.
  Use plain language. Explain necessary technical terms.
