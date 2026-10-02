# Agrobot agent instructions

## Mission

Rebuild Agrobot's software from scratch with the engineering team. The FAST
(Function Analysis System Technique) diagram in `md_docs/agrobot-functions.md`
drives all development.

## Before writing code

- Read the FAST diagram and name the functions your change serves. Use them to
  guide design, implementation, and tests. If the diagram is unclear, ask the
  engineer to clarify it.
- Read project diagrams from `md_docs/`, which lists each diagram's nodes and
  edges as text for agents. `pdf_docs/` holds the same diagrams drawn for
  people; open a PDF only when an engineer asks. If a Markdown file is missing
  or disagrees with its PDF, stop and ask the engineer which one is correct.
- Reason from first principles: decide what the system must do and why, then
  derive the solution from the FAST functions, current requirements, and
  verified physical constraints. Do not carry over old designs or translate old
  code without this reasoning.
- Do NOT search, open, or read `previous-codebase-docs/`, or any copy or summary
  of it, unless the user explicitly asks for historical context. Even then, it
  never replaces the FAST diagram.

## Work step by step

- Develop and generate software one step at a time. Move to the next step only
  when an engineer says to.
- Treat each goal an engineer gives as its own discrete task. If a request holds
  several goals, split them into separate chunks and confirm the order.
- Before changing any file, reply with a numbered list of steps. For each step,
  say what it changes and how it will be checked. Wait for the engineer to
  approve the list before starting step 1.
- Keep each step small: one change that can be checked on its own, such as one
  file, one node, or one behavior.
- Zoom in on request: when the engineer asks to zoom in on, expand, or break
  down a step (for example, "zoom in on step 3" or "break step 2 down"), split
  that step into small numbered substeps (3.1, 3.2, ...). For each substep, say
  what it changes and how it will be checked. Reply with only that substep list,
  then wait for approval before starting any substep. A substep can be zoomed
  in on the same way (3.2.1, 3.2.2, ...).
- Do exactly one step per go-ahead. After it, report what changed and what was
  checked, show the step list with finished steps marked and the next step
  named, then stop.
- If new work appears during a step, such as a bug, a missing setting, or a
  failed check outside the step, do not fix it in that step. Add it to the list
  as a new step and ask the engineer.
- Finish, check, and report each step, then stop and wait for the engineer's
  go-ahead. Do not start later steps or bundle work ahead of approval.

## Scope and decisions

- Do only what was asked: no unrequested features, refactors, or fixes. Keep
  diffs minimal and focused on one change. Mention unrelated problems in one line.
- Decide routine, reversible matters yourself. Bring major choices and unclear
  hardware requirements to the team with a recommendation and alternatives.
- If a missing detail would change the result, ask one question; otherwise use
  the simplest reasonable reading.

## System design

- **One owner:** Give every command and state exactly one owner. Before
  connecting components, define who owns perception, task coordination,
  planning, motion execution, hardware, robot configuration, and startup.
  Package boundaries may change.
- **Coordinator vs. execution:** The coordinator requests operations. The
  execution layer owns motion limits, watchdogs (checks for stalled or missing
  activity), stopping, and stop confirmation.
- **Interfaces:** Keep components replaceable. Each interface states units,
  coordinate frames, timing, feedback, completion, cancellation, timeouts,
  failures, and recovery. Report every failure to the calling component.
- **Hardware:** Put hardware behind interfaces that support mocks (test
  substitutes) and simulation.
- **Configuration:** Use one authoritative robot configuration for geometry,
  planning, joint mappings, units, and drive settings.
- **Known problems to resolve:** competing command owners, missing gripper
  control, and conflicting joint settings.

## Language policy

Use a C++ control core and keep Python where it helps development. An all-C++
codebase is not required.

| Area | Default language |
| --- | --- |
| Hardware interfaces, drive communication, and trajectory execution | C++ |
| Motion validation, stopping, fault handling, and planning integration | C++ |
| Picking coordinator and perception/ML | Python initially |
| Launch, calibration, analysis, and test orchestration | Python |

- C++ alone does not guarantee timing. Measure delays and check scheduling,
  blocking calls, memory use, drivers, and hardware against required deadlines.
- Change a language choice only for measured performance needs or clear
  maintenance benefits, counting the cost of two toolchains and of tests across
  language boundaries.
- Define interfaces and complete one mock picking workflow before any broad
  language conversion. Bring a proposed move to all C++ to the team with
  evidence and tradeoffs.

## Code standards

### General

- **Simplicity:** Write the smallest correct solution in plain, readable code.
  Add no abstractions, configuration options, or generic helpers until there is
  a second real use. Add no speculative fallbacks, retries, or feature flags.
  Add a dependency only if it removes substantial code or risk, and explain why.
- **Style:** Match nearby names, layout, and patterns. Keep functions short with
  one job; use early returns instead of deep nesting; use descriptive names.
  Delete dead code instead of commenting it out. Do not reformat untouched lines.
- **Errors:** Handle errors explicitly; never silently swallow them.

### Comments

- Comment every logical step, including simple ones, and put a short summary
  comment above each function and major block.
- Put each comment on its own line directly above its code, at the same indentation.
- Keep each comment to one line under about 80 characters, in plain present
  tense, saying what the step does and why, such as
  "Skip header row so parsing starts at data".
- Use domain terms instead of variable names, such as "stack pointer", not "sp".
- Start with a space after the comment marker and a capital letter; use no
  trailing period.
- Update comments whenever the code changes.

### Implementation standards

| Area | Standard |
| --- | --- |
| C++ baseline | Follow the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines), especially types, array bounds, and object lifetimes. |
| Types and conversions | Use distinct types for units, joint IDs, and states. Initialize variables; check numeric conversions for lost precision or out-of-range values. |
| Unit naming | Name every variable that carries units as `variablename_unit`, such as `position_m`, `angle_rad`, `duration_s`, and `velocity_rad_per_s`. |
| Angular values | Use `pi` expressions for recognizable angles in Xacro, such as `${pi/2}` and `${2*pi/3}`. Keep arbitrary limits and speeds numeric, with clear units; do not round them to convenient fractions of `pi`. |
| Memory ownership | Prefer values and `std::unique_ptr`. Use `std::shared_ptr` only for shared ownership. Avoid manual `new`/`delete` and owning raw pointers. |
| Resource lifetime | Use RAII: objects automatically release memory, locks, and device handles when their lifetime ends. Keep referenced objects alive while callbacks, pointers, or views can access them. |
| Bounds and inputs | Validate user input, I/O, API data, ROS messages, and configuration at runtime; trust internal calls. Check array indexes, message lengths, ranges, units, and non-finite numbers (NaN or infinity). Standard containers alone do not ensure safe indexing. |
| Concurrency | Define who may change shared data. Use locks or suitable atomic operations to prevent conflicting access from concurrent threads. |
| Python typing | Annotate code; annotations do not replace runtime validation. Limit `Any` and type-check suppressions to justified, reviewed cases. |

## Build and verify

### Workflow

- Build a complete picking workflow in small, testable steps, guided by evidence
  and team priorities. Track unknowns while continuing work they do not block.
- Identify the execution mode (mock, simulation, or hardware). Test full
  workflows and failures without hardware; verify physical limits and
  assumptions on hardware.

### State machines and formal verification

- Use small, explicit state machines. Check that:
  - only one motion owner is active;
  - cancellation ends in a confirmed stop or an explicit stop failure;
  - success requires every required component to finish;
  - missing required configuration or homing (establishing joint reference
    positions) prevents motion.
- Write precise rules for critical behavior and check them with mathematical
  models or automated reasoning tools (formal verification). Align them with
  code checks and tests that deliberately introduce failures, in any language.

### Tests and analysis

- Before finishing, run existing tests and linters. Check affected interfaces
  and failure handling.
- Tests are engineer guided: write, change, or remove tests only when an
  engineer asks for them or approves them. Propose tests as listed steps
  instead of adding them on your own.
- Computer vision tests save the images they produce, such as inputs, heat
  maps, masks, and overlays, to `src/agrobot_perception/test/test_images/` on
  every run. Use fixed file names so each run replaces the last and engineers
  can check results by eye.
- Enforce compiler warnings, selected `clang-tidy` checks, and strict mypy in
  automated builds. Fix failures or use specific, reviewed exceptions.
- Run C++ tests with AddressSanitizer and UndefinedBehaviorSanitizer (memory
  errors and undefined behavior), and in a separate ThreadSanitizer build (data
  races: conflicting, unsynchronized memory access).
- Know the limits: sanitizers check only code that runs, and state-machine
  proofs do not prove memory safety.

### Measurement

- Log enough context to explain each task outcome. Measure picking success,
  speed, damage, and recovery.

## Commits

- Commit only when an engineer asks or approves, and make one commit per
  finished, checked step.
- Write each commit message as one sentence and nothing more: no body, no
  bullet list, and no trailing period.
- Start the sentence with an imperative verb and keep it under about 72
  characters, such as "Add mock gripper driver with open and close commands".
- Say what changed and why it matters, using domain terms, not file or
  variable names.
- Stage only the files the step changed. Never commit secrets, build outputs,
  or logs.
- Push each commit to GitHub immediately after making it. If the push fails,
  report the error instead of retrying or forcing it.
- Do not amend, rebase, or force-push shared history without engineer approval.

## Reporting and documentation

- Share findings as work progresses. Lead with the outcome or recommendation and
  include only the evidence and tradeoffs needed to understand it. Base
  recommendations on requirements, gaps, or test evidence; explain benefits,
  costs, and effects on connected components.
- Summarize changes in 1–3 bullets without repeating the code. Report working
  behavior and remaining integration gaps, and say whether each result comes
  from a build, mock, simulation, or hardware. Skip next-step suggestions unless asked.
- Never invent commands or claim checks you did not run.
- Create or update documentation, including READMEs, only when an engineer
  explicitly asks or approves. Then document verified dependencies and setup,
  build, launch, and check commands.
- Keep additions and edits to this file brief and readable for a general
  undergraduate engineering audience: plain language, with necessary technical
  terms explained.
