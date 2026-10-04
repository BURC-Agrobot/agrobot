#include "agrobot_motion/motion_rules.hpp"

#include <chrono>
#include <cmath>
#include <csignal>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/srv/get_motion_plan.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

namespace agrobot_motion
{
// Use monotonic time so clock corrections do not change deadlines
using Clock = std::chrono::steady_clock;
// Name the request and controller interfaces used by this node
using Motion = action::MoveJoints;
using MotionHandle = rclcpp_action::ServerGoalHandle<Motion>;
using Trajectory = control_msgs::action::FollowJointTrajectory;
using TrajectoryHandle = rclcpp_action::ClientGoalHandle<Trajectory>;
using Hardware = controller_manager_msgs::srv::ListHardwareComponents;
using Controllers = controller_manager_msgs::srv::ListControllers;
using Plan = moveit_msgs::srv::GetMotionPlan;
using namespace std::chrono_literals;

// Run periodic checks often enough to notice stalls between feedback samples
constexpr auto tick_period_ms = 20ms;
// Treat feedback as stale once it is older than this
constexpr auto feedback_timeout_ms = 500ms;
// Allow feedback stamps this far in the future to absorb clock skew
constexpr double feedback_lead_s = 0.1;
// Restart stop evidence when samples arrive further apart than this
constexpr auto feedback_gap_ms = 100ms;
// Require this many distinct still samples before confirming a stop
constexpr std::uint64_t still_samples = 10;
// Require stillness to last this long before confirming a stop
constexpr auto still_window_ms = 200ms;
// Count an arm joint as still when it moves less than this
constexpr double still_arm_rad = 1e-5;
// Count the rail as still when it moves less than this
constexpr double still_rail_m = 1e-6;
// Allow this long for hardware, controllers, planner, and feedback checks
constexpr auto checking_timeout_s = 3s;
// Give the planner this much search time per request
constexpr double planning_time_s = 5.0;
// Allow planning time plus a short service response margin
constexpr auto planning_timeout_s = 7s;
// Constrain the planned endpoint this closely in radians or metres
constexpr double plan_goal_tolerance_si = 1e-5;
// Allow this long for controller acceptance or stop confirmation
constexpr auto handshake_timeout_s = 3s;
// Add this margin to the planned duration before execution times out
constexpr double execution_margin_s = 5.0;
// Give the controller this long to settle at its endpoint
constexpr double settle_time_s = 2.0;

// Own one motion request through planning, execution, and stopping
class MotionNode : public rclcpp::Node
{
public:
  // Load launch parameters so validation uses the shared robot model
  MotionNode() : Node("agrobot_motion", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)) {}

  // Connect the motion interfaces after shared node ownership exists
  void start()
  {
    // Load model data without starting unused kinematics plugins
    const robot_model_loader::RobotModelLoader loader(shared_from_this(), "robot_description", false);
    model_ = loader.getModel();
    // Refuse startup when groups or limits cannot support validation
    if (!model_ || !valid_model(*model_)) {
      throw std::runtime_error("Missing or incompatible robot model, groups, or joint limits");
    }
    // Connect readiness checks and planning without blocking callbacks
    hardware_client_ = create_client<Hardware>("/controller_manager/list_hardware_components");
    controllers_client_ = create_client<Controllers>("/controller_manager/list_controllers");
    plan_client_ = create_client<Plan>("/plan_kinematic_path");
    // Keep separate controller clients for arm and rail trajectories
    arm_client_ = rclcpp_action::create_client<Trajectory>(this, "/arm_controller/follow_joint_trajectory");
    rail_client_ = rclcpp_action::create_client<Trajectory>(this, "/rail_controller/follow_joint_trajectory");
    // Observe joint feedback to check progress and confirm stops
    joint_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::JointState & message) {receive_joints(message);});
    // Expose one action endpoint so both groups share motion ownership
    server_ = rclcpp_action::create_server<Motion>(
      this, "/agrobot_motion/move_joints",
      // Validate each goal before reserving the shared motion slot
      [this](const auto &, const auto & goal) {
        // Reject invalid or overlapping goals instead of queuing them
        const auto reason = rejection(*goal);
        if (!reason.empty()) {
          // Log the specific reason so the caller can diagnose the refusal
          RCLCPP_WARN(get_logger(), "Rejected motion: %s", reason.c_str());
          return rclcpp_action::GoalResponse::REJECT;
        }
        // Reserve ownership before another goal callback can run
        transition(Event::request);
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      // Cancel only the current request while stop handling remains active
      [this](const std::shared_ptr<MotionHandle> & goal) {
        // Reject cancellation for unrelated goals or a latched fault
        if (goal != goal_ || state_.get() == State::faulted) {
          return rclcpp_action::CancelResponse::REJECT;
        }
        // Start stop handling before acknowledging the cancellation
        stop(Motion::Result::CANCELED, "Cancellation requested");
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      // Initialize the accepted request for asynchronous processing
      [this](const std::shared_ptr<MotionHandle> & goal) {begin(goal);});
    // Check progress regularly without blocking action callbacks
    timer_ = create_wall_timer(tick_period_ms, [this]() {tick();});
    // Identify mock operation so readiness cannot imply hardware support
    RCLCPP_INFO(get_logger(), "MOCK ONLY: /agrobot_motion/move_joints ready; one arm or rail request at a time");
  }

  // Block new goals and stop active motion before ROS shuts down
  void shutdown_motion()
  {
    // Close the acceptance gate before handling the active request
    shutting_down_ = true;
    // Route active motion through the normal stop confirmation path
    if (goal_) {
      stop(Motion::Result::EXECUTION_FAILED, "Node shutting down");
    }
  }

  // Report pending work so shutdown can wait for a terminal result
  bool has_goal() const { return static_cast<bool>(goal_); }

private:
  // Return why a new goal cannot start, or nothing when it may start
  std::string rejection(const Motion::Goal & goal) const
  {
    // Refuse new work once shutdown closes the acceptance gate
    if (shutting_down_) {
      return "node is shutting down";
    }
    // Refuse overlapping work so only one request owns motion
    if (state_.get() != State::idle) {
      return std::string("motion state is ") + state_.name();
    }
    // Refuse targets that break the action contract or joint limits
    if (!valid_goal(goal, *model_)) {
      return "unknown group, non-finite or out-of-limit target, or nonzero unused field";
    }
    return {};
  }

  // Apply and log only transitions allowed by the motion rules
  void transition(Event event)
  {
    // Expose invalid internal transitions as programming errors
    if (!state_.apply(event)) {
      throw std::logic_error("Invalid motion state transition");
    }
    // Record state changes to explain motion outcomes
    RCLCPP_INFO(get_logger(), "Motion state: %s", state_.name());
  }

  // Require valid recent feedback before trusting the robot position
  bool fresh() const
  {
    return feedback_valid_ && Clock::now() - last_sample_ns_ < feedback_timeout_ms;
  }

  // Validate feedback and measure how long all joints remain still
  void receive_joints(const sensor_msgs::msg::JointState & message)
  {
    // Stage the incoming positions before changing trusted feedback
    Positions sample;
    // Reject malformed timestamps before constructing ROS time
    if (message.header.stamp.sec < 0 || message.header.stamp.nanosec >= 1000000000U) {
      // Limit warning frequency so a bad publisher does not flood the logs
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Ignored joint feedback: malformed timestamp");
      feedback_valid_ = false;
      stationary_samples_ = 0;
      return;
    }
    // Measure message age to detect delayed or future feedback
    const rclcpp::Time stamp_ns(message.header.stamp);
    const double age_s = (now() - stamp_ns).seconds();
    // Invalidate stop evidence when timestamps or positions are unusable
    if (age_s > std::chrono::duration<double>(feedback_timeout_ms).count() || age_s < -feedback_lead_s ||
      !read_positions(message, sample))
    {
      // Limit warning frequency so a bad publisher does not flood the logs
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "Ignored joint feedback: stale, future-stamped, or missing required joints");
      feedback_valid_ = false;
      stationary_samples_ = 0;
      return;
    }
    // Ignore repeated or older samples so they cannot prove a stop
    if (stamp_ns <= last_stamp_ns_) {
      return;
    }
    // Remember the timestamp so only newer feedback counts
    last_stamp_ns_ = stamp_ns;
    // Restart stop evidence after a gap in the feedback stream
    if (Clock::now() - last_sample_ns_ > feedback_gap_ms) {
      stationary_samples_ = 0;
    }
    // Commit valid positions and their local receipt time
    last_sample_ns_ = Clock::now();
    feedback_valid_ = true;
    actual_ = sample;

    // Check rail displacement because mock velocity can stay zero
    bool stationary = std::abs(actual_.rail.value_m - stationary_anchor_.rail.value_m) <= still_rail_m;
    // Include every arm joint in the displacement check
    for (std::size_t index = 0; index < actual_.arm.size(); ++index) {
      stationary = stationary &&
        std::abs(actual_.arm[index].value_rad - stationary_anchor_.arm[index].value_rad) <= still_arm_rad;
    }
    // Start a new stillness window whenever any joint moves
    if (!stationary || stationary_samples_ == 0) {
      stationary_anchor_ = actual_;
      stationary_since_ns_ = last_sample_ns_;
      stationary_samples_ = 1;
    } else {
      // Count another distinct sample supporting the same stop window
      ++stationary_samples_;
    }
  }

  // Reset per-request state before checking the motion dependencies
  void begin(const std::shared_ptr<MotionHandle> & goal)
  {
    // Keep the accepted goal available for feedback and completion
    goal_ = goal;
    // Distinguish this request from delayed callbacks for older goals
    ++generation_;
    // Select the requested group and decode its target units
    group_ = goal->get_goal()->group == Motion::Goal::ARM ? Group::arm : Group::rail;
    target_ = target_positions(*goal->get_goal());
    // Bound the time allowed for dependencies and feedback to become ready
    started_ns_ = Clock::now();
    deadline_ns_ = started_ns_ + checking_timeout_s;
    // Clear readiness flags so each request checks the running system
    hardware_ready_ = false;
    controllers_ready_ = false;
    preflight_sent_ = false;
    // Clear controller state so previous motion cannot prove completion
    dispatched_ = false;
    child_terminal_ = false;
    child_.reset();
  }

  // Accept service replies only for the active request in the expected phase
  bool current(std::uint64_t generation, State phase) const
  {
    return generation == generation_ && goal_ && state_.get() == phase;
  }

  // Check live mock hardware and controller ownership before planning
  void preflight()
  {
    // Wait for every dependency and fresh feedback within the deadline
    if (!hardware_client_->service_is_ready() || !controllers_client_->service_is_ready() ||
      !plan_client_->service_is_ready() || !arm_client_->action_server_is_ready() ||
      !rail_client_->action_server_is_ready() || !fresh())
    {
      return;
    }
    // Send readiness queries once and tag them with this request
    preflight_sent_ = true;
    const auto generation = generation_;
    // Query the active hardware plugin before permitting motion
    hardware_request_ = hardware_client_->async_send_request(std::make_shared<Hardware::Request>(),
      // Evaluate the hardware reply only while this request is checking
      [this, generation](rclcpp::Client<Hardware>::SharedFuture response) {
        // Ignore replies that arrive after cancellation or a newer request
        if (!current(generation, State::checking)) {return;}
        // Release the completed query and check the mock hardware identity
        hardware_request_.reset();
        hardware_ready_ = mock_hardware(*response.get());
        // Report unsupported hardware before sending any trajectory
        if (!hardware_ready_) {
          stop(Motion::Result::UNAVAILABLE, "Requires one active FakeSystem using mock_components/GenericSystem");
        }
      }).request_id;
    // Query both controllers so their command ownership can be checked
    controllers_request_ = controllers_client_->async_send_request(std::make_shared<Controllers::Request>(),
      // Evaluate controller readiness for the current checking request
      [this, generation](rclcpp::Client<Controllers>::SharedFuture response) {
        // Ignore replies that arrive after cancellation or a newer request
        if (!current(generation, State::checking)) {return;}
        // Release the query and validate active position interfaces
        controllers_request_.reset();
        controllers_ready_ = controllers_ready(*response.get());
        // Stop the request when either controller lacks its required joints
        if (!controllers_ready_) {
          stop(Motion::Result::UNAVAILABLE, "Arm and rail controllers must be active and own their position interfaces");
        }
      }).request_id;
  }

  // Request a bounded joint-space plan from the measured robot state
  void plan()
  {
    // Capture the measured start state that the plan must begin from
    start_ = actual_;
    // Bound the planner search and its service response
    deadline_ns_ = Clock::now() + planning_timeout_s;
    // Build an OMPL request for only the selected joint group
    auto request = std::make_shared<Plan::Request>();
    auto & motion = request->motion_plan_request;
    motion.group_name = group_ == Group::arm ? "arm" : "rail";
    motion.pipeline_id = "ompl";
    motion.allowed_planning_time = planning_time_s;
    motion.num_planning_attempts = 1;
    // Scale configured speed and acceleration to the shared motion limit
    motion.max_velocity_scaling_factor = motion_scale;
    motion.max_acceleration_scaling_factor = motion_scale;
    // Include both groups so collision checking uses the full robot pose
    motion.start_state.joint_state.name = arm_joints;
    motion.start_state.joint_state.name.push_back(rail_joints.front());
    motion.start_state.joint_state.position = positions_si(start_, Group::arm);
    motion.start_state.joint_state.position.push_back(start_.rail.value_m);
    motion.start_state.is_diff = false;
    // Prepare endpoint constraints in radians or metres for this group
    moveit_msgs::msg::Constraints constraints;
    const auto target_si = positions_si(target_, group_);
    const auto & names = joint_names(group_);
    // Constrain each requested joint with a small endpoint tolerance
    for (std::size_t index = 0; index < names.size(); ++index) {
      moveit_msgs::msg::JointConstraint joint;
      joint.joint_name = names[index];
      joint.position = target_si[index];
      joint.tolerance_above = plan_goal_tolerance_si;
      joint.tolerance_below = plan_goal_tolerance_si;
      joint.weight = 1.0;
      constraints.joint_constraints.push_back(joint);
    }
    // Attach the complete target and tag the asynchronous plan request
    motion.goal_constraints.push_back(constraints);
    const auto generation = generation_;
    // Send the plan request while keeping cancellation responsive
    plan_request_ = plan_client_->async_send_request(request,
      // Validate the returned plan before handing it to a controller
      [this, generation](rclcpp::Client<Plan>::SharedFuture response) {
        // Ignore replies that arrive after cancellation or a newer request
        if (!current(generation, State::planning)) {return;}
        // Release the completed query and inspect the planner result
        plan_request_.reset();
        const auto & result = response.get()->motion_plan_response;
        // Report planning failures without sending a controller goal
        if (result.error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
          stop(Motion::Result::PLANNING_FAILED, "MoveIt planning failed: " + std::to_string(result.error_code.val));
          return;
        }
        // Reject trajectories outside the supported single-axis joints
        if (!result.trajectory.multi_dof_joint_trajectory.joint_names.empty() ||
          !result.trajectory.multi_dof_joint_trajectory.points.empty())
        {
          stop(Motion::Result::PLANNING_FAILED, "Unexpected multi-DOF trajectory");
          return;
        }
        // Check trajectory contents against the target and configured limits
        const auto error = validate_trajectory(result.trajectory.joint_trajectory, group_, start_, target_, *model_);
        // Return the validation reason when planner output is unusable
        if (!error.empty()) {
          stop(Motion::Result::PLANNING_FAILED, error);
          return;
        }
        // Reject a plan if feedback no longer matches its starting pose
        if (!fresh() || !near(actual_, start_)) {
          stop(Motion::Result::EXECUTION_FAILED, "Robot state changed or became stale during planning");
          return;
        }
        // Send only the validated plan for the unchanged start state
        dispatch(result.trajectory.joint_trajectory);
      }).request_id;
  }

  // Select the controller belonging to the active motion group
  rclcpp_action::Client<Trajectory>::SharedPtr trajectory_client() const
  {
    return group_ == Group::arm ? arm_client_ : rail_client_;
  }

  // Send a validated trajectory and track acceptance and completion
  void dispatch(const trajectory_msgs::msg::JointTrajectory & trajectory)
  {
    // Keep ownership while waiting for the controller to accept motion
    transition(Event::dispatch);
    dispatched_ = true;
    // Bound the wait for controller acceptance
    deadline_ns_ = Clock::now() + handshake_timeout_s;
    // Use planned duration to set the later execution deadline
    const double duration_s = rclcpp::Duration(trajectory.points.back().time_from_start).seconds();
    const auto generation = generation_;
    // Keep controller callbacks associated with this motion request
    rclcpp_action::Client<Trajectory>::SendGoalOptions options;
    // Handle controller acceptance before monitoring execution
    options.goal_response_callback = [this, generation, duration_s](const TrajectoryHandle::SharedPtr & child) {
        // Ignore controller callbacks belonging to an older request
        if (generation != generation_) {return;}
        // Store the controller handle so cancellation can reach it
        child_ = child;
        // Treat controller rejection as terminal without assuming success
        if (!child_) {
          child_terminal_ = true;
          stop(Motion::Result::EXECUTION_FAILED, "Controller rejected the trajectory");
          return;
        }
        // Cancel late acceptance if stopping already started
        if (state_.get() == State::stopping || state_.get() == State::faulted) {
          cancel_child();
          return;
        }
        // Bound execution by planned duration plus a completion margin
        deadline_ns_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(duration_s + execution_margin_s));
      };
    // Inspect controller status before checking the final stopped pose
    options.result_callback = [this, generation](const TrajectoryHandle::WrappedResult & result) {
        // Ignore controller callbacks belonging to an older request
        if (generation != generation_) {return;}
        // Require new stop evidence after the controller result arrives
        child_terminal_ = true;
        stationary_samples_ = 0;
        // Preserve an existing cancellation or fault instead of overwriting it
        if (state_.get() == State::stopping || state_.get() == State::faulted) {return;}
        // Require both successful action status and a successful payload
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result &&
          result.result->error_code == Trajectory::Result::SUCCESSFUL)
        {
          stop(Motion::Result::SUCCESS, "Controller completed; checking final position and stop");
        } else {
          // Report controller failure while still requiring stop confirmation
          stop(Motion::Result::EXECUTION_FAILED, "Controller failed: " +
            (result.result ? result.result->error_string : "no result"));
        }
      };
    // Create the controller goal from the validated trajectory
    Trajectory::Goal request;
    request.trajectory = trajectory;
    // Start on receipt so an old planning timestamp cannot delay motion
    request.trajectory.header.stamp = builtin_interfaces::msg::Time{};
    // Give the controller a bounded time to settle at its endpoint
    request.goal_time_tolerance = rclcpp::Duration::from_seconds(settle_time_s);
    // Set endpoint tolerances in each selected joint group’s units
    for (const auto & name : joint_names(group_)) {
      control_msgs::msg::JointTolerance tolerance;
      tolerance.name = name;
      tolerance.position = group_ == Group::arm ? arm_tolerance_rad : rail_tolerance_m;
      request.goal_tolerance.push_back(tolerance);
    }
    // Send without blocking feedback or cancellation processing
    trajectory_client()->async_send_goal(request, options);
  }

  // Request cancellation without treating acknowledgement as a stop
  void cancel_child()
  {
    // Skip cancellation until a live controller goal handle exists
    if (!child_ || child_terminal_) {return;}
    try {
      // Ask the controller to stop the accepted trajectory
      trajectory_client()->async_cancel_goal(child_);
    } catch (const rclcpp_action::exceptions::UnknownGoalHandleError & error) {
      // Keep waiting for result and feedback if the handle disappears
      RCLCPP_WARN(get_logger(), "Cancel could not find goal: %s", error.what());
    }
  }

  // Release pending service replies once their motion request stops
  void clear_requests()
  {
    // Remove the hardware query so an abandoned reply is not retained
    if (hardware_request_) {hardware_client_->remove_pending_request(*hardware_request_); hardware_request_.reset();}
    // Remove the controller query so an abandoned reply is not retained
    if (controllers_request_) {controllers_client_->remove_pending_request(*controllers_request_); controllers_request_.reset();}
    // Drop the plan reply without assuming the remote planner stops
    if (plan_request_) {plan_client_->remove_pending_request(*plan_request_); plan_request_.reset();}
  }

  // Preserve the motion outcome while waiting for confirmed stillness
  void stop(std::uint8_t code, const std::string & message)
  {
    // Leave terminal states unchanged when another stop arrives
    if (state_.get() == State::faulted || state_.get() == State::idle) {return;}
    // Keep an ongoing stop within its original confirmation deadline
    if (state_.get() == State::stopping) {
      // Let accepted cancellation replace an outcome that remains pending
      if (code == Motion::Result::CANCELED) {
        // Save the reason that initiated this stop
        result_code_ = code;
        result_message_ = message;
      }
      return;
    }
    // Save the reason that initiated this stop
    result_code_ = code;
    result_message_ = message;
    // Enter stopping before releasing queries or canceling motion
    transition(Event::stop);
    clear_requests();
    // Bound the wait for stop confirmation
    deadline_ns_ = Clock::now() + handshake_timeout_s;
    // Require new feedback and ask any active controller goal to stop
    stationary_samples_ = 0;
    cancel_child();
  }

  // Publish a terminal result only after evaluating stop confirmation
  void finish(bool confirmed)
  {
    // Return stop evidence alongside the reported motion outcome
    auto result = std::make_shared<Motion::Result>();
    result->stop_confirmed = confirmed;
    // Latch failure when controller completion or stillness is unproven
    if (!confirmed) {
      result_code_ = Motion::Result::STOP_FAILED;
      result_message_ += "; stop could not be confirmed; restart required after inspection";
    } else if (result_code_ == Motion::Result::SUCCESS &&
      (!reached(actual_, target_, group_) ||
      !reached(actual_, start_, group_ == Group::arm ? Group::rail : Group::arm)))
    {
      // Reject success when the target or unchanged group misses tolerance
      result_code_ = Motion::Result::EXECUTION_FAILED;
      result_message_ = "Controller reported success but final joint feedback missed the target";
    } else if (result_code_ == Motion::Result::SUCCESS) {
      // Describe success only after endpoint and stop checks pass
      result_message_ = "Mock motion completed; target reached and stop confirmed";
    }
    // Copy the final outcome into the action response
    result->code = result_code_;
    result->message = result_message_;
    // Release ownership on confirmed stop or latch the fault state
    transition(confirmed ? Event::confirmed : Event::stop_failed);
    // Log the terminal code and reason for later diagnosis
    RCLCPP_INFO(get_logger(), "Motion result %u: %s", static_cast<unsigned>(result->code), result->message.c_str());
    // Complete the action with success, cancellation, or failure status
    if (result_code_ == Motion::Result::SUCCESS) {
      goal_->succeed(result);
    } else if (result_code_ == Motion::Result::CANCELED && goal_->is_canceling()) {
      goal_->canceled(result);
    } else {
      goal_->abort(result);
    }
    // Release the parent handle after publishing its terminal result
    goal_.reset();
  }

  // Stop the request with the failure that matches the expired phase
  void time_out()
  {
    // Report missing dependencies when readiness checks never finish
    if (state_.get() == State::checking) {
      stop(Motion::Result::UNAVAILABLE,
        "Preflight timed out: mock hardware, controllers, planner, or feedback unavailable");
    } else if (state_.get() == State::planning) {
      // Report a planner that did not answer within its time budget
      stop(Motion::Result::PLANNING_FAILED, "Planning timed out");
    } else {
      // Report a controller that did not accept or finish in time
      stop(Motion::Result::EXECUTION_FAILED, "Controller acceptance or execution timed out");
    }
  }

  // Advance readiness checks and enforce execution and stop deadlines
  void tick()
  {
    // Skip periodic work when no motion request is active
    if (!goal_) {return;}
    // Publish the current phase and elapsed time to the caller
    auto feedback = std::make_shared<Motion::Feedback>();
    feedback->state = state_.name();
    feedback->elapsed_s = std::chrono::duration<double>(Clock::now() - started_ns_).count();
    goal_->publish_feedback(feedback);
    // Keep an ongoing stop within its original confirmation deadline
    if (state_.get() == State::stopping) {
      // Require a terminal controller and fresh stillness before release
      if ((!dispatched_ || child_terminal_) && fresh() && stationary_samples_ >= still_samples &&
        Clock::now() - stationary_since_ns_ >= still_window_ms)
      {
        finish(true);
      } else if (Clock::now() >= deadline_ns_) {
        // Latch a fault when stop confirmation exceeds its deadline
        finish(false);
      }
      return;
    }
    // End the current phase when its allowed response time expires
    if (Clock::now() >= deadline_ns_) {
      time_out();
      return;
    }
    // Advance readiness checks until planning can begin
    if (state_.get() == State::checking) {
      // Start readiness queries once per request
      if (!preflight_sent_) {preflight();}
      // Plan once live dependencies and measured state are ready
      if (hardware_ready_ && controllers_ready_ && fresh()) {
        transition(Event::ready);
        plan();
      }
      return;
    }
    // Wait for the asynchronous planner reply
    if (state_.get() == State::planning) {return;}
    // Cancel execution when feedback can no longer support monitoring
    if (!fresh()) {
      stop(Motion::Result::EXECUTION_FAILED, "Joint feedback missing, invalid, or stale");
      return;
    }
    // Detect motion outside the group this request owns
    const auto other_group = group_ == Group::arm ? Group::rail : Group::arm;
    if (!reached(actual_, start_, other_group)) {
      stop(Motion::Result::EXECUTION_FAILED, "Uncommanded group moved");
    }
  }

  // Keep all mutable motion state on the single executor thread
  MotionState state_;
  bool shutting_down_{false};
  // Share immutable model data across validation operations
  moveit::core::RobotModelConstPtr model_;
  // Retain ROS interfaces and goal handles for callback lifetimes
  rclcpp_action::Server<Motion>::SharedPtr server_;
  std::shared_ptr<MotionHandle> goal_;
  rclcpp_action::Client<Trajectory>::SharedPtr arm_client_;
  rclcpp_action::Client<Trajectory>::SharedPtr rail_client_;
  TrajectoryHandle::SharedPtr child_;
  rclcpp::Client<Hardware>::SharedPtr hardware_client_;
  rclcpp::Client<Controllers>::SharedPtr controllers_client_;
  rclcpp::Client<Plan>::SharedPtr plan_client_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  // Track pending service requests so stopping can release them
  std::optional<std::int64_t> hardware_request_;
  std::optional<std::int64_t> controllers_request_;
  std::optional<std::int64_t> plan_request_;
  // Keep measured, planned, and target positions separate
  Group group_{Group::arm};
  Positions actual_;
  Positions start_;
  Positions target_;
  Positions stationary_anchor_;
  // Use monotonic deadlines and ROS timestamps for distinct checks
  Clock::time_point last_sample_ns_{};
  Clock::time_point stationary_since_ns_{};
  Clock::time_point started_ns_{};
  Clock::time_point deadline_ns_{};
  rclcpp::Time last_stamp_ns_{0, 0, RCL_ROS_TIME};
  // Track stop evidence and reject callbacks from earlier requests
  std::uint64_t stationary_samples_{0};
  std::uint64_t generation_{0};
  // Track readiness and dispatch progress without blocking callbacks
  bool feedback_valid_{false};
  bool hardware_ready_{false};
  bool controllers_ready_{false};
  bool preflight_sent_{false};
  bool dispatched_{false};
  bool child_terminal_{false};
  // Retain the pending result until stop confirmation finishes
  std::uint8_t result_code_{Motion::Result::UNAVAILABLE};
  std::string result_message_;
};
}

namespace
{
// Keep signal handling limited to a signal-safe shutdown flag
volatile std::sig_atomic_t shutdown_requested = 0;
// Request orderly shutdown without calling ROS from a signal handler
void signal_shutdown(int) { shutdown_requested = 1; }
}

// Run one callback thread and keep ROS alive while motion stops
int main(int argc, char ** argv)
{
  // Disable default shutdown handling so cancellation can finish first
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  // Route termination signals through the orderly shutdown path
  std::signal(SIGINT, signal_shutdown);
  std::signal(SIGTERM, signal_shutdown);
  // Default to success unless startup or execution raises an exception
  int exit_code = 0;
  try {
    // Construct shared node ownership before connecting its interfaces
    const auto node = std::make_shared<agrobot_motion::MotionNode>();
    node->start();
    // Serialize callbacks so motion ownership needs no shared-data locks
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    // Track whether a signal already started shutdown
    bool stopping = false;
    // Keep processing ROS events while shutdown can still make progress
    while (rclcpp::ok()) {
      // Begin shutdown once when a termination signal arrives
      if (shutdown_requested != 0 && !stopping) {
        stopping = true;
        node->shutdown_motion();
      }
      // Service action results and timers so stopping can complete
      executor.spin_some();
      // Exit after the active goal reaches a terminal result
      if (stopping && !node->has_goal()) {break;}
      // Yield between callback batches to avoid busy polling
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  } catch (const std::exception & error) {
    // Report unexpected exceptions and return a failing process status
    RCLCPP_FATAL(rclcpp::get_logger("agrobot_motion"), "%s", error.what());
    exit_code = 1;
  }
  // Release ROS resources after motion processing ends
  rclcpp::shutdown();
  // Return process status so launch tools can detect failure
  return exit_code;
}
