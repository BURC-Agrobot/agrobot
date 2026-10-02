#include "agrobot_motion/motion_rules.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <rclcpp/duration.hpp>

namespace agrobot_motion
{
// Allow only transitions that preserve single-request motion ownership
bool MotionState::apply(Event event)
{
  // Reserve idle ownership before checking any motion dependencies
  if (state_ == State::idle && event == Event::request) {
    state_ = State::checking;
  } else if (state_ == State::checking && event == Event::ready) {
    // Start planning only after hardware, controllers, and feedback pass
    state_ = State::planning;
  } else if (state_ == State::planning && event == Event::dispatch) {
    // Mark dispatch so later stopping must account for the controller
    state_ = State::executing;
  } else if ((state_ == State::checking || state_ == State::planning || state_ == State::executing) &&
    event == Event::stop)
  {
    // Require stop confirmation from every active phase
    state_ = State::stopping;
  } else if (state_ == State::stopping && event == Event::confirmed) {
    // Release ownership only after the stop is confirmed
    state_ = State::idle;
  } else if (state_ == State::stopping && event == Event::stop_failed) {
    // Latch an unconfirmed stop so another request cannot begin
    state_ = State::faulted;
  } else {
    return false;
  }
  return true;
}

// Expose readable state names for logs and action feedback
const char * MotionState::name() const
{
  // Map every supported state to its public feedback label
  switch (state_) {
    case State::idle: return "IDLE";
    case State::checking: return "CHECKING";
    case State::planning: return "PLANNING";
    case State::executing: return "EXECUTING";
    case State::stopping: return "STOPPING";
    case State::faulted: return "FAULTED";
  }
  // Use the fault label if an invalid state reaches this boundary
  return "FAULTED";
}

// Select the expected joint order for the requested group
const std::vector<std::string> & joint_names(Group group)
{
  return group == Group::arm ? arm_joints : rail_joints;
}

// Convert typed positions to ROS values in radians or metres
std::vector<double> positions_si(const Positions & positions, Group group)
{
  // Handle the rail separately because its position uses metres
  if (group == Group::rail) {
    return {positions.rail.value_m};
  }
  // Collect arm angles in the configured controller joint order
  std::vector<double> result_rad;
  // Copy each arm angle without changing its radian units
  for (const auto angle : positions.arm) {
    result_rad.push_back(angle.value_rad);
  }
  // Return the ordered arm positions for ROS message fields
  return result_rad;
}

// Convert the action target into separate angular and linear types
Positions target_positions(const action::MoveJoints::Goal & goal)
{
  // Initialize all target positions before copying request fields
  Positions target;
  // Copy each arm target while preserving joint order
  for (std::size_t index = 0; index < target.arm.size(); ++index) {
    target.arm[index].value_rad = goal.arm_positions_rad[index];
  }
  // Copy the rail target without mixing metres and radians
  target.rail.value_m = goal.rail_position_m;
  // Return the complete typed target for later group selection
  return target;
}

// Require the expected groups and usable limits before motion starts
bool valid_model(const moveit::core::RobotModel & model)
{
  // Check both groups so the complete robot contract is covered
  for (const auto group : {Group::arm, Group::rail}) {
    // Look up the group in the shared robot model
    const auto * joint_group = model.getJointModelGroup(group == Group::arm ? "arm" : "rail");
    // Reject missing groups or mismatched controller joint ordering
    if (joint_group == nullptr || joint_group->getVariableNames() != joint_names(group)) {
      return false;
    }
    // Check every configured joint limit before trusting the model
    for (const auto & name : joint_names(group)) {
      // Read limits from the model instead of duplicating configuration
      const auto & bounds = model.getVariableBounds(name);
      // Require finite position bounds and positive motion limits
      if (!bounds.position_bounded_ || !bounds.velocity_bounded_ || !bounds.acceleration_bounded_ ||
        !std::isfinite(bounds.min_position_) || !std::isfinite(bounds.max_position_) ||
        bounds.min_position_ >= bounds.max_position_ ||
        !std::isfinite(bounds.max_velocity_) || bounds.max_velocity_ <= 0.0 ||
        !std::isfinite(bounds.max_acceleration_) || bounds.max_acceleration_ <= 0.0)
      {
        return false;
      }
    }
  }
  // Reject extra model variables outside the supported seven joints
  return model.getVariableCount() == 7;
}

// Reject unsupported, non-finite, or out-of-bounds action targets
bool valid_goal(const action::MoveJoints::Goal & goal, const moveit::core::RobotModel & model)
{
  // Allow only the arm and rail group identifiers
  if (goal.group != action::MoveJoints::Goal::ARM && goal.group != action::MoveJoints::Goal::RAIL) {
    return false;
  }
  // Reject non-finite target values before comparing limits
  if (!std::isfinite(goal.rail_position_m) ||
    !std::all_of(goal.arm_positions_rad.begin(), goal.arm_positions_rad.end(),
    [](double position_rad) {return std::isfinite(position_rad);}))
  {
    return false;
  }
  // Select the target unit and joint order from the requested group
  const auto group = goal.group == action::MoveJoints::Goal::ARM ? Group::arm : Group::rail;
  // Require unused fields to stay zero so a request names one group
  if ((group == Group::arm && goal.rail_position_m != 0.0) ||
    (group == Group::rail && std::any_of(goal.arm_positions_rad.begin(), goal.arm_positions_rad.end(),
    [](double position_rad) {return position_rad != 0.0;})))
  {
    return false;
  }
  // Read selected targets in radians for arm or metres for rail
  const auto target_si = positions_si(target_positions(goal), group);
  // Use the configured joint order when indexing position arrays
  const auto & names = joint_names(group);
  // Check every joint so a partial match cannot validate the request
  for (std::size_t index = 0; index < names.size(); ++index) {
    // Read this joint’s limits from the authoritative model
    const auto & bounds = model.getVariableBounds(names[index]);
    // Reject targets outside the joint’s configured position bounds
    if (target_si[index] < bounds.min_position_ || target_si[index] > bounds.max_position_) {
      return false;
    }
  }
  return true;
}

// Validate and stage feedback before replacing trusted positions
bool read_positions(const sensor_msgs::msg::JointState & message, Positions & positions)
{
  // Reject mismatched arrays and duplicate joint names before indexing
  if (message.name.size() != message.position.size() ||
    (!message.velocity.empty() && message.velocity.size() != message.name.size()) ||
    (!message.effort.empty() && message.effort.size() != message.name.size()) ||
    std::set<std::string>(message.name.begin(), message.name.end()).size() != message.name.size())
  {
    return false;
  }
  // Require finite positions while allowing unused effort to be unknown
  if (!std::all_of(message.position.begin(), message.position.end(),
    [](double position_si) {return std::isfinite(position_si);}))
  {
    return false;
  }
  // Stage all joints so rejected feedback leaves trusted data unchanged
  Positions sample;
  // Check both groups so the complete robot contract is covered
  for (const auto group : {Group::arm, Group::rail}) {
    // Select required joint names independently of message ordering
    const auto & names = joint_names(group);
    // Resolve every required joint by name before reading its position
    for (std::size_t index = 0; index < names.size(); ++index) {
      // Find the expected joint rather than assuming publisher ordering
      const auto found = std::find(message.name.begin(), message.name.end(), names[index]);
      // Reject samples missing any required arm or rail joint
      if (found == message.name.end()) {
        return false;
      }
      // Convert the matched position to a checked array offset
      const auto offset = static_cast<std::size_t>(std::distance(message.name.begin(), found));
      // Store arm feedback in radians to preserve angular units
      if (group == Group::arm) {
        sample.arm[index].value_rad = message.position[offset];
      } else {
        // Store rail feedback in metres to preserve linear units
        sample.rail.value_m = message.position[offset];
      }
    }
  }
  // Commit the sample only after all required positions are valid
  positions = sample;
  return true;
}

// Compare one group’s positions using its unit-specific tolerance
bool reached(const Positions & actual, const Positions & target, Group group)
{
  // Handle the rail separately because its position uses metres
  if (group == Group::rail) {
    return std::abs(actual.rail.value_m - target.rail.value_m) <= rail_tolerance_m;
  }
  // Require every arm joint to meet the angular tolerance
  for (std::size_t index = 0; index < actual.arm.size(); ++index) {
    // Reject the group when any arm joint misses its target
    if (std::abs(actual.arm[index].value_rad - target.arm[index].value_rad) > arm_tolerance_rad) {
      return false;
    }
  }
  return true;
}

// Require both groups to match when checking a complete robot pose
bool near(const Positions & left, const Positions & right)
{
  return reached(left, right, Group::arm) && reached(left, right, Group::rail);
}

// Accept only the expected active mock hardware component
bool mock_hardware(const controller_manager_msgs::srv::ListHardwareComponents::Response & response)
{
  // Reject missing or multiple hardware components
  if (response.component.size() != 1) {
    return false;
  }
  // Inspect the sole component’s identity and active state
  const auto & hardware = response.component.front();
  return hardware.name == "FakeSystem" && hardware.plugin_name == "mock_components/GenericSystem" &&
         hardware.state.label == "active";
}

// Require both controllers to own exactly their position interfaces
bool controllers_ready(const controller_manager_msgs::srv::ListControllers::Response & response)
{
  // Check both groups so the complete robot contract is covered
  for (const auto group : {Group::arm, Group::rail}) {
    // Select the controller belonging to this joint group
    const std::string controller_name = group == Group::arm ? "arm_controller" : "rail_controller";
    // Find the controller by name instead of relying on response order
    const auto found = std::find_if(response.controller.begin(), response.controller.end(),
      [&controller_name](const auto & controller) {return controller.name == controller_name;});
    // Reject missing, inactive, or unsupported controllers
    if (found == response.controller.end() || found->state != "active" ||
      found->type != "joint_trajectory_controller/JointTrajectoryController")
    {
      return false;
    }
    // Build the exact position-interface set required for this group
    std::set<std::string> expected;
    // Include each joint’s position command interface
    for (const auto & joint : joint_names(group)) {
      expected.insert(joint + "/position");
    }
    // Reject missing or extra claims that change command ownership
    if (std::set<std::string>(found->claimed_interfaces.begin(), found->claimed_interfaces.end()) != expected) {
      return false;
    }
  }
  return true;
}

// Validate planner output before sending any controller command
std::string validate_trajectory(
  const trajectory_msgs::msg::JointTrajectory & trajectory, Group group,
  const Positions & start, const Positions & target, const moveit::core::RobotModel & model)
{
  // Use the configured joint order when indexing position arrays
  const auto & names = joint_names(group);
  // Require the complete expected joint order and at least one point
  if (trajectory.joint_names != names || trajectory.points.empty()) {
    return "Planner returned missing, extra, or reordered joints, or no points";
  }
  // Use the measured start pose to validate the first trajectory point
  const auto start_si = positions_si(start, group);
  // Read selected targets in radians for arm or metres for rail
  const auto target_si = positions_si(target, group);
  // Choose endpoint tolerance in radians or metres for this group
  const double tolerance_si = group == Group::arm ? arm_tolerance_rad : rail_tolerance_m;
  // Allow an initial zero timestamp while enforcing increasing time
  double previous_time_s = -1.0;
  // Validate every point before accepting the whole trajectory
  for (const auto & point : trajectory.points) {
    // Convert relative point timing to seconds for deadline checks
    const double time_s = rclcpp::Duration(point.time_from_start).seconds();
    // Reject invalid timing, oversized duration, and mismatched arrays
    if (point.time_from_start.sec < 0 || point.time_from_start.nanosec >= 1000000000U ||
      time_s <= previous_time_s || time_s > max_trajectory_duration_s ||
      point.positions.size() != names.size() || point.velocities.size() != names.size() ||
      point.accelerations.size() != names.size() || !point.effort.empty())
    {
      return "Planner returned invalid timing or point dimensions";
    }
    // Advance the timing reference only after this point is valid
    previous_time_s = time_s;
    // Check ordered joint values against their model limits
    for (std::size_t index = 0; index < names.size(); ++index) {
      // Read this joint’s model limits for the trajectory check
      const auto & bounds = model.getVariableBounds(names[index]);
      // Enforce finite values and the scaled model motion limits
      if (!std::isfinite(point.positions[index]) || !std::isfinite(point.velocities[index]) ||
        !std::isfinite(point.accelerations[index]) ||
        point.positions[index] < bounds.min_position_ - position_slack_si ||
        point.positions[index] > bounds.max_position_ + position_slack_si ||
        std::abs(point.velocities[index]) > bounds.max_velocity_ * motion_scale + rate_slack_si ||
        std::abs(point.accelerations[index]) > bounds.max_acceleration_ * motion_scale + rate_slack_si)
      {
        return "Planner returned non-finite values or exceeded scaled configured limits";
      }
    }
  }
  // Check every joint so a partial match cannot validate the request
  for (std::size_t index = 0; index < names.size(); ++index) {
    // Require the measured start, requested endpoint, and zero end speed
    if (std::abs(trajectory.points.front().positions[index] - start_si[index]) > tolerance_si ||
      std::abs(trajectory.points.back().positions[index] - target_si[index]) > tolerance_si ||
      std::abs(trajectory.points.back().velocities[index]) > rate_slack_si)
    {
      return "Planner returned an unexpected start, endpoint, or nonzero final velocity";
    }
  }
  // Return no error after every trajectory check passes
  return {};
}
}
