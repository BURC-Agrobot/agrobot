#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <controller_manager_msgs/srv/list_controllers.hpp>
#include <controller_manager_msgs/srv/list_hardware_components.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "agrobot_motion/action/move_joints.hpp"

namespace agrobot_motion
{
// Represent motion phases so terminal faults remain distinct from idle
enum class State { idle, checking, planning, executing, stopping, faulted };
// Name the only events allowed to change motion ownership
enum class Event { request, ready, dispatch, stop, confirmed, stop_failed };
// Keep arm and rail selection separate from raw action values
enum class Group { arm, rail };
// Wrap angular positions so their units remain explicit
struct Radians { double value_rad{}; };
// Wrap linear positions so they cannot be confused with arm angles
struct Metres { double value_m{}; };
// Keep a complete robot pose with separate angular and linear units
struct Positions
{
  // Store arm positions in controller joint order
  std::array<Radians, 6> arm{};
  // Store the rail position independently from arm angles
  Metres rail{};
};

// Share the same transition rules between execution and model checks
class MotionState
{
public:
  // Apply a legal transition and reject all other state-event pairs
  bool apply(Event event);
  // Read the current phase without changing ownership
  State get() const { return state_; }
  // Return the phase label used by logs and action feedback
  const char * name() const;
private:
  // Start idle so the first valid request can reserve ownership
  State state_{State::idle};
};

// Define the expected arm joint order for messages and validation
inline const std::vector<std::string> arm_joints{
  "arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5", "arm_joint_6"};
// Keep the rail joint set separate from the arm
inline const std::vector<std::string> rail_joints{"rail_joint"};
// Set the angular endpoint tolerance for mock completion checks
inline constexpr double arm_tolerance_rad = 0.002;
// Set the linear endpoint tolerance for mock completion checks
inline constexpr double rail_tolerance_m = 0.0002;
// Limit planned speed and acceleration to this fraction of model limits
inline constexpr double motion_scale = 0.1;
// Reject plans longer than this so execution deadlines stay bounded
inline constexpr double max_trajectory_duration_s = 180.0;
// Absorb planner rounding at position bounds in radians or metres
inline constexpr double position_slack_si = 1e-8;
// Absorb planner rounding at speed, acceleration, and final-stop checks
inline constexpr double rate_slack_si = 1e-6;

// Return the expected joint names for one selected group
const std::vector<std::string> & joint_names(Group group);
// Return arm radians or rail metres in the selected joint order
std::vector<double> positions_si(const Positions & positions, Group group);
// Convert an action target into typed robot positions
Positions target_positions(const action::MoveJoints::Goal & goal);
// Check group selection, finite targets, and model position limits
bool valid_goal(const action::MoveJoints::Goal & goal, const moveit::core::RobotModel & model);
// Validate feedback and update positions only for a complete sample
bool read_positions(const sensor_msgs::msg::JointState & message, Positions & positions);
// Compare both groups using their separate endpoint tolerances
bool near(const Positions & left, const Positions & right);
// Check whether one group reaches its requested positions
bool reached(const Positions & actual, const Positions & target, Group group);
// Check required groups, joint order, and finite motion limits
bool valid_model(const moveit::core::RobotModel & model);
// Check that the active component is the supported mock system
bool mock_hardware(const controller_manager_msgs::srv::ListHardwareComponents::Response & response);
// Check active controllers and their exact position-interface claims
bool controllers_ready(const controller_manager_msgs::srv::ListControllers::Response & response);
// Return a reason if planner output violates the motion contract
std::string validate_trajectory(
  const trajectory_msgs::msg::JointTrajectory & trajectory, Group group,
  const Positions & start, const Positions & target, const moveit::core::RobotModel & model);
}
