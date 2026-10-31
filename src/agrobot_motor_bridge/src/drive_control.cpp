#include "agrobot_motor_bridge/drive_control.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace agrobot_motor_bridge
{
namespace
{
// Name the objects this layer uses
constexpr std::uint16_t kControlword = 0x6040;
constexpr std::uint16_t kStatusword = 0x6041;
constexpr std::uint16_t kModeOfOperation = 0x6060;
constexpr std::uint16_t kModeDisplay = 0x6061;
constexpr std::uint16_t kPositionActual = 0x6064;
constexpr std::uint16_t kTargetPosition = 0x607A;
constexpr std::uint16_t kProfileVelocity = 0x6081;
constexpr std::uint16_t kProfileAcceleration = 0x6083;
constexpr std::uint16_t kProfileDeceleration = 0x6084;
// Name Profile Position Mode and its bits from Tables 5-16 and 5-19
constexpr std::uint32_t kProfilePositionMode = 1;
constexpr std::uint16_t kNewSetpoint = 0x001F;
constexpr std::uint16_t kTargetReachedBit = 0x0400;
constexpr std::uint16_t kSetpointAcknowledgeBit = 0x1000;
// Limit how long a move may take, so a stalled drive cannot hang the caller
constexpr std::chrono::seconds kMoveTimeout{10};
// Name the commands from EPOS2 Firmware Specification Table 3-6
constexpr std::uint16_t kShutdown = 0x0006;
constexpr std::uint16_t kSwitchOn = 0x0007;
constexpr std::uint16_t kEnableOperation = 0x000F;
constexpr std::uint16_t kDisableVoltage = 0x0000;
// Give the drive time to change state, and avoid flooding the bus while it does
constexpr std::chrono::milliseconds kStateTimeout{500};
constexpr std::chrono::milliseconds kPollPeriod{10};

// Format a Statusword and the state it means for error messages
std::string describe(std::uint16_t statusword)
{
  char text[8];
  std::snprintf(text, sizeof(text), "0x%04X", statusword);
  return state_name(decode_state(statusword)) + " (" + text + ")";
}
}  // namespace

DriveState decode_state(std::uint16_t statusword)
{
  // These states are defined by bits 0, 1, 2, 3 and 6
  switch (statusword & 0x004FU) {
    case 0x0000U: return DriveState::not_ready_to_switch_on;
    case 0x0040U: return DriveState::switch_on_disabled;
    case 0x000FU: return DriveState::fault_reaction_active;
    case 0x0008U: return DriveState::fault;
    default: break;
  }
  // These states also need bit 5
  switch (statusword & 0x006FU) {
    case 0x0021U: return DriveState::ready_to_switch_on;
    case 0x0023U: return DriveState::switched_on;
    case 0x0027U: return DriveState::operation_enabled;
    case 0x0007U: return DriveState::quickstop_active;
    default: return DriveState::unknown;
  }
}

std::string state_name(DriveState state)
{
  switch (state) {
    case DriveState::not_ready_to_switch_on: return "Not Ready to Switch On";
    case DriveState::switch_on_disabled: return "Switch On Disabled";
    case DriveState::ready_to_switch_on: return "Ready to Switch On";
    case DriveState::switched_on: return "Switched On";
    case DriveState::operation_enabled: return "Operation Enable";
    case DriveState::quickstop_active: return "Quickstop Active";
    case DriveState::fault_reaction_active: return "Fault Reaction Active";
    case DriveState::fault: return "Fault";
    case DriveState::unknown: break;
  }
  return "Unknown";
}

DriveControl::DriveControl(const SdoClient & sdo)
: sdo_(sdo)
{
}

std::uint16_t DriveControl::statusword() const
{
  return static_cast<std::uint16_t>(sdo_.read(kStatusword, 0x00));
}

void DriveControl::require_switch_on_disabled() const
{
  const DriveState state = decode_state(statusword());
  if (state != DriveState::switch_on_disabled) {
    throw std::runtime_error(
      "Drive must start in Switch On Disabled, but is in " + state_name(state));
  }
}

void DriveControl::shutdown() const
{
  command(kShutdown, DriveState::ready_to_switch_on);
}

void DriveControl::switch_on() const
{
  command(kSwitchOn, DriveState::switched_on);
}

void DriveControl::enable_operation() const
{
  command(kEnableOperation, DriveState::operation_enabled);
}

void DriveControl::disable_voltage() const
{
  command(kDisableVoltage, DriveState::switch_on_disabled);
}

void DriveControl::set_profile_position_mode(
  std::uint32_t velocity_rpm, std::uint32_t acceleration_rpm_s,
  std::uint32_t deceleration_rpm_s) const
{
  // Confirm the mode through its display, because the drive decides when it applies
  sdo_.write(kModeOfOperation, 0x00, kProfilePositionMode, 1);
  if (sdo_.read(kModeDisplay, 0x00) != kProfilePositionMode) {
    throw std::runtime_error("Drive did not change to Profile Position Mode");
  }
  sdo_.write(kProfileVelocity, 0x00, velocity_rpm, 4);
  sdo_.write(kProfileAcceleration, 0x00, acceleration_rpm_s, 4);
  sdo_.write(kProfileDeceleration, 0x00, deceleration_rpm_s, 4);
}

void DriveControl::move_to(std::int32_t target_counts) const
{
  // Send the target, then raise the new setpoint bit to start the move
  sdo_.write(kTargetPosition, 0x00, static_cast<std::uint32_t>(target_counts), 4);
  sdo_.write(kControlword, 0x00, kNewSetpoint, 2);

  // Wait for the drive to take the setpoint before lowering the bit
  wait_during_move(kSetpointAcknowledgeBit, kStateTimeout, "Drive did not acknowledge the new setpoint");
  sdo_.write(kControlword, 0x00, kEnableOperation, 2);

  // Target reached is only meaningful once the setpoint has been taken
  wait_during_move(kTargetReachedBit, kMoveTimeout, "Drive did not reach the target in time");
}

std::int32_t DriveControl::position_counts() const
{
  return static_cast<std::int32_t>(sdo_.read(kPositionActual, 0x00));
}

void DriveControl::wait_during_move(
  std::uint16_t status_bit, std::chrono::milliseconds timeout, const std::string & timeout_error) const
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    const std::uint16_t status = statusword();
    if (decode_state(status) != DriveState::operation_enabled) {
      throw std::runtime_error("Drive reported " + describe(status) + " during a move");
    }
    if ((status & status_bit) != 0) {
      return;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(timeout_error);
    }
    std::this_thread::sleep_for(kPollPeriod);
  }
}

void DriveControl::command(std::uint16_t controlword, DriveState expected) const
{
  sdo_.write(kControlword, 0x00, controlword, 2);

  // A drive can take time to change state, so poll until it reports the new one
  const auto deadline = std::chrono::steady_clock::now() + kStateTimeout;
  while (true) {
    const std::uint16_t status = statusword();
    const DriveState state = decode_state(status);
    if (state == expected) {
      return;
    }
    // Stop at once on a fault, because waiting cannot clear it
    if (state == DriveState::fault || state == DriveState::fault_reaction_active) {
      throw std::runtime_error(
        "Drive reported " + describe(status) + " while going to " + state_name(expected));
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(
        "Drive stayed in " + describe(status) + " instead of going to " + state_name(expected));
    }
    std::this_thread::sleep_for(kPollPeriod);
  }
}
}  // namespace agrobot_motor_bridge
