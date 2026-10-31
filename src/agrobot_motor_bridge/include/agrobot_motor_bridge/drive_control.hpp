#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "agrobot_motor_bridge/sdo_client.hpp"

namespace agrobot_motor_bridge
{
// Name the CiA 402 states a drive reports through Statusword
enum class DriveState
{
  not_ready_to_switch_on,
  switch_on_disabled,
  ready_to_switch_on,
  switched_on,
  operation_enabled,
  quickstop_active,
  fault_reaction_active,
  fault,
  unknown
};

// Work out the state from the Statusword bits that define it
DriveState decode_state(std::uint16_t statusword);
// Name a state for messages
std::string state_name(DriveState state);

// Move one drive through its state machine with Controlword commands
class DriveControl
{
public:
  // Use the SDO client without owning it
  explicit DriveControl(const SdoClient & sdo);

  // Read the drive's current Statusword
  std::uint16_t statusword() const;
  // Refuse anything but a clean start, such as a fault that needs a person to look
  void require_switch_on_disabled() const;
  // Each command waits for the drive to report the new state, or throws
  void shutdown() const;
  void switch_on() const;
  // Enable Operation puts power on the motor
  void enable_operation() const;
  // Disable Voltage takes power off the motor
  void disable_voltage() const;

  // Select Profile Position Mode with speeds in rpm and ramps in rpm/s, or throw
  void set_profile_position_mode(
    std::uint32_t velocity_rpm, std::uint32_t acceleration_rpm_s,
    std::uint32_t deceleration_rpm_s) const;
  // Move to an absolute position in quadcounts and wait until it is reached, or throw
  void move_to(std::int32_t target_counts) const;
  // Read the position in quadcounts
  std::int32_t position_counts() const;

private:
  // Send one command and wait for the state it should reach
  void command(std::uint16_t controlword, DriveState expected) const;
  // Wait for a Statusword bit during a move, or throw if the drive stops being enabled or time runs out
  void wait_during_move(
    std::uint16_t status_bit, std::chrono::milliseconds timeout, const std::string & timeout_error) const;

  const SdoClient & sdo_;
};
}  // namespace agrobot_motor_bridge
