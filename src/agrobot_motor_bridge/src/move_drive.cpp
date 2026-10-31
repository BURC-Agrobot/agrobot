#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "agrobot_motor_bridge/can_bus.hpp"
#include "agrobot_motor_bridge/drive_control.hpp"
#include "agrobot_motor_bridge/sdo_client.hpp"
#include "agrobot_motor_bridge/test_program.hpp"

namespace
{
using agrobot_motor_bridge::DriveControl;

const char * const kProgram = "move_drive";

// Use the profile and target from the emulator's startup sequence
constexpr std::uint32_t kVelocityRpm = 1000;
constexpr std::uint32_t kAccelerationRpmS = 10000;
constexpr std::uint32_t kDecelerationRpmS = 10000;
constexpr std::int32_t kDefaultTargetCounts = 20000;

// Enable the drive, make one move, and take power off again
void move(const DriveControl & drive, std::int32_t target_counts)
{
  drive.require_switch_on_disabled();
  agrobot_motor_bridge::print_state(kProgram, drive);
  drive.set_profile_position_mode(kVelocityRpm, kAccelerationRpmS, kDecelerationRpmS);

  agrobot_motor_bridge::power_off_on_error(kProgram, drive, [&drive, target_counts]() {
    drive.shutdown();
    drive.switch_on();
    drive.enable_operation();
    agrobot_motor_bridge::print_state(kProgram, drive);

    std::printf("[%s] Moving to %d\n", kProgram, target_counts);
    drive.move_to(target_counts);
    const std::int32_t position = drive.position_counts();
    std::printf("[%s] Position = %d\n", kProgram, position);
    if (position != target_counts) {
      throw std::runtime_error("Drive stopped away from the target");
    }

    drive.disable_voltage();
    agrobot_motor_bridge::print_state(kProgram, drive);
  });
}
}  // namespace

int main(int argc, char ** argv)
{
  try {
    const auto arguments = agrobot_motor_bridge::parse_bus_arguments(argc, argv);
    const std::int32_t target_counts = argc > 3 ? std::stoi(argv[3]) : kDefaultTargetCounts;
    agrobot_motor_bridge::require_virtual_bus(arguments.interface_name);

    const agrobot_motor_bridge::CanBus bus(arguments.interface_name);
    const agrobot_motor_bridge::SdoClient sdo(bus, arguments.node_id);
    move(DriveControl(sdo), target_counts);
  } catch (const std::exception & error) {
    // Report the failed call instead of ending with an uncaught exception
    std::cerr << "[" << kProgram << "] ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
