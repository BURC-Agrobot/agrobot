#include <exception>
#include <iostream>

#include "agrobot_motor_bridge/can_bus.hpp"
#include "agrobot_motor_bridge/drive_control.hpp"
#include "agrobot_motor_bridge/sdo_client.hpp"
#include "agrobot_motor_bridge/test_program.hpp"

namespace
{
using agrobot_motor_bridge::DriveControl;

const char * const kProgram = "walk_drive_states";

// Walk up to Operation Enable and back down
void walk(const DriveControl & drive)
{
  drive.require_switch_on_disabled();
  agrobot_motor_bridge::print_state(kProgram, drive);

  agrobot_motor_bridge::power_off_on_error(kProgram, drive, [&drive]() {
    drive.shutdown();
    agrobot_motor_bridge::print_state(kProgram, drive);
    drive.switch_on();
    agrobot_motor_bridge::print_state(kProgram, drive);
    drive.enable_operation();
    agrobot_motor_bridge::print_state(kProgram, drive);
    drive.disable_voltage();
    agrobot_motor_bridge::print_state(kProgram, drive);
  });
}
}  // namespace

int main(int argc, char ** argv)
{
  try {
    const auto arguments = agrobot_motor_bridge::parse_bus_arguments(argc, argv);
    agrobot_motor_bridge::require_virtual_bus(arguments.interface_name);

    const agrobot_motor_bridge::CanBus bus(arguments.interface_name);
    const agrobot_motor_bridge::SdoClient sdo(bus, arguments.node_id);
    walk(DriveControl(sdo));
  } catch (const std::exception & error) {
    // Report the failed call instead of ending with an uncaught exception
    std::cerr << "[" << kProgram << "] ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
