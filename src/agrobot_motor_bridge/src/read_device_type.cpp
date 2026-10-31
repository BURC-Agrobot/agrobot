#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>

#include "agrobot_motor_bridge/can_bus.hpp"
#include "agrobot_motor_bridge/sdo_client.hpp"
#include "agrobot_motor_bridge/test_program.hpp"

int main(int argc, char ** argv)
{
  try {
    const auto arguments = agrobot_motor_bridge::parse_bus_arguments(argc, argv);

    // Read Device Type (0x1000), which every CANopen device must have
    const agrobot_motor_bridge::CanBus bus(arguments.interface_name);
    const agrobot_motor_bridge::SdoClient sdo(bus, arguments.node_id);
    const std::uint32_t device_type = sdo.read(0x1000, 0x00);
    std::printf(
      "[read_device_type] Node %d Device Type = 0x%08X\n", arguments.node_id, device_type);
  } catch (const std::exception & error) {
    // Report the failed call instead of ending with an uncaught exception
    std::cerr << "[read_device_type] ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
