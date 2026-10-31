#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>

#include "agrobot_motor_bridge/can_bus.hpp"
#include "agrobot_motor_bridge/sdo_client.hpp"
#include "agrobot_motor_bridge/test_program.hpp"

namespace
{
// Name the objects this check uses
constexpr std::uint16_t kControlword = 0x6040;
constexpr std::uint16_t kStatusword = 0x6041;
// Shutdown leaves the motor without power, and Disable Voltage returns to the start
constexpr std::uint32_t kShutdown = 0x0006;
constexpr std::uint32_t kDisableVoltage = 0x0000;
// Expect this code when writing a read-only object
constexpr std::uint32_t kReadOnlyAbort = 0x06010002;
}  // namespace

int main(int argc, char ** argv)
{
  try {
    const auto arguments = agrobot_motor_bridge::parse_bus_arguments(argc, argv);
    const agrobot_motor_bridge::CanBus bus(arguments.interface_name);
    const agrobot_motor_bridge::SdoClient sdo(bus, arguments.node_id);

    // Write Controlword, then read it back to prove the value was stored
    sdo.write(kControlword, 0x00, kShutdown, 2);
    const std::uint32_t stored = sdo.read(kControlword, 0x00);
    std::printf("[check_sdo_write] Controlword = 0x%04X\n", stored);
    if (stored != kShutdown) {
      std::cerr << "[check_sdo_write] ERROR: Controlword did not store the written value\n";
      return 1;
    }
    std::printf("[check_sdo_write] Statusword = 0x%04X\n", sdo.read(kStatusword, 0x00));

    // Write read-only Statusword, which the drive must refuse
    bool refused = false;
    try {
      sdo.write(kStatusword, 0x00, 0x0001, 2);
    } catch (const agrobot_motor_bridge::SdoAbort & abort) {
      std::cout << "[check_sdo_write] Refused as expected: " << abort.what() << '\n';
      refused = abort.code() == kReadOnlyAbort;
    }

    // Return the drive to where a fresh start leaves it
    sdo.write(kControlword, 0x00, kDisableVoltage, 2);
    if (!refused) {
      std::cerr << "[check_sdo_write] ERROR: Statusword write was not refused as read-only\n";
      return 1;
    }
  } catch (const std::exception & error) {
    // Report the failed call instead of ending with an uncaught exception
    std::cerr << "[check_sdo_write] ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
