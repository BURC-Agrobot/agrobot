#pragma once

#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "agrobot_motor_bridge/drive_control.hpp"

// Share the pieces that every drive test program needs
namespace agrobot_motor_bridge
{
// Hold which bus and drive a test program talks to
struct BusArguments
{
  std::string interface_name;
  int node_id;
};

// Use the virtual bus and the emulator's node unless others are named
inline BusArguments parse_bus_arguments(int argc, char ** argv)
{
  return {argc > 1 ? argv[1] : "vcan0", argc > 2 ? std::stoi(argv[2]) : 1};
}

// Refuse real buses, because the test programs power the motor with no way to stop it
inline void require_virtual_bus(const std::string & interface_name)
{
  if (interface_name.rfind("vcan", 0) != 0) {
    throw std::invalid_argument("This program only runs on virtual buses, not " + interface_name);
  }
}

// Print the drive's current state
inline void print_state(const std::string & program, const DriveControl & drive)
{
  const std::uint16_t status = drive.statusword();
  std::printf(
    "[%s] 0x%04X %s\n", program.c_str(), status, state_name(decode_state(status)).c_str());
}

// Run steps that power the motor, taking power off if one fails
template<typename Steps>
void power_off_on_error(const std::string & program, const DriveControl & drive, Steps steps)
{
  try {
    steps();
  } catch (const std::exception &) {
    // Keep the first error as the one reported
    try {
      drive.disable_voltage();
    } catch (const std::exception & error) {
      std::cerr << "[" << program << "] ERROR: Could not disable the drive: " << error.what() << '\n';
    }
    throw;
  }
}
}  // namespace agrobot_motor_bridge
