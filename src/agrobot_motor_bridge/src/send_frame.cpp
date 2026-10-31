#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "agrobot_motor_bridge/can_bus.hpp"

int main(int argc, char ** argv)
{
  // Use the virtual bus unless another interface is named
  const std::string interface_name = argc > 1 ? argv[1] : "vcan0";

  // Ask node 1 for its Statusword (0x6041), which changes nothing on the drive
  const std::array<std::uint8_t, CAN_MAX_DLEN> data{0x40, 0x41, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00};
  can_frame frame{};
  frame.can_id = 0x601;
  frame.len = CAN_MAX_DLEN;
  std::memcpy(frame.data, data.data(), data.size());

  try {
    // Open the bus and send the frame once
    const agrobot_motor_bridge::CanBus bus(interface_name);
    bus.send(frame);
  } catch (const std::exception & error) {
    // Report the failed call instead of ending with an uncaught exception
    std::cerr << "[send_frame] ERROR: " << error.what() << '\n';
    return 1;
  }

  std::cout << "[send_frame] Sent 601#4041600000000000 on " << interface_name << '\n';
  return 0;
}
