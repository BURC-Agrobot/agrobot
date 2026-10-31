#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

#include "agrobot_motor_bridge/can_bus.hpp"

namespace
{
// Give a person time to send a frame from another terminal
constexpr std::chrono::seconds kTimeout{5};

// Print a frame in the same layout as candump
void print_frame(const can_frame & frame)
{
  std::printf("[receive_frame] %03X  [%u] ", frame.can_id & CAN_EFF_MASK, frame.len);
  for (unsigned int i = 0; i < frame.len; ++i) {
    std::printf(" %02X", frame.data[i]);
  }
  std::printf("\n");
}
}  // namespace

int main(int argc, char ** argv)
{
  // Use the virtual bus unless another interface is named
  const std::string interface_name = argc > 1 ? argv[1] : "vcan0";

  try {
    // Open the bus and wait for one frame
    const agrobot_motor_bridge::CanBus bus(interface_name);
    std::cout << "[receive_frame] Waiting " << kTimeout.count() << " s on " << interface_name
              << std::endl;
    const std::optional<can_frame> frame = bus.receive(kTimeout);
    if (!frame) {
      // Report silence as a failure so scripts can detect it
      std::cerr << "[receive_frame] ERROR: Timed out with no frame\n";
      return 1;
    }
    print_frame(*frame);
  } catch (const std::exception & error) {
    // Report the failed call instead of ending with an uncaught exception
    std::cerr << "[receive_frame] ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
