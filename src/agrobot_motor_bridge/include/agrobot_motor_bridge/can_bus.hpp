#pragma once

#include <chrono>
#include <optional>
#include <string>

#include <linux/can.h>

namespace agrobot_motor_bridge
{
// Own one raw CAN socket so it closes on every exit path
class CanBus
{
public:
  // Open a socket and attach it to the named interface, or throw
  explicit CanBus(const std::string & interface_name);
  // Release the socket
  ~CanBus();
  // Forbid copies so only one object owns the socket
  CanBus(const CanBus &) = delete;
  CanBus & operator=(const CanBus &) = delete;

  // Hand one frame to the kernel, or throw
  void send(const can_frame & frame) const;
  // Wait for one frame, returning nothing if the timeout passes first, or throw
  std::optional<can_frame> receive(std::chrono::milliseconds timeout) const;

private:
  // Store the kernel's handle for the open socket
  int socket_fd_{-1};
};
}  // namespace agrobot_motor_bridge
