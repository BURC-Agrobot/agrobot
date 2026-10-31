#include "agrobot_motor_bridge/can_bus.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace agrobot_motor_bridge
{
namespace
{
// Build an error naming the failed call and the system's reason
std::runtime_error system_error(const std::string & call)
{
  return std::runtime_error(call + " failed: " + std::strerror(errno));
}

// Close a socket whose setup failed, because the destructor will not run
[[noreturn]] void close_and_throw(int socket_fd, const std::string & call)
{
  // Build the error first, because close can change errno
  const std::runtime_error error = system_error(call);
  close(socket_fd);
  throw error;
}

// The kernel moves a whole frame or fails, so treat anything else as an error
void require_whole_frame(ssize_t transferred, const std::string & call)
{
  if (transferred < 0) {
    throw system_error(call);
  }
  if (static_cast<size_t>(transferred) != sizeof(can_frame)) {
    throw std::runtime_error(call + " moved a partial CAN frame");
  }
}
}  // namespace

CanBus::CanBus(const std::string & interface_name)
{
  // Create a CAN socket that is not yet tied to a bus
  socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (socket_fd_ < 0) {
    throw system_error("socket");
  }

  // Turn the interface name into the number the kernel uses
  const unsigned int interface_index = if_nametoindex(interface_name.c_str());
  if (interface_index == 0) {
    close_and_throw(socket_fd_, "if_nametoindex(" + interface_name + ")");
  }

  // Attach the socket to that bus
  sockaddr_can address{};
  address.can_family = AF_CAN;
  address.can_ifindex = static_cast<int>(interface_index);
  if (bind(socket_fd_, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0) {
    close_and_throw(socket_fd_, "bind(" + interface_name + ")");
  }
}

CanBus::~CanBus()
{
  close(socket_fd_);
}

void CanBus::send(const can_frame & frame) const
{
  require_whole_frame(write(socket_fd_, &frame, sizeof(frame)), "write");
}

std::optional<can_frame> CanBus::receive(std::chrono::milliseconds timeout) const
{
  // Sleep until a frame is waiting or the timeout passes
  pollfd waiting{};
  waiting.fd = socket_fd_;
  waiting.events = POLLIN;
  const int ready = poll(&waiting, 1, static_cast<int>(timeout.count()));
  if (ready < 0) {
    throw system_error("poll");
  }
  if (ready == 0) {
    return std::nullopt;
  }

  can_frame frame{};
  require_whole_frame(read(socket_fd_, &frame, sizeof(frame)), "read");
  return frame;
}
}  // namespace agrobot_motor_bridge
