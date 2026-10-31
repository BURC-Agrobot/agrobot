#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "agrobot_motor_bridge/can_bus.hpp"

namespace agrobot_motor_bridge
{
// Report a drive refusing a request, keeping its code so callers can check it
class SdoAbort : public std::runtime_error
{
public:
  SdoAbort(const std::string & message, std::uint32_t code);
  // Return the abort code the drive sent
  std::uint32_t code() const;

private:
  std::uint32_t code_;
};

// Read and write one drive's object dictionary over a shared bus
class SdoClient
{
public:
  // Address one CANopen node (1 to 127), or throw
  SdoClient(const CanBus & bus, int node_id);

  // Read one value of up to 4 bytes, or throw on a timeout or refusal
  std::uint32_t read(std::uint16_t index, std::uint8_t subindex) const;
  // Write one value of 1, 2 or 4 bytes, or throw on a timeout or refusal
  void write(
    std::uint16_t index, std::uint8_t subindex, std::uint32_t value, std::size_t size_bytes) const;

private:
  // Send a request and return the reply about the same object, or throw
  can_frame exchange(const can_frame & request, const std::string & action) const;

  // Use the bus without owning it, because every drive shares one bus
  const CanBus & bus_;
  // Store the CAN IDs derived from the node ID
  canid_t request_id_;
  canid_t response_id_;
};
}  // namespace agrobot_motor_bridge
