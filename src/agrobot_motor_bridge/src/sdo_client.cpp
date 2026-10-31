#include "agrobot_motor_bridge/sdo_client.hpp"

#include <chrono>
#include <cstdio>
#include <optional>

namespace agrobot_motor_bridge
{
namespace
{
// Give a drive time to answer without stalling on a silent one
constexpr std::chrono::milliseconds kResponseTimeout{500};

// Name the command bytes from the CANopen SDO protocol
constexpr std::uint8_t kReadRequest = 0x40;
constexpr std::uint8_t kReadReplyMask = 0xE0;
constexpr std::uint8_t kReadReply = 0x40;
constexpr std::uint8_t kExpeditedBit = 0x02;
constexpr std::uint8_t kSizeStatedBit = 0x01;
constexpr std::uint8_t kWrite1Byte = 0x2F;
constexpr std::uint8_t kWrite2Bytes = 0x2B;
constexpr std::uint8_t kWrite4Bytes = 0x23;
constexpr std::uint8_t kWriteReply = 0x60;
constexpr std::uint8_t kAbort = 0x80;

// Format an object address the way the drive manuals write it
std::string describe(std::uint16_t index, std::uint8_t subindex)
{
  char text[32];
  std::snprintf(text, sizeof(text), "0x%04X sub 0x%02X", index, subindex);
  return text;
}

// Explain the abort codes this driver is likely to meet
std::string abort_meaning(std::uint32_t code)
{
  switch (code) {
    case 0x05040001U: return "unknown command";
    case 0x06010002U: return "object is read-only";
    case 0x06020000U: return "object does not exist";
    case 0x06070010U: return "size does not match the object";
    case 0x06090011U: return "subindex does not exist";
    case 0x06090030U: return "value is out of range";
    default: return "see the drive's abort code table";
  }
}

// Join 4 data bytes, sent low byte first, starting at byte 4 of a frame
std::uint32_t payload(const can_frame & frame)
{
  return static_cast<std::uint32_t>(frame.data[4]) |
         (static_cast<std::uint32_t>(frame.data[5]) << 8U) |
         (static_cast<std::uint32_t>(frame.data[6]) << 16U) |
         (static_cast<std::uint32_t>(frame.data[7]) << 24U);
}

// Start a request frame, sending the index low byte first
can_frame make_request(
  canid_t can_id, std::uint8_t command, std::uint16_t index, std::uint8_t subindex)
{
  can_frame request{};
  request.can_id = can_id;
  request.len = CAN_MAX_DLEN;
  request.data[0] = command;
  request.data[1] = static_cast<std::uint8_t>(index & 0xFFU);
  request.data[2] = static_cast<std::uint8_t>(index >> 8U);
  request.data[3] = subindex;
  return request;
}

// Tell the reply to a request apart from other traffic, including replies about other objects
bool is_reply_to(const can_frame & frame, const can_frame & request, canid_t response_id)
{
  return frame.can_id == response_id && frame.len == CAN_MAX_DLEN &&
         frame.data[1] == request.data[1] && frame.data[2] == request.data[2] &&
         frame.data[3] == request.data[3];
}
}  // namespace

SdoAbort::SdoAbort(const std::string & message, std::uint32_t code)
: std::runtime_error(message), code_(code)
{
}

std::uint32_t SdoAbort::code() const
{
  return code_;
}

SdoClient::SdoClient(const CanBus & bus, int node_id)
: bus_(bus)
{
  if (node_id < 1 || node_id > 127) {
    throw std::invalid_argument("Node ID must be 1 to 127, got " + std::to_string(node_id));
  }
  request_id_ = 0x600U + static_cast<canid_t>(node_id);
  response_id_ = 0x580U + static_cast<canid_t>(node_id);
}

can_frame SdoClient::exchange(const can_frame & request, const std::string & action) const
{
  bus_.send(request);

  // Keep one deadline so other traffic cannot extend the wait
  const auto deadline = std::chrono::steady_clock::now() + kResponseTimeout;
  while (true) {
    const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now());
    const std::optional<can_frame> frame =
      remaining.count() > 0 ? bus_.receive(remaining) : std::nullopt;
    if (!frame) {
      throw std::runtime_error(action + " timed out");
    }

    if (!is_reply_to(*frame, request, response_id_)) {
      continue;
    }

    // Report a refusal the same way for reads and writes
    if (frame->data[0] == kAbort) {
      const std::uint32_t abort_code = payload(*frame);
      char code_text[16];
      std::snprintf(code_text, sizeof(code_text), "0x%08X", abort_code);
      throw SdoAbort(
        action + " aborted with " + code_text + " (" + abort_meaning(abort_code) + ")", abort_code);
    }
    return *frame;
  }
}

std::uint32_t SdoClient::read(std::uint16_t index, std::uint8_t subindex) const
{
  const std::string action = "SDO read of " + describe(index, subindex);
  const can_frame reply =
    exchange(make_request(request_id_, kReadRequest, index, subindex), action);

  const std::uint8_t command = reply.data[0];
  if ((command & kReadReplyMask) != kReadReply || (command & kExpeditedBit) == 0) {
    throw std::runtime_error(action + " got an unsupported reply");
  }

  // Bits 2 and 3 count the unused bytes, when the drive states a size
  const std::uint32_t value = payload(reply);
  if ((command & kSizeStatedBit) == 0) {
    return value;
  }
  const unsigned int unused_bytes = (static_cast<unsigned int>(command) >> 2U) & 0x3U;
  return value & (0xFFFFFFFFU >> (unused_bytes * 8U));
}

void SdoClient::write(
  std::uint16_t index, std::uint8_t subindex, std::uint32_t value, std::size_t size_bytes) const
{
  const std::string action = "SDO write of " + describe(index, subindex);

  // Choose the command for this size, and refuse a value that would be cut short
  std::uint8_t command = 0;
  std::uint32_t largest = 0;
  switch (size_bytes) {
    case 1: command = kWrite1Byte; largest = 0xFFU; break;
    case 2: command = kWrite2Bytes; largest = 0xFFFFU; break;
    case 4: command = kWrite4Bytes; largest = 0xFFFFFFFFU; break;
    default: throw std::invalid_argument(action + " needs a size of 1, 2 or 4 bytes");
  }
  if (value > largest) {
    throw std::invalid_argument(action + " has a value too large for its size");
  }

  // Send the value low byte first, leaving unused bytes at zero
  can_frame request = make_request(request_id_, command, index, subindex);
  for (std::size_t byte = 0; byte < size_bytes; ++byte) {
    request.data[4 + byte] = static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xFFU);
  }

  const can_frame reply = exchange(request, action);
  if (reply.data[0] != kWriteReply) {
    throw std::runtime_error(action + " got an unsupported reply");
  }
}
}  // namespace agrobot_motor_bridge
