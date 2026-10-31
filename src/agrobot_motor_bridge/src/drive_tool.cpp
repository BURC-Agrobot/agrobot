#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "agrobot_motor_bridge/can_bus.hpp"
#include "agrobot_motor_bridge/drive_control.hpp"
#include "agrobot_motor_bridge/sdo_client.hpp"

namespace
{
using agrobot_motor_bridge::CanBus;
using agrobot_motor_bridge::DriveControl;
using agrobot_motor_bridge::SdoClient;

// Hold the command line after the command name: interface, node, then target
using Arguments = std::vector<std::string>;

// Name the objects these commands use
constexpr std::uint16_t kDeviceType = 0x1000;
constexpr std::uint16_t kControlword = 0x6040;
constexpr std::uint16_t kStatusword = 0x6041;
// Shutdown leaves the motor without power, and Disable Voltage returns to the start
constexpr std::uint32_t kShutdown = 0x0006;
constexpr std::uint32_t kDisableVoltage = 0x0000;
// Expect this code when writing a read-only object
constexpr std::uint32_t kReadOnlyAbort = 0x06010002;

// Give a person time to send a frame from another terminal
constexpr std::chrono::seconds kReceiveTimeout{5};

// Use the profile and target from the emulator's startup sequence
constexpr std::uint32_t kVelocityRpm = 1000;
constexpr std::uint32_t kAccelerationRpmS = 10000;
constexpr std::uint32_t kDecelerationRpmS = 10000;
constexpr std::int32_t kDefaultTargetCounts = 20000;

// Use the virtual bus unless another interface is named
std::string interface_name(const Arguments & arguments)
{
  return arguments.empty() ? "vcan0" : arguments[0];
}

// Use the emulator's node unless another is named
int node_id(const Arguments & arguments)
{
  return arguments.size() > 1 ? std::stoi(arguments[1]) : 1;
}

// Hold the layers needed to talk to one drive, which must outlive each other in this order
struct DriveLink
{
  explicit DriveLink(const Arguments & arguments)
  : bus(interface_name(arguments)), sdo(bus, node_id(arguments)), drive(sdo)
  {
  }

  CanBus bus;
  SdoClient sdo;
  DriveControl drive;
};

// Print the drive's current state
void print_state(const DriveControl & drive)
{
  const std::uint16_t status = drive.statusword();
  std::printf(
    "[drive_tool] 0x%04X %s\n", status,
    agrobot_motor_bridge::state_name(agrobot_motor_bridge::decode_state(status)).c_str());
}

// Run steps that power the motor from a clean start, taking power off if one fails
template<typename Steps>
void run_powered(const Arguments & arguments, Steps steps)
{
  // Refuse real buses, because these commands power the motor with no way to stop it
  const std::string name = interface_name(arguments);
  if (name.rfind("vcan", 0) != 0) {
    throw std::invalid_argument("This command only runs on virtual buses, not " + name);
  }

  const DriveLink link(arguments);
  const DriveControl & drive = link.drive;
  drive.require_switch_on_disabled();
  print_state(drive);

  try {
    steps(drive);
  } catch (const std::exception &) {
    // Keep the first error as the one reported
    try {
      drive.disable_voltage();
    } catch (const std::exception & error) {
      std::fprintf(stderr, "[drive_tool] ERROR: Could not disable the drive: %s\n", error.what());
    }
    throw;
  }
}

// Send one frame without waiting for a reply
void send(const Arguments & arguments)
{
  // Ask node 1 for its Statusword (0x6041), which changes nothing on the drive
  can_frame frame{};
  frame.can_id = 0x601;
  frame.len = CAN_MAX_DLEN;
  frame.data[0] = 0x40;
  frame.data[1] = 0x41;
  frame.data[2] = 0x60;

  const std::string name = interface_name(arguments);
  const CanBus bus(name);
  bus.send(frame);
  std::printf("[drive_tool] Sent 601#4041600000000000 on %s\n", name.c_str());
}

// Wait for one frame and print it in the same layout as candump
void receive(const Arguments & arguments)
{
  const std::string name = interface_name(arguments);
  const CanBus bus(name);
  std::printf(
    "[drive_tool] Waiting %lld s on %s\n", static_cast<long long>(kReceiveTimeout.count()),
    name.c_str());
  // Show the prompt now, because the wait can be long
  std::fflush(stdout);

  const std::optional<can_frame> frame = bus.receive(kReceiveTimeout);
  if (!frame) {
    // Report silence as a failure so scripts can detect it
    throw std::runtime_error("Timed out with no frame");
  }
  std::printf("[drive_tool] %03X  [%u] ", frame->can_id & CAN_EFF_MASK, frame->len);
  for (unsigned int i = 0; i < frame->len; ++i) {
    std::printf(" %02X", frame->data[i]);
  }
  std::printf("\n");
}

// Read Device Type, which every CANopen device must have
void read_type(const Arguments & arguments)
{
  const DriveLink link(arguments);
  std::printf(
    "[drive_tool] Node %d Device Type = 0x%08X\n", node_id(arguments),
    link.sdo.read(kDeviceType, 0x00));
}

// Check that the drive stores a valid write and refuses a read-only one
void check_write(const Arguments & arguments)
{
  const DriveLink link(arguments);
  const SdoClient & sdo = link.sdo;

  // Write Controlword, then read it back to prove the value was stored
  sdo.write(kControlword, 0x00, kShutdown, 2);
  const std::uint32_t stored = sdo.read(kControlword, 0x00);
  std::printf("[drive_tool] Controlword = 0x%04X\n", stored);
  if (stored != kShutdown) {
    throw std::runtime_error("Controlword did not store the written value");
  }
  std::printf("[drive_tool] Statusword = 0x%04X\n", sdo.read(kStatusword, 0x00));

  // Write read-only Statusword, which the drive must refuse
  bool refused = false;
  try {
    sdo.write(kStatusword, 0x00, 0x0001, 2);
  } catch (const agrobot_motor_bridge::SdoAbort & abort) {
    std::printf("[drive_tool] Refused as expected: %s\n", abort.what());
    refused = abort.code() == kReadOnlyAbort;
  }

  // Return the drive to where a fresh start leaves it
  sdo.write(kControlword, 0x00, kDisableVoltage, 2);
  if (!refused) {
    throw std::runtime_error("Statusword write was not refused as read-only");
  }
}

// Walk up to Operation Enable and back down
void walk(const Arguments & arguments)
{
  run_powered(arguments, [](const DriveControl & drive) {
    drive.shutdown();
    print_state(drive);
    drive.switch_on();
    print_state(drive);
    drive.enable_operation();
    print_state(drive);
    drive.disable_voltage();
    print_state(drive);
  });
}

// Enable the drive, make one move, and take power off again
void move(const Arguments & arguments)
{
  const std::int32_t target_counts =
    arguments.size() > 2 ? std::stoi(arguments[2]) : kDefaultTargetCounts;

  run_powered(arguments, [target_counts](const DriveControl & drive) {
    drive.set_profile_position_mode(kVelocityRpm, kAccelerationRpmS, kDecelerationRpmS);
    drive.shutdown();
    drive.switch_on();
    drive.enable_operation();
    print_state(drive);

    std::printf("[drive_tool] Moving to %d\n", target_counts);
    drive.move_to(target_counts);
    const std::int32_t position = drive.position_counts();
    std::printf("[drive_tool] Position = %d\n", position);
    if (position != target_counts) {
      throw std::runtime_error("Drive stopped away from the target");
    }

    drive.disable_voltage();
    print_state(drive);
  });
}

// Describe one command in a single place, so the usage text cannot drift from it
struct Command
{
  const char * name;
  const char * help;
  void (* run)(const Arguments &);
};

constexpr std::array kCommands{
  Command{"send", "Send a Statusword read request to node 1", send},
  Command{"receive", "Wait up to 5 s for one frame", receive},
  Command{"read-type", "Read a node's Device Type", read_type},
  Command{"check-write", "Check that a node stores writes and refuses read-only ones", check_write},
  Command{"walk", "Walk a node through its power states (vcan only)", walk},
  Command{"move", "Move a node to a target in quadcounts (vcan only)", move},
};

// Explain the commands when none is chosen correctly
void print_usage()
{
  std::fprintf(stderr, "Usage: drive_tool <command> [interface] [node] [target]\n");
  for (const Command & command : kCommands) {
    std::fprintf(stderr, "  %-12s %s\n", command.name, command.help);
  }
}
}  // namespace

int main(int argc, char ** argv)
{
  const std::string chosen = argc > 1 ? argv[1] : "";
  for (const Command & command : kCommands) {
    if (chosen == command.name) {
      try {
        command.run(Arguments(argv + 2, argv + argc));
        return 0;
      } catch (const std::exception & error) {
        // Report the failed call instead of ending with an uncaught exception
        std::fprintf(stderr, "[drive_tool] ERROR: %s\n", error.what());
        return 1;
      }
    }
  }

  print_usage();
  return 1;
}
