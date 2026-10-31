# Require the CMake features used by this ROS package
cmake_minimum_required(VERSION 3.16)
# Enable C++ for the driver code
project(agrobot_motor_bridge LANGUAGES CXX)

# Require C++17 so language features remain consistent across targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Export compiler commands for code analysis tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Load build tools before defining targets
find_package(ament_cmake REQUIRED)

# Compile the socket layer shared by the driver programs
add_library(can_bus src/can_bus.cpp)
# Expose the bus declarations to targets that use the library
target_include_directories(can_bus PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# Compile the CANopen SDO layer on top of the socket layer
add_library(sdo_client src/sdo_client.cpp)
target_link_libraries(sdo_client PUBLIC can_bus)
# Compile the drive state layer on top of the SDO layer
add_library(drive_control src/drive_control.cpp)
target_link_libraries(drive_control PUBLIC sdo_client)

# Build the programs that exercise one layer each
add_executable(send_frame src/send_frame.cpp)
target_link_libraries(send_frame can_bus)
add_executable(receive_frame src/receive_frame.cpp)
target_link_libraries(receive_frame can_bus)
add_executable(read_device_type src/read_device_type.cpp)
target_link_libraries(read_device_type drive_control)
add_executable(check_sdo_write src/check_sdo_write.cpp)
target_link_libraries(check_sdo_write drive_control)
add_executable(walk_drive_states src/walk_drive_states.cpp)
target_link_libraries(walk_drive_states drive_control)
add_executable(move_drive src/move_drive.cpp)
target_link_libraries(move_drive drive_control)

# Collect targets that must share compiler and static-analysis checks
set(bridge_targets can_bus sdo_client drive_control
  send_frame receive_frame read_device_type check_sdo_write walk_drive_states move_drive)

# Require clang-tidy to enforce the configured C++ checks
if(BUILD_TESTING)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
endif()

# Apply the same C++ checks to every driver target
foreach(target IN LISTS bridge_targets)
  # Treat compiler warnings and unsafe conversions as build failures
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror)
  # Run selected clang-tidy checks while compiling test-enabled builds
  if(BUILD_TESTING)
    set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy")
  endif()
endforeach()

# Install the programs and their compiled libraries
install(TARGETS ${bridge_targets}
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
# Register the package with the ROS build and discovery system
ament_package()
