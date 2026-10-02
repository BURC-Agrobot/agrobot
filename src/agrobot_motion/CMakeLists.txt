# Require the CMake features used by this ROS package
cmake_minimum_required(VERSION 3.16)
# Enable C and C++ for generated ROS interfaces and motion code
project(agrobot_motion LANGUAGES C CXX)

# Require C++17 so language features remain consistent across targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Export compiler commands for code analysis tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Load build tools and ROS dependencies before defining targets
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)

# Generate the motion action interface for ROS clients and servers
rosidl_generate_interfaces(${PROJECT_NAME} "action/MoveJoints.action")
# Resolve C++ message support for the generated action
rosidl_get_typesupport_target(typesupport ${PROJECT_NAME} rosidl_typesupport_cpp)

# Compile shared validation rules for the motion node
add_library(motion_rules src/motion_rules.cpp)
# Expose rule declarations to targets that use the library
target_include_directories(motion_rules PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# Attach model and message dependencies needed by validation
ament_target_dependencies(motion_rules
  rclcpp controller_manager_msgs sensor_msgs trajectory_msgs moveit_core)
# Link generated action types into the validation library
target_link_libraries(motion_rules "${typesupport}")

# Build the executable that owns planning and mock execution
add_executable(motion_node src/motion_node.cpp)
# Attach action and planning dependencies to the motion node
ament_target_dependencies(motion_node rclcpp_action control_msgs moveit_msgs moveit_ros_planning)
# Link the shared validation rules into the motion executable
target_link_libraries(motion_node motion_rules)

# Collect targets that must share compiler and static-analysis checks
set(motion_targets motion_rules motion_node)
# Require the analysis tools used by test-enabled builds
if(BUILD_TESTING)
  # Require clang-tidy so configured C++ checks cannot be skipped
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  # Require mypy so launch typing is checked in test builds
  find_program(MYPY NAMES mypy REQUIRED)
  # Check the launch file with strict Python typing
  add_test(NAME python_strict COMMAND "${MYPY}" --strict
    "${CMAKE_CURRENT_SOURCE_DIR}/launch/mock.launch.py")
endif()

# Apply the same C++ checks to every motion target
foreach(target IN LISTS motion_targets)
  # Treat compiler warnings and unsafe conversions as build failures
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror)
  # Run selected clang-tidy checks while compiling test-enabled builds
  if(BUILD_TESTING)
    set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy")
  endif()
endforeach()

# Install the motion executable and its compiled validation library
install(TARGETS motion_rules motion_node
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
# Install the headless mock launch for ROS package discovery
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
# Export interface runtime support for downstream action clients
ament_export_dependencies(rosidl_default_runtime)
# Register the package with the ROS build and discovery system
ament_package()
