cmake_minimum_required(VERSION 3.5)
project(sbg_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)


set(msg_files
  "msg/SbgCAN310.msg"
  "msg/SbgCAN292.msg"
  "msg/SbgCAN311.msg"
  "msg/SbgCAN518.msg"
  "msg/SbgCAN313.msg"
  "msg/SbgCAN376.msg"
  "msg/SbgCAN338.msg"
  "msg/SbgCAN273.msg"
  "msg/SbgCAN369.msg"
  "msg/SbgCAN373.msg"
  "msg/SbgCAN368.msg"
  "msg/SbgCAN272.msg"
  "msg/SbgCAN290.msg"
  "msg/SbgCAN291.msg"
  "msg/SbgCAN336.msg"
  "msg/SbgCAN371.msg"
  "msg/SbgCAN307.msg"
  "msg/SbgCAN355.msg"
  "msg/SbgCAN514.msg"
  "msg/SbgCAN289.msg"
  "msg/SbgCAN258.msg"
  "msg/SbgCAN372.msg"
  "msg/SbgCAN519.msg"
  "msg/SbgCAN308.msg"
  "msg/SbgCAN288.msg"
  "msg/SbgCAN337.msg"
  "msg/SbgCAN513.msg"
  "msg/SbgCAN352.msg"
  "msg/SbgCAN312.msg"
  "msg/SbgCAN257.msg"
  "msg/SbgCAN375.msg"
  "msg/SbgCAN306.msg"
  "msg/SbgCAN512.msg"
  "msg/SbgCAN353.msg"
  "msg/SbgCAN354.msg"
  "msg/SbgCAN256.msg"
  "msg/SbgCAN544.msg"
  "msg/SbgCAN370.msg"
  "msg/SbgCAN309.msg"
  "msg/SbgCAN304.msg"
  "msg/SbgCAN515.msg"
  "msg/SbgCAN374.msg"
  "msg/SbgCAN377.msg"
  "msg/SbgCAN305.msg"
  "msg/SbgCAN356.msg"
  "msg/SbgCAN768.msg"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()


rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
